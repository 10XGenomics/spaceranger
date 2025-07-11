#![deny(missing_docs)]
use super::chemistry_filter::{ChemistryFilter, DetectChemistryUnit};
use super::errors::DetectChemistryErrors;
use anyhow::Result;
use barcode::{BarcodeConstruct, BcSegSeq, Whitelist, WhitelistSource, WhitelistSpec};
use cr_types::chemistry::{BarcodeExtraction, BarcodeReadComponent, ChemistryDef, ChemistryName};
use cr_types::rna_read::extract_barcode;
use fastq_set::read_pair::ReadPair;
use itertools::Itertools;
use metric::{set, PercentMetric, TxHashMap, TxHashSet};
use parameters_toml::min_fraction_whitelist_match;
use std::ops::Range;

pub(crate) struct WhitelistMatchFilter<'a> {
    allowed_chems: &'a TxHashSet<ChemistryName>,
    chem_defs: Vec<ChemistryDef>,
    index_of_chem: TxHashMap<ChemistryName, usize>,
    // One for each unique whitelist
    wl_matchers: Vec<WhitelistMatcher>,
}

impl<'a> WhitelistMatchFilter<'a> {
    pub(crate) fn new(
        allowed_chems: &'a TxHashSet<ChemistryName>,
        chems: &TxHashSet<ChemistryName>,
    ) -> Result<Self> {
        let chem_defs: Vec<_> = chems
            .iter()
            .map(|&chem| ChemistryDef::named(chem))
            .collect();

        let mut index_map = TxHashMap::default();
        let mut wl_matchers = Vec::new();
        for wl in chem_defs
            .iter()
            .map(ChemistryDef::barcode_whitelist_spec)
            .unique()
        {
            index_map.insert(wl, wl_matchers.len());
            wl_matchers.push(WhitelistMatcher::new(
                wl.map_result(WhitelistSpec::as_source)?,
            )?);
        }
        let index_of_chem = chem_defs
            .iter()
            .map(|def| (def.name, index_map[&def.barcode_whitelist_spec()]))
            .collect();

        Ok(WhitelistMatchFilter {
            allowed_chems,
            chem_defs,
            wl_matchers,
            index_of_chem,
        })
    }
}

impl<'a> ChemistryFilter<'a> for WhitelistMatchFilter<'a> {
    fn context() -> &'static str {
        "Whitelist based filtering"
    }
    fn allowed_chemistries(&self) -> &'a TxHashSet<ChemistryName> {
        self.allowed_chems
    }
    fn input_chemistries(&self) -> TxHashSet<ChemistryName> {
        self.chem_defs.iter().map(|def| def.name).collect()
    }
    fn process_unit(
        &mut self,
        unit: &DetectChemistryUnit,
        reads: &[ReadPair],
    ) -> Result<TxHashSet<ChemistryName>, DetectChemistryErrors> {
        // Special case for HD CELLRANGER-7761
        if self.chem_defs.len() == 1 && self.chem_defs[0].name.is_spatial() {
            return Ok(set![self.chem_defs[0].name]);
        }

        let min_frac_whitelist_match = *min_fraction_whitelist_match()
            .map_err(DetectChemistryErrors::MissingParametersToml)?;
        let mut all_wl_matches = TxHashMap::default();
        let mut best_wl_match = 0.0f64;
        for chem_def in &self.chem_defs {
            let wl_matches = self.wl_matchers[self.index_of_chem[&chem_def.name]]
                .count_matches(
                    reads,
                    chem_def.barcode_construct(),
                    chem_def.barcode_extraction(),
                )
                .fraction();
            // TODO: Handle reads_with_bc = 0
            if wl_matches > best_wl_match {
                best_wl_match = wl_matches;
            }
            all_wl_matches.insert(chem_def.name, wl_matches);
        }
        println!("{all_wl_matches:#?}");
        let sufficient_matches: TxHashMap<_, _> = all_wl_matches
            .iter()
            .filter(|(_, &v)| v > min_frac_whitelist_match)
            .collect();
        if sufficient_matches.is_empty() {
            return Err(DetectChemistryErrors::NotEnoughWhitelistMatch {
                frac_matches: all_wl_matches.clone(),
                unit: Box::new(unit.clone()),
            });
        }

        let mut compatible_chems: TxHashSet<ChemistryName> = sufficient_matches
            .iter()
            .filter_map(|(&&chem, &&wl_matches)| {
                (wl_matches > 0.75f64 * best_wl_match).then_some(chem)
            })
            .collect();

        // Prefer the more specific chemistry when MFRP_47 and at least one other more specific
        // MFRP chemistry are compatible.
        if compatible_chems
            .iter()
            .any(|chem| matches!(chem, ChemistryName::MFRP_RNA | ChemistryName::MFRP_Ab))
        {
            compatible_chems.remove(&ChemistryName::MFRP_47);
        }

        // Use the chemistry with more valid barcodes when both R1 and R2 chemistries are compatible.
        let select_superior_matches_of_pair = [
            (ChemistryName::MFRP_RNA, ChemistryName::MFRP_RNA_R1),
            (ChemistryName::MFRP_Ab, ChemistryName::MFRP_Ab_R1),
            (ChemistryName::MFRP_Ab_R2pos50, ChemistryName::MFRP_Ab_R1),
        ];
        for (r2_chem, r1_chem) in select_superior_matches_of_pair {
            if compatible_chems.is_superset(&set![r2_chem, r1_chem]) {
                let inferior_chemistry =
                    if sufficient_matches[&r2_chem] < sufficient_matches[&r1_chem] {
                        r2_chem
                    } else {
                        r1_chem
                    };
                assert!(compatible_chems.remove(&inferior_chemistry));
            }
        }

        Ok(compatible_chems)
    }
}

#[derive(Default)]
struct WhitelistMatchStats {
    total_reads: i64,
    reads_with_bc: i64,
    reads_with_bc_in_wl: i64,
}

impl WhitelistMatchStats {
    fn fraction(self) -> f64 {
        PercentMetric::from((self.reads_with_bc_in_wl, self.reads_with_bc))
            .fraction()
            .unwrap_or(0.0f64) // Return 0 if the denominator is 0
    }
}

pub struct WhitelistMatcher {
    whitelist: BarcodeConstruct<Whitelist>,
    barcode_lengths: BarcodeConstruct<Range<usize>>,
}

impl WhitelistMatcher {
    pub fn new(whitelist: BarcodeConstruct<WhitelistSource>) -> Result<Self> {
        // Load this whitelist as a "plain" whitelist, regardless of type.
        //
        // This is sufficient for checking membership in the whitelist, but neglects
        // all translation information. Using this method instead of as_whitelist
        // reduces memory consumption since we are working in a context where
        // we only care about matching to the whitelist, but not using translation.
        let whitelist = whitelist.map_result(|source| Ok(Whitelist::Plain(source.as_set()?)))?;
        Ok(WhitelistMatcher {
            barcode_lengths: whitelist.as_ref().map(Whitelist::sequence_lengths),
            whitelist,
        })
    }

    fn count_matches(
        &self,
        read_pairs: &[ReadPair],
        barcode_components: BarcodeConstruct<&BarcodeReadComponent>,
        barcode_extraction: Option<&BarcodeExtraction>,
    ) -> WhitelistMatchStats {
        let mut stats = WhitelistMatchStats::default();
        for read_pair in read_pairs {
            stats.total_reads += 1;

            if let Ok((_bc_ranges, bc)) = extract_barcode(
                read_pair,
                barcode_components,
                barcode_extraction,
                self.whitelist.as_ref(),
                self.barcode_lengths.as_ref(),
                0,
            ) {
                stats.reads_with_bc += 1;
                if self.contains(bc.sequence()) {
                    stats.reads_with_bc_in_wl += 1;
                }
            }
        }
        stats
    }

    fn contains(&self, sseq: BarcodeConstruct<&[u8]>) -> bool {
        // NOTE: This is robust to a single N cycle
        self.whitelist
            .as_ref()
            .zip(sseq)
            .map_option(|(wl, seq)| {
                wl.match_to_whitelist_allow_one_n(BcSegSeq::from_bytes(seq), false)
            })
            .is_some()
    }
}
