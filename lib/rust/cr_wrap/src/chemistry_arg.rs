use anyhow::{bail, Result};
use cr_types::chemistry::{AutoChemistryName, AutoOrRefinedChemistry, ChemistryName};
use itertools::Itertools;
#[allow(clippy::enum_glob_use)]
use AutoChemistryName::*;
#[allow(clippy::enum_glob_use)]
use AutoOrRefinedChemistry::*;
#[allow(clippy::enum_glob_use)]
use ChemistryName::*;

const ALLOWED_COUNT_CHEM_INPUTS: [(AutoOrRefinedChemistry, Option<&str>); 19] = [
    (Auto(Count), Some("auto detection (default)")),
    (Auto(ThreePrime), Some("Single Cell 3'")),
    (Auto(FivePrime), Some("Single Cell 5'")),
    (Refined(ThreePrimeV1), Some("Single Cell 3'v1")),
    (Refined(ThreePrimeV2), Some("Single Cell 3'v2")),
    (Refined(ThreePrimeV3), Some("Single Cell 3'v3")),
    (Refined(ThreePrimeV3HT), Some("Single Cell 3'v3 HT")),
    (Refined(ThreePrimeV4), Some("Single Cell 3'v4")),
    (Refined(ThreePrimeV4OH), Some("Single Cell 3'v4 OH")),
    (Refined(FivePrimePE), Some("Single Cell 5' paired end")),
    (Refined(FivePrimePEV3), Some("Single Cell 5' paired end v3")),
    (Refined(FivePrimeR2), Some("Single Cell 5' R2-only")),
    (Refined(FivePrimeR2V3), Some("Single Cell 5' R2-only v3")),
    (
        Refined(FivePrimeR2OHV3),
        Some("Single Cell 5' R2-only OH v3"),
    ),
    (Refined(FivePrimeHT), None),
    (Refined(FivePrimeR2), None),
    (
        Refined(FeatureBarcodingOnly),
        Some("Single Cell Antibody-only 3' v2 or 5'"),
    ),
    (Refined(SFRP), None),
    (Refined(ArcV1), Some("GEX portion only of multiome")),
];

/// Parse the provided chemistry and validate that it is of an allowed type.
/// Return a user-facing clap-compatible error message.
pub fn validate_chemistry(s: &str) -> Result<AutoOrRefinedChemistry> {
    let parse_err = || {
        format!(
            "{s} is an invalid input to `--chemistry`. Supported options are:\n - {}",
            ALLOWED_COUNT_CHEM_INPUTS
                .iter()
                .filter_map(|(chem, help)| help.as_ref().map(|h| format!("{chem} for {h}")))
                .join("\n - ")
        )
    };
    let Some(chem) = s.parse().ok() else {
        bail!(parse_err());
    };
    if ALLOWED_COUNT_CHEM_INPUTS
        .iter()
        .any(|(option, _)| *option == chem)
    {
        return Ok(chem);
    }
    if chem.refined() == Some(ThreePrimeV3LT) {
        bail!("The chemistry SC3Pv3LT (Single Cell 3'v3 LT) is no longer supported. To analyze this data, use Cell Ranger 7.2 or earlier.");
    }
    bail!(parse_err());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_count_chemistry_arg() {
        for chem in ["SC3Pv3", "SC3Pv3HT", "auto", "SC5P-R2", "SC5PHT", "ARC-v1"] {
            assert_eq!(
                validate_chemistry(chem).unwrap().to_string(),
                chem.to_string()
            );
        }
        assert!(validate_chemistry("SCVDJ").is_err());
        assert!(validate_chemistry("SC3Pv3LT").is_err());
    }
}
