//! SetupBinning stage code

use crate::square_bin_name::SquareBinName;
use itertools::Itertools;
use martian::prelude::*;
use martian_derive::{make_mro, MartianStruct};
use metric::TxHashMap;
use serde::{Deserialize, Serialize};
use slide_design::VisiumHdSlide;

#[derive(Debug, Clone, Deserialize, MartianStruct)]
pub struct SetupBinningStageInputs {
    slide_name: Option<String>,
    no_secondary_analysis: Option<bool>,
    scales: Vec<u32>,
    custom_bin_size: Option<u32>,
}
#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
pub struct BinLevelInfo {
    scale: u32,
    no_secondary_analysis: bool,
    disable_cloupe: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, MartianStruct)]
pub struct SetupBinningStageOutputs {
    bin_infos: TxHashMap<SquareBinName, BinLevelInfo>,
    disable_binning: bool,
}

// This is our stage struct
pub struct SetupBinning;

const VISIUM_HD_PITCH: f32 = 2.0;
const BIN_8UM_SCALE: u32 = 4;

#[make_mro(mem_gb = 2, volatile = strict, stage_name = SETUP_BINNING)]
impl MartianMain for SetupBinning {
    type StageInputs = SetupBinningStageInputs;
    type StageOutputs = SetupBinningStageOutputs;
    fn main(
        &self,
        args: Self::StageInputs,
        _rover: MartianRover,
    ) -> Result<Self::StageOutputs, Error> {
        let default_return = Ok(SetupBinningStageOutputs {
            bin_infos: TxHashMap::default(),
            disable_binning: true,
        });

        match args.slide_name {
            Some(slide_name) => {
                // Pull the pitch from the slide design file
                let slide = VisiumHdSlide::from_name_and_layout(&slide_name, None)?;
                let pitch = slide.spot_pitch();
                assert!(pitch.fract() == 0.0);

                let custom_bin_scale = args.custom_bin_size.map(|x| x / pitch as u32);

                if pitch == VISIUM_HD_PITCH {
                    Ok(SetupBinningStageOutputs {
                        bin_infos: args
                            .scales
                            .into_iter()
                            .chain(custom_bin_scale)
                            .unique()
                            .map(|scale| {
                                SquareBinName::new(scale * (pitch as u32)).map(|bin_name| {
                                    (
                                        bin_name,
                                        BinLevelInfo {
                                            scale,
                                            // Disable secondary analysis if either it is globally disabled
                                            // of if the bin scale is <4 (under 8 µm for HD)
                                            no_secondary_analysis: args
                                                .no_secondary_analysis
                                                .unwrap_or_default()
                                                || scale < 4,
                                            disable_cloupe: (scale != BIN_8UM_SCALE)
                                                && !((Some(scale) == custom_bin_scale)
                                                    && (scale > BIN_8UM_SCALE)),
                                        },
                                    )
                                })
                            })
                            .try_collect()?,
                        disable_binning: false,
                    })
                } else {
                    default_return
                }
            }
            _ => default_return,
        }
    }
}
