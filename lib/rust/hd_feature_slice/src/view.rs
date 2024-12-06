use crate::scaling::{Normalize, Scaling};
use image::{DynamicImage, GrayImage, RgbImage, RgbaImage};
use itertools::{izip, zip_eq};
use ndarray::Array2;

fn u8_from_normalized_counts(norm_count: f64) -> u8 {
    (255.0 * norm_count) as u8
}

#[derive(Clone, Copy)]
pub enum CompositeChannel {
    Red,
    Green,
    Blue,
    Cyan,
    Magenta,
    Yellow,
    Black,
    White,
}

impl CompositeChannel {
    fn rgb_pixel(self) -> [u8; 3] {
        match self {
            CompositeChannel::Red => [255, 0, 0],
            CompositeChannel::Green => [0, 255, 0],
            CompositeChannel::Blue => [0, 0, 255],
            CompositeChannel::Cyan => [0, 255, 255],
            CompositeChannel::Magenta => [255, 0, 255],
            CompositeChannel::Yellow => [255, 255, 0],
            CompositeChannel::Black => [0, 0, 0],
            CompositeChannel::White => [255, 255, 255],
        }
    }
}

#[derive(Clone, Copy, Default)]
pub enum CompositionRule {
    #[default]
    Zero,
    Scaled,
}

#[derive(Clone, Copy, Default)]
pub struct RgbComposition {
    r: CompositionRule,
    g: CompositionRule,
    b: CompositionRule,
}

impl From<CompositeChannel> for RgbComposition {
    fn from(channel: CompositeChannel) -> Self {
        use CompositionRule::Scaled;
        match channel {
            CompositeChannel::Red => RgbComposition::new().r(Scaled),
            CompositeChannel::Blue => RgbComposition::new().b(Scaled),
            CompositeChannel::Green => RgbComposition::new().g(Scaled),
            CompositeChannel::Cyan => RgbComposition::new().g(Scaled).b(Scaled),
            CompositeChannel::Magenta => RgbComposition::new().r(Scaled).b(Scaled),
            CompositeChannel::Yellow => RgbComposition::new().r(Scaled).g(Scaled),
            CompositeChannel::Black => RgbComposition::new(),
            CompositeChannel::White => RgbComposition::new().r(Scaled).g(Scaled).b(Scaled),
        }
    }
}

impl RgbComposition {
    fn new() -> Self {
        RgbComposition::default()
    }
    fn r(mut self, rule: CompositionRule) -> Self {
        self.r = rule;
        self
    }
    fn g(mut self, rule: CompositionRule) -> Self {
        self.g = rule;
        self
    }
    fn b(mut self, rule: CompositionRule) -> Self {
        self.b = rule;
        self
    }
    fn as_rgb_array(self) -> [CompositionRule; 3] {
        [self.r, self.g, self.b]
    }
}

pub struct ComposeCounts {
    pub counts: Array2<u32>,
    pub composition: RgbComposition,
    pub scaling: Scaling,
}

#[derive(Copy, Clone)]
pub enum ColorMapName {
    GrayScale,
    Viridis,
}

pub struct ColorMap {
    pub name: ColorMapName,
    pub invert: bool,
}

impl ColorMap {
    pub fn create_image(&self, normalized_counts: Array2<f64>) -> DynamicImage {
        match self.name {
            ColorMapName::GrayScale => GrayImage::from_vec(
                normalized_counts.ncols() as u32,
                normalized_counts.nrows() as u32,
                normalized_counts
                    .into_iter()
                    .map(|count| {
                        u8_from_normalized_counts(if self.invert { 1.0 - count } else { count })
                    })
                    .collect(),
            )
            .unwrap()
            .into(),
            ColorMapName::Viridis => RgbImage::from_vec(
                normalized_counts.ncols() as u32,
                normalized_counts.nrows() as u32,
                normalized_counts
                    .into_iter()
                    .flat_map(|norm_count| {
                        colorous::VIRIDIS
                            .eval_continuous(if self.invert {
                                1.0 - norm_count
                            } else {
                                norm_count
                            })
                            .as_array()
                    })
                    .collect(),
            )
            .unwrap()
            .into(),
        }
    }
}

pub enum ImageSpec {
    FeatureCounts {
        counts: Array2<u32>,
        scaling: Scaling,
        colormap: ColorMap,
    },
    CompositeFeatureCounts {
        compositions: Vec<ComposeCounts>,
    },
    AlphaFeatureCounts {
        counts: Array2<u32>,
        scaling: Scaling,
        channel: CompositeChannel,
    },
    GrayImage {
        image: Array2<u8>,
    },
    Continuous {
        values: Array2<f64>,
        normalize: Normalize,
        colormap: ColorMap,
    },
}

fn scale_and_normalize_counts(counts: &Array2<u32>, scaling: Scaling) -> Array2<f64> {
    let scaled_counts = counts.map(|&x| scaling.apply(x as f64));
    Normalize::MaxCounts.apply(&scaled_counts)
}

impl ImageSpec {
    pub fn create_image(&self) -> DynamicImage {
        match self {
            ImageSpec::FeatureCounts {
                counts,
                scaling,
                colormap,
            } => {
                let normalized_counts = scale_and_normalize_counts(counts, *scaling);

                colormap.create_image(normalized_counts)
            }
            ImageSpec::CompositeFeatureCounts { compositions } => {
                assert!(!compositions.is_empty());

                let mut rgb_buffer =
                    vec![Array2::<f64>::zeros(compositions[0].counts.raw_dim()); 3];
                for ComposeCounts {
                    counts,
                    composition,
                    scaling,
                } in compositions
                {
                    let normalized_counts = scale_and_normalize_counts(counts, *scaling);
                    for (index, value) in normalized_counts.indexed_iter() {
                        for (buf, channel) in zip_eq(&mut rgb_buffer, composition.as_rgb_array()) {
                            buf[index] += match channel {
                                CompositionRule::Zero => 0.0,
                                CompositionRule::Scaled => *value,
                            };
                            buf[index] = buf[index].min(1.0);
                        }
                    }
                }

                RgbImage::from_vec(
                    rgb_buffer[0].ncols() as u32,
                    rgb_buffer[0].nrows() as u32,
                    izip!(&rgb_buffer[0], &rgb_buffer[1], &rgb_buffer[2])
                        .flat_map(|(rv, gv, bv)| [*rv, *gv, *bv])
                        .map(u8_from_normalized_counts)
                        .collect(),
                )
                .unwrap()
                .into()
            }
            ImageSpec::AlphaFeatureCounts {
                counts,
                scaling,
                channel,
            } => {
                let channel = *channel;
                let normalized_counts = scale_and_normalize_counts(counts, *scaling);
                RgbaImage::from_vec(
                    normalized_counts.ncols() as u32,
                    normalized_counts.nrows() as u32,
                    normalized_counts
                        .into_iter()
                        .flat_map(|norm_count| {
                            channel
                                .rgb_pixel()
                                .into_iter()
                                .chain(std::iter::once(u8_from_normalized_counts(norm_count)))
                        })
                        .collect(),
                )
                .unwrap()
                .into()
            }
            ImageSpec::GrayImage { image } => GrayImage::from_vec(
                image.ncols() as u32,
                image.nrows() as u32,
                image.iter().copied().collect(),
            )
            .unwrap()
            .into(),
            ImageSpec::Continuous {
                values,
                normalize,
                colormap,
            } => {
                let normalized = normalize.apply(values);
                colormap.create_image(normalized)
            }
        }
    }
}
