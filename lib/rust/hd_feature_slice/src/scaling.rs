use itertools::Itertools;
use ndarray::Array2;
use serde::Deserialize;

#[derive(Clone, Copy, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Scaling {
    Linear,
    #[default]
    Log1p,
    LinearWithMax(f64),
}

impl Scaling {
    pub fn apply(self, value: f64) -> f64 {
        match self {
            Scaling::Linear => value,
            Scaling::Log1p => (value + 1.0).log2(),
            Scaling::LinearWithMax(m) => value.min(m),
        }
    }
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Normalize {
    MaxCounts,
    MinMax,
    ZeroToMax,
}

impl Normalize {
    pub fn apply(self, counts: &Array2<f64>) -> Array2<f64> {
        let (min_val, den) = match self {
            Normalize::MaxCounts => {
                let max_val = counts.iter().copied().reduce(f64::max).unwrap().max(1.0);
                (0.0, max_val)
            }
            Normalize::MinMax => {
                let (min_val, max_val) = counts.iter().minmax().into_option().unwrap();
                let den = if max_val == min_val {
                    1.0
                } else {
                    max_val - min_val
                };
                (*min_val, den)
            }
            Normalize::ZeroToMax => {
                let mut max_val = counts.iter().copied().reduce(f64::max).unwrap();
                if max_val <= 0.0 {
                    max_val = 1.0;
                }
                (0.0, max_val)
            }
        };

        counts.map(|&x| (x - min_val) / den)
    }
}
