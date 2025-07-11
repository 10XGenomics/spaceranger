//! stats
#![deny(missing_docs)]
pub mod nx;
mod reservoir_sampling;
pub use crate::reservoir_sampling::ReservoirSampler;
pub use nx::{n50, n90};

/// Compute the effective diversity (inverse Simpson index) of a distribution of numbers.
pub fn effective_diversity(members: impl Iterator<Item = f64>) -> f64 {
    // inverse Simpson index
    let mut s = 0_f64;
    let mut s2 = 0_f64;
    for count in members {
        s += count;
        s2 += count.powi(2);
    }
    s.powi(2) / s2
}
