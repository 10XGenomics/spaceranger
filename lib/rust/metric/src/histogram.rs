//! This module defines structures useful in keeping track of distributions.
//! The following structs are defined
//! * `SimpleHistogram<K>` : Histogram to keep track of counts of different items
#![deny(missing_docs)]
use crate::{CountMetric, JsonReport, JsonReporter, Metric, TxHashMap};
use itertools::Itertools;
use metric_derive::Metric;
use num_traits::cast::AsPrimitive;
use num_traits::int::PrimInt;
use serde::{Deserialize, Serialize};
use stats::effective_diversity;
use std::borrow::Borrow;
use std::cmp::Reverse;
use std::collections::{btree_map, hash_map, BTreeMap, BinaryHeap};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::iter::FromIterator;
use std::ops::AddAssign;
use std::{cmp, f64};

/// Abstract over the storage that backs a Histogram.
///
/// The methods are only used to implement the provided method set of Histogram
/// and are in a separate trait to ensure they don't leak into the public API.
trait HistogramStorage<K>
where
    K: Hash + Eq + Ord,
{
    /// Get a value from the histogram backing storage.
    fn get<Q>(&self, key: &Q) -> Option<&CountMetric>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;

    /// Get a value from the histogram backing storage, mutably.
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut CountMetric>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;

    /// Insert a value into the histogram backing storage.
    /// Overwrites any current value.
    /// Returns the previous value if there was one.
    fn insert(&mut self, k: K, v: CountMetric) -> Option<CountMetric>;

    /// Convert this storage into an owned iterator.
    fn into_iter(self) -> impl Iterator<Item = (K, CountMetric)>;

    /// Iterate over counts.
    fn counts(&self) -> impl Iterator<Item = &CountMetric>;

    /// Return a iterator over the (K, Count) pairs, in key-sorted order.
    fn sorted<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a CountMetric)>
    where
        K: 'a;
}

/// The interface presented by a generic implementation of a histogram.
#[allow(private_bounds)] // This trait isn't intended to be implemented outside this module.
pub trait Histogram<K>: HistogramStorage<K>
where
    K: Hash + Eq + Ord,
{
    /// Get the number of items in the histogram.
    fn len(&self) -> usize;

    /// Return true if the histogram is empty.
    fn is_empty(&self) -> bool;

    /// Return an iterator over the counts.
    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a CountMetric)>
    where
        K: 'a;

    /// Increment the counter for the `key` by `val`, inserting it into the
    /// distribution if it is not already present
    ///
    /// # Example
    /// ```rust
    /// use metric::{Metric, Histogram, SimpleHistogram};
    /// let mut hist = SimpleHistogram::default();
    /// assert!(hist.get(&10)==0);
    /// hist.observe_by(&10, 1);
    /// hist.observe_by(&10, 2);
    /// assert!(hist.get(&10)==3);
    /// ```
    fn observe_by<Q, T>(&mut self, key: &Q, val: T)
    where
        K: Borrow<Q>,
        Q: ToOwned<Owned = K> + Hash + Eq + Ord + ?Sized,
        T: Into<i64>,
    {
        let val = val.into();
        if val == 0 {
            return;
        }
        if let Some(v) = self.get_mut(key) {
            v.increment_by(val);
        } else {
            HistogramStorage::insert(self, key.to_owned(), val.into());
        }
    }

    /// Increment the counter of `key` by `val`.
    fn observe_by_owned(&mut self, key: K, val: impl Into<i64>)
    where
        K: Sized,
    {
        let val = val.into();
        if val == 0 {
            return;
        }
        if let Some(count) = self.get_mut(&key) {
            count.increment_by(val);
        } else {
            HistogramStorage::insert(self, key, val.into());
        }
    }

    /// Increment the counter for the `key` by one, inserting it into the
    /// distribution if it is not already present
    fn observe<Q>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: ToOwned<Owned = K> + Hash + Ord + Eq + ?Sized,
    {
        if let Some(count) = self.get_mut(key) {
            count.increment_by(1);
        } else {
            HistogramStorage::insert(self, key.to_owned(), 1.into());
        }
    }

    /// Increment the counter of the owned item by one.
    fn observe_owned(&mut self, key: K)
    where
        K: Sized,
    {
        if let Some(count) = self.get_mut(&key) {
            count.increment_by(1);
        } else {
            HistogramStorage::insert(self, key, 1.into());
        }
    }

    /// Observe multiple owned items.
    fn extend_owned(&mut self, iter: impl IntoIterator<Item = K>)
    where
        K: Sized,
    {
        for x in iter {
            self.observe_owned(x);
        }
    }

    /// Create a `SimpleHistogram` from an iterator of owned items.
    fn from_iter_owned(iter: impl IntoIterator<Item = K>) -> Self
    where
        Self: Sized + Default,
    {
        let mut histogram = Self::default();
        histogram.extend_owned(iter);
        histogram
    }

    /// Insert an entry into the histogram, replacing the entry if it already exists.
    fn insert<V: Into<CountMetric>>(&mut self, key: K, value: V)
    where
        K: Sized,
    {
        HistogramStorage::insert(self, key, value.into());
    }

    /// Get the count for the `key` as i64. Returns 0 if the key
    /// is not present
    ///
    /// # Example
    /// ```rust
    /// use metric::{Metric, Histogram, SimpleHistogram};
    /// let mut hist = SimpleHistogram::default();
    /// assert!(hist.get(&10)==0);
    /// hist.observe(&10);
    /// hist.observe(&10);
    /// hist.observe(&11);
    /// assert!(hist.get(&10)==2);
    /// assert!(hist.get(&11)==1);
    /// assert!(hist.get(&8)==0);
    /// ```
    fn get<Q>(&self, key: &Q) -> i64
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        HistogramStorage::get(self, key).map_or(0i64, |c| c.count())
    }

    /// Get an iterator over the raw counts
    fn raw_counts<'a>(&'a self) -> impl Iterator<Item = &'a i64> + 'a
    where
        K: 'a,
    {
        self.counts().map(|x| &x.count)
    }

    /// Compute the effective diversity (inverse Simpson index) of the distribution.
    fn effective_diversity(&self) -> f64 {
        effective_diversity(self.counts().map(|c| c.count() as f64))
    }

    /// Return the mean of the histogram,
    /// or None if the histogram is empty
    fn mean(&self) -> Option<f64>
    where
        K: AsPrimitive<i64>,
    {
        if self.is_empty() {
            return None;
        }
        let mut m = 0.0;
        let mut n = 0;
        for (k, v) in self.iter() {
            m += (v.count() * k.as_()) as f64;
            n += v.count();
        }
        Some(m / n as f64)
    }

    /// Only retain values that are true under some predicate `pred`.
    fn retain<F>(&mut self, pred: F)
    where
        F: FnMut(&K, &mut CountMetric) -> bool;

    /// Return the maximum value observed,
    /// or None if the histogram is empty
    fn max_key(&self) -> Option<K>
    where
        K: Ord + Clone;

    /// Return the minimum value observed,
    /// or None if the histogram is empty
    fn min_key(&self) -> Option<K>
    where
        K: Ord + Clone;

    /// Return the provided quantiles of the histogram, or `None` if the histogram is empty.
    /// Each call to `quantiles()` triggers an allocation and sort in order to compute the
    /// cumulative distribution. The quantiles need to be in the open range (0, 1).
    ///
    /// Uses Type 7 interpolation from:
    /// Hyndman, R.J.; Fan, Y. (1996). "Sample Quantiles in Statistical Packages".
    /// See https://www.amherst.edu/media/view/129116/original/Sample+Quantiles.pdf for details
    ///
    /// # Example
    /// ```
    /// use metric::{Metric, Histogram, SimpleHistogram};
    /// let mut hist = SimpleHistogram::default();
    /// for i in 0..=100 {
    ///     hist.observe(&i)
    /// }
    /// let quantiles = hist.quantiles(&[0.25, 0.5, 0.75]);
    /// assert_eq!(quantiles, Some(vec![25.0, 50.0, 75.0]));
    /// ```
    fn quantiles<I, F>(&self, quantile_queries: I) -> Option<Vec<f64>>
    where
        K: Ord + AsPrimitive<f64>,
        I: IntoIterator<Item = F>,
        F: Borrow<f64>,
    {
        if self.is_empty() {
            return None;
        }

        // Compute the cumulative distribution. We will map both the key
        // and values to f64 for convenience
        let cdf: Vec<_> = self
            .sorted()
            .scan((0.0, 0.0), |(_, acc), (k, c)| {
                *acc += c.count() as f64;
                Some((k.as_(), *acc))
            })
            .collect();

        // The total number of observations. cdf is guaranteed to have >=1 element, the unwrap will not fail.
        let n = cdf.last().unwrap().1;

        Some(
            quantile_queries
                .into_iter()
                .map(|quantile| {
                    let quantile = *quantile.borrow(); // f64
                    assert!(
                        quantile > 0.0 && quantile < 1.0,
                        "Quantiles are only defined in the open range (0, 1). Found {quantile}"
                    );
                    let h = (n - 1.0) * quantile;

                    // We want to find two values here. If we collect all the individual observations in
                    // the histogram, collect it in a vector (of size n) and sort it, we want to find the
                    // element that appears at the index `h.floor()` (called `low_value`) and the element
                    // that appears at the index `h.ceil()` (called `high_value`). Then we do linear
                    // interpolation to find the quartile.
                    let low_index = cdf.iter().position(|(_, v)| *v > h.floor()).unwrap();
                    let low_value = cdf[low_index].0;
                    let high_value = if cdf[low_index].1 > h.ceil() {
                        cdf[low_index].0
                    } else {
                        cdf[low_index + 1].0
                    };
                    low_value + (h - h.floor()) * (high_value - low_value)
                })
                .collect(),
        )
    }

    /// Return the provided percentiles of the histogram, or `None` if the histogram is empty.
    /// Each call to `percentiles()` triggers an allocation and sort in order to compute the
    /// cumulative distribution. The percentiles need to be in the open range (0, 100).
    ///
    /// Uses Type 7 interpolation from:
    /// Hyndman, R.J.; Fan, Y. (1996). "Sample Quantiles in Statistical Packages".
    /// See https://www.amherst.edu/media/view/129116/original/Sample+Quantiles.pdf for details
    ///
    /// # Example
    /// ```
    /// use metric::{Metric, Histogram, SimpleHistogram};
    /// let mut hist = SimpleHistogram::default();
    /// for i in 0..=100 {
    ///     hist.observe(&i)
    /// }
    /// let percentiles = hist.percentiles(&[25.0, 50.0, 75.0]);
    /// assert_eq!(percentiles, Some(vec![25.0, 50.0, 75.0]));
    /// ```
    fn percentiles<I, F>(&self, percentile_queries: I) -> Option<Vec<f64>>
    where
        K: Ord + AsPrimitive<f64>,
        I: IntoIterator<Item = F>,
        F: Borrow<f64>,
    {
        self.quantiles(percentile_queries.into_iter().map(|x| *x.borrow() / 100.0))
    }

    /// Make a new `Histogram` that retains only the top `n` most common items in the histogram.
    fn top_n(&self, n: usize) -> Self
    where
        Self: Sized + Default,
        K: Clone + Ord,
    {
        let mut seqs = BinaryHeap::with_capacity(n + 1);
        for (k, v) in self.iter() {
            seqs.push(Reverse((v, k)));
            while seqs.len() > n {
                let _ = seqs.pop();
            }
        }
        let mut hist = Self::default();
        for Reverse((count, key)) in seqs {
            HistogramStorage::insert(&mut hist, key.clone(), *count);
        }
        hist
    }

    /// Create a new histogram modifying the existing keys using the function `f`
    fn map_key<F, K2, H>(self, mut f: F) -> H
    where
        Self: Sized,
        H: Histogram<K2> + Default,
        F: FnMut(K) -> K2,
        K2: Clone + Eq + Hash + Ord,
    {
        let mut result = H::default();
        for (k, v) in self.into_iter().map(|(k, v)| (f(k), v)) {
            result.observe_by(&k, v.count());
        }
        result
    }
}

/// Use this struct to keep track of counts of various items.
/// The underlying representation is a `Hashmap`. The items
/// need to have a string representation (`ToString` trait)
/// so that they are report friendly
#[derive(Clone, Debug, Serialize, Deserialize, Metric)]
#[serde(transparent, bound = "")]
pub struct SimpleHistogram<K>
where
    K: Eq + Hash + Serialize + for<'a> Deserialize<'a>,
{
    distribution: TxHashMap<K, CountMetric>,
}

impl<K> HistogramStorage<K> for SimpleHistogram<K>
where
    K: Eq + Ord + Hash + Serialize + for<'a> Deserialize<'a>,
{
    fn get<Q>(&self, key: &Q) -> Option<&CountMetric>
    where
        K: Borrow<Q> + Hash + Eq,
        Q: Hash + Eq + ?Sized,
    {
        self.distribution.get(key)
    }

    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut CountMetric>
    where
        K: Borrow<Q> + Hash + Eq,
        Q: Hash + Eq + ?Sized,
    {
        self.distribution.get_mut(key)
    }

    fn insert(&mut self, k: K, v: CountMetric) -> Option<CountMetric> {
        self.distribution.insert(k, v)
    }

    fn counts(&self) -> impl Iterator<Item = &CountMetric> {
        self.distribution.values()
    }

    fn sorted<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a CountMetric)>
    where
        K: 'a + Ord,
    {
        self.distribution.iter().sorted()
    }

    fn into_iter(self) -> impl Iterator<Item = (K, CountMetric)> {
        self.distribution.into_iter()
    }
}

impl<K> Histogram<K> for SimpleHistogram<K>
where
    K: Eq + Ord + Hash + Serialize + for<'a> Deserialize<'a>,
{
    fn len(&self) -> usize {
        self.distribution.len()
    }

    fn is_empty(&self) -> bool {
        self.distribution.is_empty()
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a CountMetric)>
    where
        K: 'a,
    {
        self.distribution.iter()
    }

    fn retain<F>(&mut self, pred: F)
    where
        F: FnMut(&K, &mut CountMetric) -> bool,
    {
        self.distribution.retain(pred);
    }

    fn max_key(&self) -> Option<K>
    where
        K: Ord + Clone,
    {
        self.distribution.keys().max().cloned()
    }

    fn min_key(&self) -> Option<K>
    where
        K: Ord + Clone,
    {
        self.distribution.keys().min().cloned()
    }
}

impl<K> SimpleHistogram<K>
where
    K: Eq + Hash + Ord + Serialize + for<'a> Deserialize<'a>,
{
    // TODO: lift this method to the Histogram trait.
    /// Create a binned histogram with keys `(start..=stop).step_by(step)`
    /// from the existing histogram. The returned histogram will contain
    /// all the keys in the range with zero values for any bin which is
    /// empty. Any value less that `start` is added to the `start` bin and
    /// any value greater that `stop` is added to the `stop` bin. This is
    /// usually useful in reporting.
    ///
    /// Panics if:
    /// - start is not less than stop
    /// - step is equal to zero
    ///
    /// TODO: May be create a `BinnedHistogram` class in the future?
    ///
    /// # Example
    /// ```
    /// use metric::{Metric, Histogram, SimpleHistogram};
    /// let mut hist = SimpleHistogram::default();
    /// for i in 20..100 {
    ///     hist.observe(&i);
    /// }
    /// let binned = hist.binned(0, 50, 10);
    /// assert_eq!(binned.get(&10), 0);
    /// assert_eq!(binned.get(&30), 10);
    /// assert_eq!(binned.get(&50), 50);
    /// ```
    pub fn binned(self, start: K, stop: K, step: K) -> Self
    where
        K: PrimInt + Ord + Copy + Display + AddAssign<K>,
    {
        assert!(start < stop);
        let mut result: Self = self.map_key(|k: K| {
            cmp::min(
                stop,
                cmp::max(
                    start,
                    k.checked_div(&step)
                        .and_then(|d| d.checked_mul(&step))
                        .expect("Histogram binning failed!"),
                ),
            )
        });
        let mut k = start;
        while k <= stop {
            result.distribution.entry(k).or_insert_with(|| 0.into());
            k += step;
        }
        result
    }
}

impl<K> Default for SimpleHistogram<K>
where
    K: Eq + Hash + Serialize + for<'a> Deserialize<'a>,
{
    fn default() -> Self {
        Self {
            distribution: TxHashMap::default(),
        }
    }
}

impl<'a, K, Q> Extend<&'a Q> for SimpleHistogram<K>
where
    K: Borrow<Q> + Eq + Ord + Hash + Serialize + for<'d> Deserialize<'d>,
    Q: 'a + Hash + Ord + Eq + ToOwned<Owned = K>,
{
    fn extend<T: IntoIterator<Item = &'a Q>>(&mut self, iter: T) {
        for x in iter {
            self.observe(x);
        }
    }
}

impl<'a, K, Q> FromIterator<&'a Q> for SimpleHistogram<K>
where
    K: Default + Borrow<Q> + Ord + Eq + Hash + Serialize + for<'d> Deserialize<'d>,
    Q: 'a + Hash + Eq + Ord + ToOwned<Owned = K>,
{
    fn from_iter<T: IntoIterator<Item = &'a Q>>(iter: T) -> Self {
        let mut histogram = Self::default();
        histogram.extend(iter);
        histogram
    }
}

impl<'a, K> IntoIterator for &'a SimpleHistogram<K>
where
    K: Eq + Hash + Serialize + for<'d> Deserialize<'d>,
{
    type Item = (&'a K, &'a CountMetric);
    type IntoIter = hash_map::Iter<'a, K, CountMetric>;

    fn into_iter(self) -> Self::IntoIter {
        self.distribution.iter()
    }
}

impl<K> IntoIterator for SimpleHistogram<K>
where
    K: Eq + Hash + Serialize + for<'d> Deserialize<'d>,
{
    type Item = (K, CountMetric);
    type IntoIter = hash_map::IntoIter<K, CountMetric>;

    fn into_iter(self) -> Self::IntoIter {
        self.distribution.into_iter()
    }
}

impl<K> JsonReport for SimpleHistogram<K>
where
    K: Eq + Hash + ToString + Serialize + for<'a> Deserialize<'a>,
{
    fn to_json_reporter(&self) -> JsonReporter {
        let reporter: JsonReporter = self
            .distribution
            .iter()
            .map(|(k, v)| (k.to_string(), v.count()))
            .collect();
        assert_eq!(
            self.distribution.len(),
            reporter.hashmap.len(),
            "Different keys in the histogram map to same string"
        );
        reporter
    }
}

/// Use this struct to keep track of counts of various items when order matters.
/// The underlying representation is a `BTreeMap`.
#[derive(Clone, Debug, Serialize, Deserialize, Metric)]
#[serde(transparent, bound = "")]
pub struct OrderedHistogram<K>
where
    K: Eq + Hash + Ord + Serialize + for<'a> Deserialize<'a>,
{
    distribution: BTreeMap<K, CountMetric>,
}

impl<K> HistogramStorage<K> for OrderedHistogram<K>
where
    K: Eq + Hash + Ord + Serialize + for<'a> Deserialize<'a>,
{
    fn get<Q>(&self, key: &Q) -> Option<&CountMetric>
    where
        K: Borrow<Q>,
        Q: Hash + Ord + Eq + ?Sized,
    {
        self.distribution.get(key)
    }

    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut CountMetric>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.distribution.get_mut(key)
    }

    fn insert(&mut self, k: K, v: CountMetric) -> Option<CountMetric> {
        self.distribution.insert(k, v)
    }

    fn counts(&self) -> impl Iterator<Item = &CountMetric> {
        self.distribution.values()
    }

    fn sorted<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a CountMetric)>
    where
        K: 'a + Ord,
    {
        self.distribution.iter().sorted()
    }

    fn into_iter(self) -> impl Iterator<Item = (K, CountMetric)> {
        self.distribution.into_iter()
    }
}

impl<K> Histogram<K> for OrderedHistogram<K>
where
    K: Eq + Hash + Ord + Serialize + for<'a> Deserialize<'a>,
{
    fn len(&self) -> usize {
        self.distribution.len()
    }

    fn is_empty(&self) -> bool {
        self.distribution.is_empty()
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a CountMetric)>
    where
        K: 'a,
    {
        self.distribution.iter()
    }

    fn retain<F>(&mut self, pred: F)
    where
        F: FnMut(&K, &mut CountMetric) -> bool,
    {
        self.distribution.retain(pred);
    }

    fn max_key(&self) -> Option<K>
    where
        K: Ord + Clone,
    {
        self.distribution.last_key_value().map(|(k, _)| k.clone())
    }

    fn min_key(&self) -> Option<K>
    where
        K: Ord + Clone,
    {
        self.distribution.first_key_value().map(|(k, _)| k.clone())
    }
}

impl<K> Default for OrderedHistogram<K>
where
    K: Eq + Hash + Ord + Serialize + for<'a> Deserialize<'a>,
{
    fn default() -> Self {
        Self {
            distribution: BTreeMap::default(),
        }
    }
}

impl<'a, K, Q> Extend<&'a Q> for OrderedHistogram<K>
where
    K: Borrow<Q> + Eq + Ord + Hash + Serialize + for<'d> Deserialize<'d>,
    Q: 'a + Hash + Ord + Eq + ToOwned<Owned = K>,
{
    fn extend<T: IntoIterator<Item = &'a Q>>(&mut self, iter: T) {
        for x in iter {
            self.observe(x);
        }
    }
}

impl<'a, K, Q> FromIterator<&'a Q> for OrderedHistogram<K>
where
    K: Default + Borrow<Q> + Ord + Eq + Hash + Serialize + for<'d> Deserialize<'d>,
    Q: 'a + Hash + Eq + Ord + ToOwned<Owned = K>,
{
    fn from_iter<T: IntoIterator<Item = &'a Q>>(iter: T) -> Self {
        let mut histogram = Self::default();
        histogram.extend(iter);
        histogram
    }
}

impl<'a, K> IntoIterator for &'a OrderedHistogram<K>
where
    K: Eq + Hash + Ord + Serialize + for<'d> Deserialize<'d>,
{
    type Item = (&'a K, &'a CountMetric);
    type IntoIter = btree_map::Iter<'a, K, CountMetric>;

    fn into_iter(self) -> Self::IntoIter {
        self.distribution.iter()
    }
}

impl<K> IntoIterator for OrderedHistogram<K>
where
    K: Eq + Hash + Ord + Serialize + for<'d> Deserialize<'d>,
{
    type Item = (K, CountMetric);
    type IntoIter = btree_map::IntoIter<K, CountMetric>;

    fn into_iter(self) -> Self::IntoIter {
        self.distribution.into_iter()
    }
}

impl<K> JsonReport for OrderedHistogram<K>
where
    K: Eq + Hash + Ord + ToString + Serialize + for<'a> Deserialize<'a>,
{
    fn to_json_reporter(&self) -> JsonReporter {
        let reporter: JsonReporter = self
            .distribution
            .iter()
            .map(|(k, v)| (k.to_string(), v.count()))
            .collect();
        assert_eq!(
            self.distribution.len(),
            reporter.hashmap.len(),
            "Different keys in the histogram map to same string"
        );
        reporter
    }
}

#[cfg(test)]
mod tests {
    use super::{Histogram, OrderedHistogram, SimpleHistogram};
    use anyhow::Error;
    use serde::Deserialize;
    use std::fs::File;
    use std::iter::zip;

    #[test]
    fn test_observe_ref() {
        let mut hist = SimpleHistogram::<i32>::default();
        hist.observe(&0);
        assert_eq!(hist.len(), 1);

        let mut hist = SimpleHistogram::<String>::default();
        hist.observe("");
        assert_eq!(hist.len(), 1);
    }

    #[test]
    fn test_observe_owned() {
        let mut hist = SimpleHistogram::<i32>::default();
        hist.observe_owned(0);
        assert_eq!(hist.len(), 1);

        let mut hist = SimpleHistogram::<String>::default();
        hist.observe_owned(String::new());
        assert_eq!(hist.len(), 1);
    }

    #[test]
    fn test_extend_ref() {
        let mut hist = SimpleHistogram::<i32>::default();
        hist.extend(&[1, 2, 3]);
        assert_eq!(hist.len(), 3);
    }

    #[test]
    fn test_extend_owned() {
        let mut hist = SimpleHistogram::<i32>::default();
        hist.extend_owned(0..3);
        assert_eq!(hist.len(), 3);
    }

    #[test]
    fn test_from_iterator_ref() {
        let hist = SimpleHistogram::<i32>::from_iter(&[1, 2, 3]);
        assert_eq!(hist.len(), 3);
    }

    #[test]
    fn test_from_iterator_owned() {
        let hist = SimpleHistogram::<i32>::from_iter_owned(0..3);
        assert_eq!(hist.len(), 3);
    }

    #[test]
    fn test_collect() {
        let hist: SimpleHistogram<i32> = [1, 2, 3].iter().collect();
        assert_eq!(hist.len(), 3);
    }

    #[test]
    fn test_ordered() {
        let hist = OrderedHistogram::<i32>::from_iter_owned([3, 1, 2, 3, 5]);
        assert_eq!(
            hist.into_iter()
                .map(|(k, count)| (k, count.count()))
                .collect::<Vec<_>>(),
            vec![(1, 1), (2, 1), (3, 2), (5, 1)]
        );
    }

    #[test]
    #[ignore]
    fn test_percentile() -> Result<(), Error> {
        // The test data is generated by computing the percentile using numpy
        #[derive(Deserialize)]
        struct TestData {
            samples: Vec<i64>,
            percentile_queries: Vec<f64>,
            percentile_results: Vec<f64>,
        }
        let test_data: Vec<TestData> =
            serde_json::from_reader(File::open("test_data/percentile_test.json")?)?;
        for data in test_data {
            let mut hist = SimpleHistogram::default();
            for s in data.samples {
                hist.observe(&s);
            }
            let results = hist.percentiles(&data.percentile_queries).unwrap();
            for (a, e) in zip(results, data.percentile_results) {
                assert!((a - e).abs() < 1e-6, "a = {a}, e = {e}");
            }
        }
        Ok(())
    }

    #[test]
    fn test_top_n() -> Result<(), Error> {
        let data = vec![0, 0, 0, 1, 1, 2, 2, 2, 2];
        let hist = {
            let mut hist = SimpleHistogram::default();
            for s in &data {
                hist.observe(s);
            }
            hist
        };
        let top_n = hist.top_n(2);
        let expected = {
            let mut expected = SimpleHistogram::default();
            for s in data.iter().filter(|&&s| s != 1) {
                expected.observe(s);
            }
            expected
        };
        assert!(
            top_n.distribution == expected.distribution,
            "top_n = {top_n:?}, expected = {expected:?}"
        );
        Ok(())
    }
}
