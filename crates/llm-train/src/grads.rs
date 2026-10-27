//! Gradients as a monoid.
//!
//! A backward pass over a long sequence cannot materialise `positions ×
//! vocabulary` logits at once: for a 512-token example and Qwen3's 151 936
//! classes that is 600 MB in `f64` before anything else. So the pass is
//! *chunked*, and the partial results are combined.
//!
//! Combining is where chunked backprop usually goes wrong, because the
//! per-chunk results are not simply addable: cross-entropy divides by the
//! number of unmasked positions *in its own call*, so adding chunk gradients
//! directly weights a chunk of 3 supervised tokens the same as one of 300.
//!
//! The fix is to make the thing being combined a monoid under a correct
//! operation, and to normalise once at the end:
//!
//! * [`Grads`] — named tensors under elementwise addition. Identity is the
//!   empty map; a missing key behaves as a zero tensor.
//! * [`LossSum`] — a weighted sum with its weight, so the mean is recovered
//!   by one division at the very end.
//! * [`Step`] — the product of the two. A product of monoids is a monoid,
//!   which is why one `fold` suffices for the whole pass.
//!
//! All three operations are associative *and* commutative, so the chunk order
//! does not change the result and the fold may be reassociated or
//! parallelised freely. The tests check the laws rather than one example.

use std::collections::BTreeMap;

use candle_core::Tensor;

use crate::error::TrainError;

type Result<T> = std::result::Result<T, TrainError>;

fn candle(e: candle_core::Error) -> TrainError {
    TrainError::Candle(e.to_string())
}

/// A weighted sum, carrying its own weight.
///
/// The monoid is `(0, 0)` under componentwise addition. [`LossSum::mean`]
/// turns it back into a loss. Keeping the weight means a chunk's loss is
/// pooled in proportion to how many supervised positions it actually had.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LossSum {
    /// `Σ loss_chunk × unmasked_chunk`.
    pub weighted: f64,
    /// `Σ unmasked_chunk`.
    pub unmasked: usize,
}

impl LossSum {
    pub const EMPTY: Self = Self {
        weighted: 0.0,
        unmasked: 0,
    };

    /// One chunk's contribution: a mean over `unmasked` positions.
    pub fn of(mean: f64, unmasked: usize) -> Self {
        Self {
            weighted: mean * unmasked as f64,
            unmasked,
        }
    }

    pub fn combine(self, rhs: Self) -> Self {
        Self {
            weighted: self.weighted + rhs.weighted,
            unmasked: self.unmasked + rhs.unmasked,
        }
    }

    /// The pooled mean. Zero when nothing was supervised, which is the right
    /// answer for a batch in which every position was masked.
    pub fn mean(self) -> f64 {
        if self.unmasked == 0 {
            0.0
        } else {
            self.weighted / self.unmasked as f64
        }
    }
}

/// Named gradients under elementwise addition.
///
/// `BTreeMap`, not `HashMap`: iteration order is then deterministic, so two
/// runs over the same data produce the same optimiser updates in the same
/// order. Reproducibility is cheap here and expensive to add later.
#[derive(Debug, Clone, Default)]
pub struct Grads(BTreeMap<String, Tensor>);

impl Grads {
    /// The identity element.
    pub fn empty() -> Self {
        Self(BTreeMap::new())
    }

    /// A single named gradient.
    pub fn of(name: impl Into<String>, grad: Tensor) -> Self {
        Self(BTreeMap::from([(name.into(), grad)]))
    }

    /// The monoid operation: union of keys, elementwise sum where both sides
    /// have one.
    ///
    /// A key present on one side only is carried through unchanged, which is
    /// what makes [`Grads::empty`] a true identity rather than a special case
    /// the caller has to remember.
    pub fn combine(mut self, rhs: Self) -> Result<Self> {
        for (name, g) in rhs.0 {
            match self.0.remove(&name) {
                None => {
                    self.0.insert(name, g);
                }
                Some(existing) => {
                    let summed = (&existing + &g).map_err(candle)?;
                    self.0.insert(name, summed);
                }
            }
        }
        Ok(self)
    }

    /// Multiply every gradient by `factor`.
    pub fn scaled(self, factor: f64) -> Result<Self> {
        self.0
            .into_iter()
            .try_fold(Self::empty(), |acc, (name, g)| {
                let scaled = g.affine(factor, 0.0).map_err(candle)?;
                acc.combine(Self::of(name, scaled))
            })
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.0.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The global L2 norm over all gradients, as PyTorch's
    /// `clip_grad_norm_` defines it: one norm over the concatenation, not a
    /// norm per tensor.
    pub fn global_norm(&self) -> Result<f64> {
        let total = self.0.values().try_fold(0.0f64, |acc, g| {
            let sq = g
                .to_dtype(candle_core::DType::F32)
                .and_then(|g| g.sqr())
                .and_then(|g| g.sum_all())
                .and_then(|s| s.to_scalar::<f32>())
                .map_err(candle)?;
            Ok::<f64, TrainError>(acc + sq as f64)
        })?;
        Ok(total.sqrt())
    }

    /// Scale every gradient down so the global norm is at most `max_norm`.
    ///
    /// Below the threshold this is the identity, so it can be applied
    /// unconditionally. A non-finite norm — which a diverging run produces
    /// before it produces NaN weights — zeroes the step rather than poisoning
    /// the optimiser moments, and says so.
    pub fn clipped(self, max_norm: f64) -> Result<(Self, f64)> {
        let norm = self.global_norm()?;
        if !norm.is_finite() {
            tracing::warn!(norm, "non-finite gradient norm; skipping this update");
            return Ok((Self::empty(), norm));
        }
        if max_norm <= 0.0 || norm <= max_norm {
            return Ok((self, norm));
        }
        Ok((self.scaled(max_norm / norm)?, norm))
    }
}

/// A product of monoids: one chunk's loss contribution and its gradients.
#[derive(Debug, Clone, Default)]
pub struct Step {
    pub loss: LossSum,
    pub grads: Grads,
}

impl Step {
    pub fn empty() -> Self {
        Self {
            loss: LossSum::EMPTY,
            grads: Grads::empty(),
        }
    }

    pub fn combine(self, rhs: Self) -> Result<Self> {
        Ok(Self {
            loss: self.loss.combine(rhs.loss),
            grads: self.grads.combine(rhs.grads)?,
        })
    }

    /// Fold a sequence of chunk results into one.
    ///
    /// Associativity and commutativity are what make this safe to reassociate
    /// — the chunks may be produced in any order.
    pub fn concat(parts: impl IntoIterator<Item = Result<Self>>) -> Result<Self> {
        parts
            .into_iter()
            .try_fold(Self::empty(), |acc, part| acc.combine(part?))
    }

    /// Finish the pass: pooled mean loss, and gradients divided by the same
    /// weight so they are the gradient *of that mean*.
    pub fn finish(self) -> Result<(f64, Grads)> {
        let n = self.loss.unmasked;
        let mean = self.loss.mean();
        let grads = if n == 0 {
            Grads::empty()
        } else {
            self.grads.scaled(1.0 / n as f64)?
        };
        Ok((mean, grads))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn t(vals: &[f32]) -> Tensor {
        Tensor::from_slice(vals, (vals.len(),), &Device::Cpu).unwrap()
    }

    fn v(g: &Grads, name: &str) -> Vec<f32> {
        g.get(name).unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn empty_is_a_two_sided_identity() {
        let x = Grads::of("a", t(&[1.0, 2.0]));
        let left = Grads::empty().combine(x.clone()).unwrap();
        let right = x.clone().combine(Grads::empty()).unwrap();
        assert_eq!(v(&left, "a"), vec![1.0, 2.0]);
        assert_eq!(v(&right, "a"), vec![1.0, 2.0]);
    }

    #[test]
    fn combine_is_associative_and_commutative() {
        let a = Grads::of("w", t(&[1.0, 0.0]));
        let b = Grads::of("w", t(&[0.0, 2.0]));
        let c = Grads::of("w", t(&[3.0, 3.0]));

        let left = a.clone().combine(b.clone()).unwrap().combine(c.clone()).unwrap();
        let right = a.clone().combine(b.clone().combine(c.clone()).unwrap()).unwrap();
        assert_eq!(v(&left, "w"), v(&right, "w"));

        let fwd = a.clone().combine(b.clone()).unwrap();
        let rev = b.combine(a).unwrap();
        assert_eq!(v(&fwd, "w"), v(&rev, "w"));
        assert_eq!(v(&fwd, "w"), vec![1.0, 2.0]);
    }

    #[test]
    fn disjoint_keys_are_unioned_not_dropped() {
        let g = Grads::of("a", t(&[1.0]))
            .combine(Grads::of("b", t(&[2.0])))
            .unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(v(&g, "a"), vec![1.0]);
        assert_eq!(v(&g, "b"), vec![2.0]);
    }

    #[test]
    fn loss_sum_pools_by_weight_not_by_chunk_count() {
        // A chunk with 3 supervised tokens at loss 1.0 and one with 300 at
        // loss 2.0 must pool to ~1.99, not to the unweighted 1.5 that naive
        // averaging gives.
        let pooled = LossSum::of(1.0, 3).combine(LossSum::of(2.0, 300));
        assert!((pooled.mean() - 603.0 / 303.0).abs() < 1e-12);
        assert!((pooled.mean() - 1.5).abs() > 0.4);
    }

    #[test]
    fn an_all_masked_batch_has_zero_loss_and_no_gradient() {
        let (loss, grads) = Step::empty().finish().unwrap();
        assert_eq!(loss, 0.0);
        assert!(grads.is_empty());
    }

    #[test]
    fn chunking_matches_the_single_pass_result() {
        // Two chunks of unequal weight, combined then normalised, must equal
        // the gradient computed as if there had been one chunk.
        let whole = Step {
            loss: LossSum::of(1.0, 4),
            grads: Grads::of("w", t(&[8.0, 4.0])),
        };
        let part_a = Step {
            loss: LossSum::of(1.0, 1),
            grads: Grads::of("w", t(&[2.0, 1.0])),
        };
        let part_b = Step {
            loss: LossSum::of(1.0, 3),
            grads: Grads::of("w", t(&[6.0, 3.0])),
        };
        let (l1, g1) = whole.finish().unwrap();
        let (l2, g2) = part_a.combine(part_b).unwrap().finish().unwrap();
        assert!((l1 - l2).abs() < 1e-12);
        assert_eq!(v(&g1, "w"), v(&g2, "w"));
    }

    #[test]
    fn clipping_below_the_threshold_changes_nothing() {
        let g = Grads::of("w", t(&[0.3, 0.4])); // norm 0.5
        let (clipped, norm) = g.clipped(1.0).unwrap();
        assert!((norm - 0.5).abs() < 1e-6);
        assert_eq!(v(&clipped, "w"), vec![0.3, 0.4]);
    }

    #[test]
    fn clipping_rescales_to_exactly_the_threshold() {
        let g = Grads::of("a", t(&[3.0, 0.0])).combine(Grads::of("b", t(&[4.0]))).unwrap();
        // global norm over both tensors is 5
        assert!((g.global_norm().unwrap() - 5.0).abs() < 1e-6);
        let (clipped, _) = g.clipped(1.0).unwrap();
        assert!((clipped.global_norm().unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_non_finite_norm_zeroes_the_step() {
        let nan = Tensor::from_slice(&[f32::NAN], (1,), &Device::Cpu).unwrap();
        let (clipped, norm) = Grads::of("w", nan).clipped(1.0).unwrap();
        assert!(norm.is_nan());
        assert!(clipped.is_empty(), "a NaN gradient must not reach the optimiser");
    }

    #[test]
    fn scaling_is_linear() {
        let g = Grads::of("w", t(&[1.0, -2.0])).scaled(2.5).unwrap();
        assert_eq!(v(&g, "w"), vec![2.5, -5.0]);
    }

    #[test]
    fn global_norm_spans_every_tensor() {
        let zeros = Tensor::zeros((3,), DType::F32, &Device::Cpu).unwrap();
        let g = Grads::of("z", zeros);
        assert_eq!(g.global_norm().unwrap(), 0.0);
    }
}
