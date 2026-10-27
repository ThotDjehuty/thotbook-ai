//! Training objectives, delegated to `optimiz-rs`.
//!
//! `optimizr::losses` is the single source of truth for every objective this
//! pipeline minimises. Nothing here re-derives a loss or a gradient: this file
//! only moves data between candle tensors and the `optimizr::losses::Loss`
//! trait, so there is exactly one definition of cross-entropy in the stack and
//! it is the one with finite-difference-checked gradients.
//!
//! This replaces a hand-rolled implementation that returned `+mean(log q)`
//! instead of `-mean(log q)` — the negation was missing, so minimising it
//! drove the model away from its targets — and that silently ignored both its
//! `label_smoothing` argument and any notion of masking.
//!
//! Label convention: a label `< 0` marks a position to ignore (the usual
//! `-100` for padding and prompt tokens), matching what `trainer.rs` feeds in.

use candle_core::{DType, Tensor, D};
use optimizr::losses::{CrossEntropy, Loss, LossError};

/// Sentinel label for "do not train on this position".
pub const IGNORE_INDEX: i64 = -100;

/// Value of the cross-entropy loss, plus its gradient with respect to the
/// logits.
#[derive(Debug, Clone)]
pub struct LossOutput {
    /// `-(1/N) Σ_n Σ_i p̃_{n,i} log q_{n,i}`, averaged over unmasked positions.
    /// Non-negative by construction.
    pub value: f32,
    /// `∂loss/∂logits`, shaped like the logits that produced it. Masked
    /// positions are exactly zero.
    pub grad: Tensor,
    /// How many positions actually contributed.
    pub unmasked: usize,
}

fn map_loss_err(e: LossError) -> candle_core::Error {
    candle_core::Error::Msg(format!("optimizr::losses: {e}"))
}

/// Flatten `[.., vocab]` logits and `[..]` labels into the shapes
/// `optimizr::losses` expects, returning `(logits, labels, n, vocab)`.
fn flatten(
    logits: &Tensor,
    labels: &Tensor,
) -> Result<(Vec<f64>, Vec<i64>, usize, usize), candle_core::Error> {
    let vocab = logits.dim(D::Minus1)?;
    let n = logits.elem_count() / vocab;

    let flat_logits = logits.reshape((n, vocab))?.to_dtype(DType::F64)?;
    let logit_rows: Vec<f64> = flat_logits.flatten_all()?.to_vec1::<f64>()?;

    let flat_labels = labels.reshape((labels.elem_count(),))?.to_dtype(DType::I64)?;
    let label_vec: Vec<i64> = flat_labels.to_vec1::<i64>()?;

    if label_vec.len() != n {
        return Err(candle_core::Error::Msg(format!(
            "labels have {} entries but logits imply {n} positions",
            label_vec.len()
        )));
    }
    Ok((logit_rows, label_vec, n, vocab))
}

/// Cross-entropy over logits with optional label smoothing, masking every
/// position whose label is negative.
///
/// Returns the scalar loss *and* `∂loss/∂logits`, so a caller can update
/// parameters without an autograd tape.
pub fn cross_entropy(
    logits: &Tensor,
    labels: &Tensor,
    label_smoothing: f32,
) -> Result<LossOutput, candle_core::Error> {
    let device = logits.device().clone();
    let (logit_rows, label_vec, n, vocab) = flatten(logits, labels)?;

    // optimizr masks on a class index, so map the negative sentinel onto a
    // reserved index one past the vocabulary.
    let pad = vocab;
    let targets: Vec<usize> = label_vec
        .iter()
        .map(|&l| if l < 0 { pad } else { l as usize })
        .collect();
    let unmasked = targets.iter().filter(|&&t| t != pad).count();

    let mut ce = CrossEntropy::new(vocab).with_ignore_index(pad);
    if label_smoothing > 0.0 {
        ce = ce
            .with_label_smoothing(label_smoothing as f64)
            .map_err(map_loss_err)?;
    }

    let (value, grad) = ce
        .value_and_grad(&logit_rows, &targets)
        .map_err(map_loss_err)?;

    let grad_f32: Vec<f32> = grad.into_iter().map(|g| g as f32).collect();
    let grad = Tensor::from_vec(grad_f32, (n, vocab), &device)?.reshape(logits.shape())?;

    Ok(LossOutput {
        value: value as f32,
        grad,
        unmasked,
    })
}

/// The loss value alone, as a scalar tensor on the logits' device.
///
/// Kept for callers that only report the number. The sign is correct here:
/// cross-entropy is non-negative, and a confidently wrong prediction scores
/// higher than a confidently right one.
pub fn cross_entropy_loss(
    logits: &Tensor,
    labels: &Tensor,
    label_smoothing: f32,
) -> Result<Tensor, candle_core::Error> {
    let out = cross_entropy(logits, labels, label_smoothing)?;
    Tensor::new(&[out.value], logits.device())?.squeeze(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn scalar(t: &Tensor) -> f32 {
        t.to_scalar::<f32>().unwrap()
    }

    #[test]
    fn uniform_logits_give_ln_vocab_and_are_non_negative() {
        let device = Device::Cpu;
        let vocab = 10;
        let logits = Tensor::zeros((2, vocab), DType::F32, &device).unwrap();
        let labels = Tensor::new(&[1i64, 2], &device).unwrap();

        let v = scalar(&cross_entropy_loss(&logits, &labels, 0.0).unwrap());
        assert!(v > 0.0, "cross-entropy must be non-negative, got {v}");
        assert!(
            (v - (vocab as f32).ln()).abs() < 1e-4,
            "expected -ln(1/{vocab}) = {}, got {v}",
            (vocab as f32).ln()
        );
    }

    #[test]
    fn a_confidently_wrong_prediction_costs_more() {
        let device = Device::Cpu;
        let labels = Tensor::new(&[0i64], &device).unwrap();
        let right = Tensor::new(&[[8.0f32, 0.0, 0.0, 0.0]], &device).unwrap();
        let wrong = Tensor::new(&[[0.0f32, 0.0, 0.0, 8.0]], &device).unwrap();

        let l_right = scalar(&cross_entropy_loss(&right, &labels, 0.0).unwrap());
        let l_wrong = scalar(&cross_entropy_loss(&wrong, &labels, 0.0).unwrap());
        assert!(
            l_wrong > l_right,
            "wrong={l_wrong} must exceed right={l_right}; this is the assertion \
             the previous implementation inverted"
        );
    }

    #[test]
    fn label_smoothing_is_actually_applied() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[2.0f32, 0.5, -1.0, 0.25, 0.0]], &device).unwrap();
        let labels = Tensor::new(&[0i64], &device).unwrap();

        let plain = scalar(&cross_entropy_loss(&logits, &labels, 0.0).unwrap());
        let smooth = scalar(&cross_entropy_loss(&logits, &labels, 0.1).unwrap());
        assert!(
            (plain - smooth).abs() > 1e-5,
            "smoothing must change the value: {plain} vs {smooth}"
        );
    }

    #[test]
    fn negative_labels_are_masked_out() {
        let device = Device::Cpu;
        // Three positions, middle one padded with IGNORE_INDEX.
        let logits = Tensor::new(
            &[
                [1.0f32, 0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0, 5.0],
                [0.0, 0.0, 2.0, 0.0],
            ],
            &device,
        )
        .unwrap();
        let labels = Tensor::new(&[2i64, IGNORE_INDEX, 2], &device).unwrap();

        let out = cross_entropy(&logits, &labels, 0.0).unwrap();
        assert_eq!(out.unmasked, 2);

        // The padded row's gradient is exactly zero.
        let g: Vec<f32> = out.grad.flatten_all().unwrap().to_vec1().unwrap();
        assert!(g[4..8].iter().all(|x| *x == 0.0), "pad gradient must vanish: {g:?}");
        assert!(g[0..4].iter().any(|x| *x != 0.0));

        // And it equals the loss over just the two real positions.
        let kept = Tensor::new(&[[1.0f32, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]], &device).unwrap();
        let kept_labels = Tensor::new(&[2i64, 2], &device).unwrap();
        let direct = scalar(&cross_entropy_loss(&kept, &kept_labels, 0.0).unwrap());
        assert!((out.value - direct).abs() < 1e-5, "{} vs {direct}", out.value);
    }

    #[test]
    fn gradient_has_the_logits_shape_and_rows_sum_to_zero() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[0.3f32, -1.1, 2.0], [0.5, 0.25, -0.75]], &device).unwrap();
        let labels = Tensor::new(&[2i64, 0], &device).unwrap();

        let out = cross_entropy(&logits, &labels, 0.0).unwrap();
        assert_eq!(out.grad.shape(), logits.shape());

        // softmax - onehot sums to zero within each position.
        let g: Vec<f32> = out.grad.flatten_all().unwrap().to_vec1().unwrap();
        assert!(g[0..3].iter().sum::<f32>().abs() < 1e-5);
        assert!(g[3..6].iter().sum::<f32>().abs() < 1e-5);
    }

    #[test]
    fn matches_the_masked_candle_path_used_by_the_trainer() {
        // trainer.rs computes its own masked NLL with candle ops. That path and
        // this one must agree, or the number in the log is not the number being
        // optimised.
        use candle_nn::ops::log_softmax;
        let device = Device::Cpu;
        let logits = Tensor::new(
            &[
                [0.4f32, -1.2, 2.5, 0.1],
                [1.0, 0.0, -0.5, 0.25],
                [0.0, 0.0, 0.0, 0.0],
            ],
            &device,
        )
        .unwrap();
        let labels = Tensor::new(&[2i64, 0, IGNORE_INDEX], &device).unwrap();

        let ours = cross_entropy(&logits, &labels, 0.0).unwrap().value;

        // The trainer's formulation, verbatim in spirit.
        let lp = log_softmax(&logits, D::Minus1).unwrap();
        let safe = labels.clamp(0i64, i64::MAX).unwrap();
        let picked = lp
            .gather(&safe.unsqueeze(1).unwrap(), 1)
            .unwrap()
            .squeeze(1)
            .unwrap();
        // NB: `labels.ge(&Tensor::new(&[0i64], dev))` — which is what
        // trainer.rs:290 does — fails with "shape mismatch in cmp, lhs: [3],
        // rhs: [1]"; candle's `ge` does not broadcast. Comparing against a
        // same-shaped tensor is the form that works.
        let mask = labels
            .ge(&Tensor::zeros_like(&labels).unwrap())
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let nll = (picked.neg().unwrap() * &mask).unwrap();
        let theirs = (nll.sum_all().unwrap() / mask.sum_all().unwrap())
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(
            (ours - theirs).abs() < 1e-5,
            "optimizr path {ours} disagrees with the trainer's candle path {theirs}"
        );
    }

    #[test]
    fn every_position_masked_is_a_typed_error() {
        let device = Device::Cpu;
        let logits = Tensor::zeros((2, 4), DType::F32, &device).unwrap();
        let labels = Tensor::new(&[IGNORE_INDEX, IGNORE_INDEX], &device).unwrap();
        assert!(cross_entropy(&logits, &labels, 0.0).is_err());
    }
}
