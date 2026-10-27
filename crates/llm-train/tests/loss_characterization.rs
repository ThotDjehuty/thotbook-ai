//! Regression tests for the training objective.
//!
//! These replace the characterization suite that recorded three defects in the
//! old hand-rolled `cross_entropy_loss`:
//!
//!   1. SIGN. It returned `+mean(log q)` instead of `-mean(log q)`, so
//!      minimising it trained the model *away* from its targets.
//!   2. LABEL SMOOTHING. The argument was accepted, documented, and ignored.
//!   3. MASKING. There was no `ignore_index` and no mask, despite the paper's
//!      section 3.1 being titled "Cross-Entropy Loss with Masking".
//!
//! `crates/llm-train/src/loss.rs` now delegates to `optimizr::losses`, whose
//! gradients are checked against central finite differences in optimiz-rs's
//! own suite. The assertions below are the inverted form of the three defect
//! recordings, so a regression on any of them fails here.

use candle_core::{DType, Device, Tensor};
use llm_train::loss::{cross_entropy, cross_entropy_loss, IGNORE_INDEX};

fn scalar(t: &Tensor) -> f32 {
    t.to_scalar::<f32>().unwrap()
}

/// -log(1/V) for a uniform distribution over V classes: the loss a freshly
/// initialised model should report.
fn uniform_ce(vocab: usize) -> f32 {
    -(1.0f32 / vocab as f32).ln()
}

#[test]
fn loss_is_non_negative_cross_entropy() {
    let device = Device::Cpu;
    let vocab = 10;
    let logits = Tensor::zeros((2, vocab), DType::F32, &device).unwrap();
    let labels = Tensor::new(&[1i64, 2], &device).unwrap();

    let got = scalar(&cross_entropy_loss(&logits, &labels, 0.0).unwrap());

    assert!(got >= 0.0, "cross-entropy must be non-negative, got {got}");
    assert!(
        (got - uniform_ce(vocab)).abs() < 1e-4,
        "uniform logits over {vocab} classes must give {}, got {got}",
        uniform_ce(vocab)
    );
}

#[test]
fn a_confident_wrong_prediction_is_penalised() {
    let device = Device::Cpu;
    let labels = Tensor::new(&[0i64], &device).unwrap();
    let right = Tensor::new(&[[8.0f32, 0.0, 0.0, 0.0]], &device).unwrap();
    let wrong = Tensor::new(&[[0.0f32, 0.0, 0.0, 8.0]], &device).unwrap();

    let l_right = scalar(&cross_entropy_loss(&right, &labels, 0.0).unwrap());
    let l_wrong = scalar(&cross_entropy_loss(&wrong, &labels, 0.0).unwrap());

    assert!(
        l_wrong > l_right,
        "being wrong must cost more: wrong={l_wrong}, right={l_right}"
    );
    assert!(l_right >= 0.0 && l_wrong >= 0.0);
}

#[test]
fn label_smoothing_argument_has_an_effect() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[[2.0f32, 1.0, 0.5, -1.0]], &device).unwrap();
    let labels = Tensor::new(&[0i64], &device).unwrap();

    let none = scalar(&cross_entropy_loss(&logits, &labels, 0.0).unwrap());
    let heavy = scalar(&cross_entropy_loss(&logits, &labels, 0.5).unwrap());

    assert!(
        (none - heavy).abs() > 1e-5,
        "smoothing 0.0 and 0.5 must differ, got {none} and {heavy}"
    );
}

#[test]
fn padding_positions_can_be_masked() {
    let device = Device::Cpu;
    let logits = Tensor::new(
        &[
            [1.0f32, 0.0, 0.0, 0.0],
            [5.0, 5.0, 5.0, 5.0],
            [0.0, 0.0, 2.0, 0.0],
        ],
        &device,
    )
    .unwrap();
    let with_pad = Tensor::new(&[2i64, IGNORE_INDEX, 2], &device).unwrap();

    let out = cross_entropy(&logits, &with_pad, 0.0).unwrap();
    assert_eq!(out.unmasked, 2, "the padded position must not be counted");

    // Masked rows contribute no gradient at all.
    let g: Vec<f32> = out.grad.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        g[4..8].iter().all(|x| *x == 0.0),
        "padding must get a zero gradient, got {:?}",
        &g[4..8]
    );

    // The masked loss equals the loss over only the real positions.
    let kept = Tensor::new(&[[1.0f32, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]], &device).unwrap();
    let kept_labels = Tensor::new(&[2i64, 2], &device).unwrap();
    let direct = scalar(&cross_entropy_loss(&kept, &kept_labels, 0.0).unwrap());
    assert!(
        (out.value - direct).abs() < 1e-5,
        "masked {} should equal unpadded {direct}",
        out.value
    );
}

#[test]
fn gradient_is_supplied_so_training_needs_no_autograd_tape() {
    // The gradient w.r.t. logits is softmax - onehot, scaled by 1/N. Stepping
    // against it must raise the target logit; this is the end-to-end property
    // the sign defect violated.
    let device = Device::Cpu;
    let labels = Tensor::new(&[0i64], &device).unwrap();
    let mut logits = vec![0.0f32, 0.0, 0.0];

    let first = {
        let t = Tensor::from_vec(logits.clone(), (1, 3), &device).unwrap();
        cross_entropy(&t, &labels, 0.0).unwrap().value
    };

    for _ in 0..200 {
        let t = Tensor::from_vec(logits.clone(), (1, 3), &device).unwrap();
        let out = cross_entropy(&t, &labels, 0.0).unwrap();
        let g: Vec<f32> = out.grad.flatten_all().unwrap().to_vec1().unwrap();
        for (z, gi) in logits.iter_mut().zip(&g) {
            *z -= 0.5 * gi;
        }
    }

    let last = {
        let t = Tensor::from_vec(logits.clone(), (1, 3), &device).unwrap();
        cross_entropy(&t, &labels, 0.0).unwrap().value
    };

    assert!(last < first, "loss must fall under descent: {first} -> {last}");
    assert!(
        logits[0] > logits[1] && logits[0] > logits[2],
        "the target logit must end up largest: {logits:?}"
    );
}

#[test]
fn the_objective_comes_from_optimiz_rs() {
    // Guard the dependency itself: llm-train must not grow a second, private
    // definition of cross-entropy. If this stops compiling, the loss was
    // re-implemented locally instead of being taken from optimiz-rs.
    use optimizr::losses::{CrossEntropy, Loss};

    let device = Device::Cpu;
    let vocab = 5;
    let raw = [0.3f32, -1.1, 2.0, 0.5, 0.25];
    let logits = Tensor::new(&[raw], &device).unwrap();
    let labels = Tensor::new(&[2i64], &device).unwrap();

    let through_llm_train = scalar(&cross_entropy_loss(&logits, &labels, 0.0).unwrap());
    let direct = CrossEntropy::new(vocab)
        .value(&raw.iter().map(|&x| x as f64).collect::<Vec<_>>(), &[2usize])
        .unwrap() as f32;

    assert!(
        (through_llm_train - direct).abs() < 1e-5,
        "llm-train {through_llm_train} must equal optimizr {direct}"
    );
}
