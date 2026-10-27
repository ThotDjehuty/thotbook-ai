//! LoRA adapters, forward and backward.
//!
//! An adapter factorises the update to a frozen weight as a product of two
//! thin matrices (Hu et al., 2021):
//!
//! ```text
//! out = scale · (x · A) · B,    A: [in, rank],  B: [rank, out],  scale = α/rank
//! ```
//!
//! `B` starts at zero, so the adapter is exactly the identity on the first
//! forward and the model it adapts is unchanged until the first optimiser
//! step. `A` is drawn from a Kaiming-uniform distribution; its scale matters
//! because `x·A` feeds straight into the gradient of `B`.
//!
//! [`LoraAdapter::backward`] is written out by hand rather than taped. The
//! factorisation is two matrix products, so its differential is three more,
//! and an explicit backward avoids holding an autograd graph over a
//! `positions × vocabulary` tensor — the thing that does not fit in memory on
//! the CPU this is meant to run on.

use candle_core::{DType, Tensor};

/// LoRA adapter layer configuration.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Rank of the low-rank matrices.
    pub rank: usize,
    /// Scaling factor (alpha / rank).
    pub alpha: usize,
    /// Dropout probability for LoRA layers.
    pub dropout: f32,
    /// Target modules to apply LoRA to (e.g., ["q_proj", "v_proj"]).
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16,
            dropout: 0.05,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
        }
    }
}

/// `∂loss/∂A` and `∂loss/∂B` for one adapter.
#[derive(Debug, Clone)]
pub struct LoraGrads {
    pub a: Tensor,
    pub b: Tensor,
}

/// LoRA adapter weights for a single layer.
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Low-rank matrix A (in_features × rank).
    pub a: Tensor,
    /// Low-rank matrix B (rank × out_features).
    pub b: Tensor,
    /// Scaling factor.
    pub scale: f32,
}

impl LoraAdapter {
    /// Create a new adapter: `A` Kaiming-uniform, `B` zero.
    ///
    /// The previous initialisation drew `A` from `N(0, 1)` while its comment
    /// claimed Kaiming. With `B = 0` that is invisible in the forward pass —
    /// the product is zero either way — but `∂loss/∂B` is proportional to
    /// `x·A`, so an `A` that is `sqrt(in_features)` too large makes the very
    /// first gradient step enormous.
    pub fn new(
        in_features: usize,
        out_features: usize,
        config: &LoraConfig,
        device: &candle_core::Device,
    ) -> Result<Self, candle_core::Error> {
        let rank = config.rank;
        let scale = config.alpha as f32 / rank as f32;

        // Kaiming uniform for a fan-in of `in_features`: U(-b, b) with
        // b = sqrt(6 / fan_in), the gain-1 case of He et al. (2015).
        let bound = (6.0 / in_features as f64).sqrt();
        let a = Tensor::rand(-bound as f32, bound as f32, (in_features, rank), device)?;
        let b = Tensor::zeros((rank, out_features), DType::F32, device)?;

        Ok(Self { a, b, scale })
    }

    pub fn rank(&self) -> usize {
        self.b.dims()[0]
    }

    /// `scale · (x · A) · B` for `x: [n, in]`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        self.hidden(input)?
            .matmul(&self.b)?
            .affine(self.scale as f64, 0.0)
    }

    /// The inner product `x · A`, shared by the forward and the backward.
    fn hidden(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        input.matmul(&self.a)
    }

    /// Gradients of the loss with respect to `A` and `B`.
    ///
    /// Given `g = ∂loss/∂out` with the same shape as [`LoraAdapter::forward`]'s
    /// output:
    ///
    /// ```text
    /// ∂loss/∂B = scale · (x·A)ᵀ · g
    /// ∂loss/∂A = scale · xᵀ · (g · Bᵀ)
    /// ```
    ///
    /// Both follow from differentiating `out = scale·(x·A)·B` once; the tests
    /// check them against central finite differences rather than against the
    /// derivation.
    pub fn backward(
        &self,
        input: &Tensor,
        grad_out: &Tensor,
    ) -> Result<LoraGrads, candle_core::Error> {
        let scale = self.scale as f64;

        let xa = self.hidden(input)?; // [n, rank]
        let grad_b = xa.t()?.matmul(grad_out)?.affine(scale, 0.0)?;

        let g_bt = grad_out.matmul(&self.b.t()?)?; // [n, rank]
        let grad_a = input.t()?.matmul(&g_bt)?.affine(scale, 0.0)?;

        Ok(LoraGrads {
            a: grad_a,
            b: grad_b,
        })
    }

    /// `∂loss/∂x`, for propagating further back into the frozen body.
    ///
    /// Unused while the body is frozen; written because it is the one piece a
    /// per-layer adapter would need, and leaving it out would hide how much
    /// of the chain is actually implemented.
    pub fn backward_input(&self, grad_out: &Tensor) -> Result<Tensor, candle_core::Error> {
        grad_out
            .matmul(&self.b.t()?)?
            .matmul(&self.a.t()?)?
            .affine(self.scale as f64, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn cfg(rank: usize) -> LoraConfig {
        LoraConfig {
            rank,
            alpha: rank, // scale 1, so the finite differences are unscaled
            ..Default::default()
        }
    }

    fn adapter(in_f: usize, out_f: usize, rank: usize) -> LoraAdapter {
        let d = Device::Cpu;
        let mut a = LoraAdapter::new(in_f, out_f, &cfg(rank), &d).unwrap();
        // B starts at zero, which makes ∂loss/∂A identically zero; give it
        // something to differentiate.
        a.b = Tensor::rand(-0.5f32, 0.5, (rank, out_f), &d).unwrap();
        a
    }

    /// A scalar objective with a known derivative: `loss = Σ (w ⊙ out)`,
    /// hence `∂loss/∂out = w`.
    fn loss(adapter: &LoraAdapter, x: &Tensor, w: &Tensor) -> f64 {
        let out = adapter.forward(x).unwrap();
        (out * w).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap() as f64
    }

    fn perturb(t: &Tensor, idx: usize, eps: f64) -> Tensor {
        let dims = t.dims().to_vec();
        let mut flat = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        flat[idx] += eps as f32;
        Tensor::from_vec(flat, dims, &Device::Cpu).unwrap()
    }

    #[test]
    fn b_starts_at_zero_so_the_adapter_is_initially_the_identity() {
        let d = Device::Cpu;
        let a = LoraAdapter::new(16, 4, &cfg(2), &d).unwrap();
        let x = Tensor::rand(-1.0f32, 1.0, (3, 16), &d).unwrap();
        let out = a.forward(&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(out.iter().all(|v| *v == 0.0), "adapter must start as a no-op");
    }

    #[test]
    fn kaiming_bound_scales_with_fan_in() {
        let d = Device::Cpu;
        let wide = LoraAdapter::new(1024, 4, &cfg(2), &d).unwrap();
        let narrow = LoraAdapter::new(16, 4, &cfg(2), &d).unwrap();
        let max = |t: &Tensor| {
            t.abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };
        // sqrt(6/1024) ≈ 0.077 vs sqrt(6/16) ≈ 0.61
        assert!(max(&wide.a) < 0.08, "{}", max(&wide.a));
        assert!(max(&narrow.a) < 0.62);
        assert!(max(&wide.a) < max(&narrow.a));
    }

    #[test]
    fn grad_b_matches_finite_differences() {
        let d = Device::Cpu;
        let mut ad = adapter(12, 5, 3);
        let x = Tensor::rand(-1.0f32, 1.0, (4, 12), &d).unwrap();
        let w = Tensor::rand(-1.0f32, 1.0, (4, 5), &d).unwrap();

        let analytic = ad
            .backward(&x, &w)
            .unwrap()
            .b
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let eps = 1e-3;
        let b0 = ad.b.clone();
        for idx in 0..analytic.len() {
            ad.b = perturb(&b0, idx, eps);
            let up = loss(&ad, &x, &w);
            ad.b = perturb(&b0, idx, -eps);
            let down = loss(&ad, &x, &w);
            let numeric = (up - down) / (2.0 * eps);
            assert!(
                (numeric - analytic[idx] as f64).abs() < 1e-2,
                "dB[{idx}]: analytic {} vs numeric {numeric}",
                analytic[idx]
            );
        }
    }

    #[test]
    fn grad_a_matches_finite_differences() {
        let d = Device::Cpu;
        let mut ad = adapter(9, 6, 2);
        let x = Tensor::rand(-1.0f32, 1.0, (5, 9), &d).unwrap();
        let w = Tensor::rand(-1.0f32, 1.0, (5, 6), &d).unwrap();

        let analytic = ad
            .backward(&x, &w)
            .unwrap()
            .a
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let eps = 1e-3;
        let a0 = ad.a.clone();
        for idx in 0..analytic.len() {
            ad.a = perturb(&a0, idx, eps);
            let up = loss(&ad, &x, &w);
            ad.a = perturb(&a0, idx, -eps);
            let down = loss(&ad, &x, &w);
            let numeric = (up - down) / (2.0 * eps);
            assert!(
                (numeric - analytic[idx] as f64).abs() < 1e-2,
                "dA[{idx}]: analytic {} vs numeric {numeric}",
                analytic[idx]
            );
        }
    }

    #[test]
    fn scale_multiplies_both_gradients() {
        let d = Device::Cpu;
        let mut ad = adapter(8, 4, 2);
        let x = Tensor::rand(-1.0f32, 1.0, (3, 8), &d).unwrap();
        let g = Tensor::rand(-1.0f32, 1.0, (3, 4), &d).unwrap();
        let at_one = ad.backward(&x, &g).unwrap();
        ad.scale = 2.0;
        let at_two = ad.backward(&x, &g).unwrap();

        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (one, two) in flat(&at_one.b).iter().zip(flat(&at_two.b)) {
            assert!((two - 2.0 * one).abs() < 1e-5);
        }
        for (one, two) in flat(&at_one.a).iter().zip(flat(&at_two.a)) {
            assert!((two - 2.0 * one).abs() < 1e-5);
        }
    }

    #[test]
    fn lora_output_shape() {
        let device = Device::Cpu;
        let config = LoraConfig::default();
        let adapter = LoraAdapter::new(64, 32, &config, &device).unwrap();

        let input = Tensor::zeros((10, 64), DType::F32, &device).unwrap();
        let output = adapter.forward(&input).unwrap();
        assert_eq!(output.dims(), &[10, 32]);
    }

    #[test]
    fn gradient_shapes_match_the_parameters() {
        let d = Device::Cpu;
        let ad = adapter(64, 32, 8);
        let x = Tensor::rand(-1.0f32, 1.0, (10, 64), &d).unwrap();
        let g = Tensor::rand(-1.0f32, 1.0, (10, 32), &d).unwrap();
        let grads = ad.backward(&x, &g).unwrap();
        assert_eq!(grads.a.dims(), ad.a.dims());
        assert_eq!(grads.b.dims(), ad.b.dims());
        assert_eq!(ad.backward_input(&g).unwrap().dims(), x.dims());
    }
}
