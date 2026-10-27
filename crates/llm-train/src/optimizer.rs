use std::collections::HashMap;

use candle_core::{DType, Tensor};

/// AdamW optimizer with decoupled weight decay.
///
/// The timestep advances once per [`AdamWOptimizer::step`], not once per
/// parameter. That distinction is the whole of the bias correction: with two
/// parameters per update, incrementing inside the per-parameter routine made
/// `t` run twice too fast, so `1 - β₁ᵗ` and `1 - β₂ᵗ` were corrected for a
/// step that had not happened and the first updates were damped by the wrong
/// factor.
pub struct AdamWOptimizer {
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step_count: usize,
    m: HashMap<String, Tensor>,
    v: HashMap<String, Tensor>,
}

impl AdamWOptimizer {
    /// `_lr` is accepted and ignored: the learning rate comes from the
    /// scheduler, per step, through [`AdamWOptimizer::update_at`]. Keeping
    /// the argument means callers that pass a nominal rate still compile,
    /// and the name says it does nothing.
    pub fn new(_lr: f32, weight_decay: f32) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    /// Advance the timestep. Call once per optimiser step, before the
    /// parameters of that step are updated.
    pub fn step(&mut self) {
        self.step_count += 1;
    }

    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Update one parameter against its gradient at learning rate `lr`.
    ///
    /// `lr` is passed per call rather than read from the optimiser so the
    /// scheduler stays the single source of truth for it — the warmup and
    /// cosine decay in `scheduler.rs` were previously computed and then
    /// discarded.
    pub fn update_at(
        &mut self,
        name: &str,
        param: &mut Tensor,
        grad: &Tensor,
        lr: f32,
    ) -> Result<(), candle_core::Error> {
        debug_assert!(
            self.step_count > 0,
            "call step() once before updating this step's parameters"
        );
        let lr = lr as f64;
        let grad = grad.to_dtype(DType::F32)?;
        let param_f32 = param.to_dtype(DType::F32)?;

        // Initialize moments
        if !self.m.contains_key(name) {
            self.m.insert(name.to_string(), Tensor::zeros_like(&param_f32)?);
            self.v.insert(name.to_string(), Tensor::zeros_like(&param_f32)?);
        }

        let m = self.m.get_mut(name).unwrap();
        let v = self.v.get_mut(name).unwrap();

        let g = grad.clone();

        // m_t = beta1 * m_{t-1} + (1 - beta1) * g
        *m = m.affine(self.beta1 as f64, 0.0)?;
        let g_1mb1 = g.affine((1.0 - self.beta1) as f64, 0.0)?;
        *m = (m.clone() + g_1mb1)?;

        // v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
        let g_sq = g.powf(2.0)?;
        *v = v.affine(self.beta2 as f64, 0.0)?;
        let g_sq_1mb2 = g_sq.affine((1.0 - self.beta2) as f64, 0.0)?;
        *v = (v.clone() + g_sq_1mb2)?;

        // Bias correction
        let b1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let b2 = 1.0 - self.beta2.powi(self.step_count as i32);
        let m_hat = m.affine(1.0 / b1 as f64, 0.0)?;
        let v_hat = v.affine(1.0 / b2 as f64, 0.0)?;

        // Update: param -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * param)
        //
        // The weight-decay term is applied to the parameter itself rather than
        // folded into the gradient — this is exactly what makes it *decoupled*
        // (Loshchilov & Hutter, 2019) and what distinguishes AdamW from
        // Adam+L2: the decay is not rescaled by the adaptive denominator
        // sqrt(v_hat). See docs/source/algorithms/training.rst.
        let denom = (v_hat.sqrt()?.affine(1.0, self.eps as f64))?;
        let update = m_hat.broadcast_div(&denom)?;
        let decay = param_f32.affine(self.weight_decay as f64, 0.0)?;
        let update = (update + decay)?.affine(lr, 0.0)?;
        *param = (param_f32 - update)?.to_dtype(param.dtype())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_timestep_advances_once_per_step_not_once_per_parameter() {
        let device = candle_core::Device::Cpu;
        let mut opt = AdamWOptimizer::new(0.1, 0.0);
        let grad = Tensor::ones((2,), DType::F32, &device).unwrap();
        opt.step();
        for name in ["a", "b", "c"] {
            let mut p = Tensor::zeros((2,), DType::F32, &device).unwrap();
            opt.update_at(name, &mut p, &grad, 0.1).unwrap();
        }
        assert_eq!(opt.step_count(), 1, "three parameters, one step");
    }

    #[test]
    fn the_scheduler_learning_rate_is_the_one_applied() {
        // Zero moments, unit gradient: the first update is almost exactly
        // -lr, so a different lr must produce a proportionally different move.
        let device = candle_core::Device::Cpu;
        let grad = Tensor::ones((1,), DType::F32, &device).unwrap();
        let moved = |lr: f32| {
            let mut opt = AdamWOptimizer::new(999.0, 0.0); // constructor lr ignored
            let mut p = Tensor::zeros((1,), DType::F32, &device).unwrap();
            opt.step();
            opt.update_at("p", &mut p, &grad, lr).unwrap();
            -p.to_vec1::<f32>().unwrap()[0]
        };
        assert!((moved(0.1) - 0.1).abs() < 1e-3, "{}", moved(0.1));
        assert!((moved(0.01) - 0.01).abs() < 1e-4);
    }

    #[test]
    fn weight_decay_is_applied() {
        // With a zero gradient, plain Adam would leave the parameter
        // untouched; AdamW must still shrink it by lr * wd * param.
        let device = candle_core::Device::Cpu;
        let mut opt = AdamWOptimizer::new(0.1, 0.5);
        let mut param = Tensor::ones((4,), DType::F32, &device).unwrap();
        let grad = Tensor::zeros((4,), DType::F32, &device).unwrap();
        opt.step();
        opt.update_at("p", &mut param, &grad, 0.1).unwrap();
        let v = param.to_vec1::<f32>().unwrap()[0];
        // expected: 1 - 0.1 * (0 + 0.5 * 1) = 0.95
        assert!((v - 0.95).abs() < 1e-5, "decay not applied: got {v}");
    }

    #[test]
    fn optimizer_step() {
        let mut opt = AdamWOptimizer::new(0.001, 0.01);
        assert_eq!(opt.step_count(), 0);

        let device = candle_core::Device::Cpu;
        let mut param = Tensor::zeros((10,), DType::F32, &device).unwrap();
        let grad = Tensor::ones((10,), DType::F32, &device).unwrap();
        opt.step();
        opt.update_at("test", &mut param, &grad, 0.001).unwrap();
        assert_eq!(opt.step_count(), 1);
    }
}
