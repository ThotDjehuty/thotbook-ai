/// Cosine learning rate schedule with linear warmup.
pub struct CosineScheduler {
    warmup_steps: usize,
    total_steps: usize,
    max_lr: f32,
    min_lr: f32,
}

impl CosineScheduler {
    pub fn new(warmup_steps: usize, total_steps: usize, max_lr: f32, min_lr: f32) -> Self {
        Self {
            warmup_steps,
            total_steps,
            max_lr,
            min_lr,
        }
    }

    /// Get the learning rate for a given step.
    pub fn lr_at(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            self.max_lr * step as f32 / self.warmup_steps as f32
        } else {
            // Cosine decay
            let progress = (step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps).max(1) as f32;
            let progress = progress.min(1.0);
            self.min_lr
                + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (progress * std::f32::consts::PI).cos())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_phase() {
        let sched = CosineScheduler::new(100, 1000, 1e-3, 1e-5);
        assert!((sched.lr_at(0) - 0.0).abs() < 1e-6);
        assert!((sched.lr_at(50) - 5e-4).abs() < 1e-4);
        assert!((sched.lr_at(100) - 1e-3).abs() < 1e-4);
    }

    #[test]
    fn cosine_decay() {
        let sched = CosineScheduler::new(100, 1000, 1e-3, 1e-5);
        let lr_mid = sched.lr_at(550);
        assert!(lr_mid < 1e-3);
        assert!(lr_mid > 1e-5);
    }
}
