//! Training loop with real LoRA fine-tuning on Qwen3 models.
//!
//! This implements actual forward pass, loss computation, and gradient descent.

use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use optimizr::losses::{CrossEntropy, Loss};

use crate::checkpoint;
use crate::dataset::{StreamingDataset, IGNORE_LABEL};
use crate::error::TrainError;
use crate::grads::{Grads, LossSum, Step};
use crate::lora::LoraConfig;
use crate::metrics::{MetricsLogger, TrainingMetrics, perplexity};
use crate::model::{self, ModelConfig, TrainableModel};
use crate::optimizer::AdamWOptimizer;
use crate::scheduler::CosineScheduler;

/// Training mode.
#[derive(Debug, Clone, PartialEq)]
pub enum TrainMode {
    /// LoRA adapter training on a frozen base model.
    Lora,
    /// Full fine-tuning of all parameters.
    Full,
}

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub mode: TrainMode,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub warmup_steps: usize,
    pub max_grad_norm: f32,
    pub label_smoothing: f32,
    pub checkpoint_dir: PathBuf,
    pub metrics_path: PathBuf,
    pub lora_config: Option<LoraConfig>,
    pub model_dir: PathBuf,
    pub tokenizer_path: PathBuf,
    pub max_seq_len: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            mode: TrainMode::Lora,
            epochs: 3,
            batch_size: 1, // Small batch for CPU
            learning_rate: 2e-4,
            weight_decay: 0.01,
            warmup_steps: 100,
            max_grad_norm: 1.0,
            label_smoothing: 0.1,
            checkpoint_dir: PathBuf::from("checkpoints"),
            metrics_path: PathBuf::from("training_metrics.jsonl"),
            lora_config: Some(LoraConfig::default()),
            model_dir: PathBuf::from("data/models/qwen3-4b"),
            tokenizer_path: PathBuf::from("data/tokenizers/qwen3-tokenizer.json"),
            max_seq_len: 512,
        }
    }
}

/// The training engine.
pub struct Trainer {
    config: TrainingConfig,
    device: Device,
    model: Option<TrainableModel>,
}

impl Trainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            device: Device::Cpu,
            model: None,
        }
    }

    /// Load the model. Call before train().
    pub fn load_model(&mut self) -> Result<(), TrainError> {
        let model_config = ModelConfig::new(
            &self.config.model_dir,
            &self.config.tokenizer_path,
        );
        let lora_config = self.config.lora_config.clone().unwrap_or_default();
        
        let model = TrainableModel::load(&model_config, &lora_config)?;
        self.model = Some(model);
        Ok(())
    }

    /// Run the training loop with real forward/backward passes.
    pub fn train(&mut self, dataset: &mut StreamingDataset) -> Result<(), TrainError> {
        std::fs::create_dir_all(&self.config.checkpoint_dir).map_err(|e| TrainError::Io {
            path: self.config.checkpoint_dir.clone(),
            source: e,
        })?;

        let model = self.model.as_mut().ok_or_else(|| {
            TrainError::Model("model not loaded, call load_model() first".to_string())
        })?;

        let total_steps = dataset.len() / self.config.batch_size * self.config.epochs;
        let scheduler = CosineScheduler::new(
            self.config.warmup_steps,
            total_steps,
            self.config.learning_rate,
            self.config.learning_rate * 0.1,
        );
        let logger = MetricsLogger::new(&self.config.metrics_path);
        let mut optimizer =
            AdamWOptimizer::new(self.config.learning_rate, self.config.weight_decay);
        let step_cfg = StepConfig {
            max_seq_len: self.config.max_seq_len,
            label_smoothing: self.config.label_smoothing,
            max_grad_norm: self.config.max_grad_norm,
            logit_chunk: 16,
        };

        tracing::info!(
            epochs = self.config.epochs,
            batch_size = self.config.batch_size,
            total_steps = total_steps,
            "starting training"
        );

        let mut global_step = 0;

        for epoch in 0..self.config.epochs {
            tracing::info!(epoch, "starting epoch");
            let epoch_start = Instant::now();

            while let Some(batch) = dataset.next_batch() {

                let step_start = Instant::now();
                let lr = scheduler.lr_at(global_step);

                let report = training_step(
                    batch,
                    model,
                    &mut optimizer,
                    lr,
                    &step_cfg,
                    &self.device,
                )?;
                let loss = report.loss;
                let step_time = step_start.elapsed();

                let tokens_processed: usize = batch.iter().map(|ex| ex.input_ids.len()).sum();
                let tokens_per_second = tokens_processed as f32 / step_time.as_secs_f32();

                let metrics = TrainingMetrics {
                    step: global_step,
                    loss,
                    learning_rate: lr,
                    perplexity: perplexity(loss),
                    tokens_per_second,
                };

                logger.log(&metrics);

                if global_step % 10 == 0 {
                    tracing::info!(
                        step = global_step,
                        loss = format!("{:.4}", metrics.loss),
                        lr = format!("{:.2e}", metrics.learning_rate),
                        ppl = format!("{:.2}", metrics.perplexity),
                        tps = format!("{:.1}", tokens_per_second),
                        gnorm = format!("{:.3}", report.grad_norm),
                        supervised = report.supervised,
                        updated = report.updated,
                        "training progress"
                    );
                }

                if global_step % 100 == 0 && global_step > 0 {
                    let ckpt_dir = self.config.checkpoint_dir.join(format!("step_{global_step}"));
                    model.save_lora(&ckpt_dir)?;
                    checkpoint::save_checkpoint(
                        &ckpt_dir.join("meta.json"),
                        global_step,
                        loss,
                        &[],
                    )?;
                }

                global_step += 1;
            }

            let epoch_time = epoch_start.elapsed();
            tracing::info!(
                epoch,
                duration_secs = epoch_time.as_secs(),
                "completed epoch"
            );
        }

        // Save final checkpoint
        let final_dir = self.config.checkpoint_dir.join("final");
        model.save_lora(&final_dir)?;
        tracing::info!(total_steps = global_step, "training complete");
        Ok(())
    }
}

/// Single training step with real forward pass and gradient descent.
/// Knobs for one step, grouped so the signature stays readable.
struct StepConfig {
    max_seq_len: usize,
    label_smoothing: f32,
    max_grad_norm: f32,
    /// Positions per logit chunk in the backward pass.
    ///
    /// The cost is `chunk × vocab` doubles held at once — about 19 MB per
    /// chunk at Qwen3's 151 936 classes. Small enough to stay resident,
    /// large enough that the matmul is not dominated by call overhead.
    logit_chunk: usize,
}

/// What one step did, beyond the loss.
struct StepReport {
    loss: f32,
    grad_norm: f64,
    /// Positions that carried a label, i.e. that the loss was taken over.
    supervised: usize,
    /// Parameter tensors the optimiser moved.
    updated: usize,
}

/// One optimiser step: forward, loss, backward, update.
///
/// # How far the gradient travels
///
/// The body is frozen, so the chain that matters is short and is written out
/// exactly rather than taped:
///
/// ```text
/// h ──(frozen head W)──┐
///                      ├─▶ logits ──▶ cross-entropy ──▶ loss
/// h ──(A, B, scale)────┘
///
/// ∂loss/∂logits   from optimizr::losses::CrossEntropy  (value_and_grad)
/// ∂loss/∂B        = scale · (h·A)ᵀ · ∂loss/∂logits
/// ∂loss/∂A        = scale · hᵀ · (∂loss/∂logits · Bᵀ)
/// ```
///
/// `∂loss/∂h` exists ([`LoraAdapter::backward_input`]) but is not used: with
/// the body frozen there is nothing behind `h` to update. That is the single
/// honest limit of this implementation, and the comment that used to sit here
/// — "full gradient-based training requires candle's autograd" — is no longer
/// true for the parameters that exist.
///
/// # Two exactness-preserving optimisations
///
/// *Only supervised positions are projected.* Cross-entropy ignores a masked
/// position entirely: zero loss, zero gradient. Computing its logits is
/// `hidden × vocab` of arithmetic thrown away, and with a prompt-masked
/// sample most positions are masked. Selecting the supervised rows first is
/// not an approximation; it is the same number computed over fewer rows.
///
/// *The pass is chunked and folded through a monoid.* See [`crate::grads`]:
/// per-chunk means cannot simply be added, so each chunk carries its own
/// weight and the normalisation happens once, at the end.
fn training_step(
    batch: &[crate::dataset::TrainingExample],
    model: &mut TrainableModel,
    optimizer: &mut AdamWOptimizer,
    lr: f32,
    cfg: &StepConfig,
    device: &Device,
) -> Result<StepReport, TrainError> {
    let vocab = model.vocab_size();
    let idle = |loss| StepReport {
        loss,
        grad_norm: 0.0,
        supervised: 0,
        updated: 0,
    };

    // ---- batch to tensors ------------------------------------------------
    let max_len = batch
        .iter()
        .map(|ex| ex.input_ids.len())
        .max()
        .unwrap_or(1)
        .min(cfg.max_seq_len);
    if max_len <= 1 {
        return Ok(idle(0.0));
    }

    let (ids_flat, labels_flat): (Vec<i64>, Vec<i64>) = batch.iter().fold(
        (Vec::new(), Vec::new()),
        |(mut ids_acc, mut lab_acc), ex| {
            let len = ex.input_ids.len().min(max_len);
            ids_acc.extend(ex.input_ids[..len].iter().map(|&x| x as i64));
            ids_acc.extend(std::iter::repeat(0).take(max_len - len));

            let lab_len = len.min(ex.labels.len());
            lab_acc.extend(ex.labels[..lab_len].iter().map(|&x| {
                if x == IGNORE_LABEL {
                    crate::loss::IGNORE_INDEX
                } else {
                    x as i64
                }
            }));
            lab_acc.extend(std::iter::repeat(crate::loss::IGNORE_INDEX).take(max_len - lab_len));
            (ids_acc, lab_acc)
        },
    );

    let b = batch.len();
    let candle = |e: candle_core::Error| TrainError::Candle(e.to_string());
    let input_ids = Tensor::from_vec(ids_flat, (b, max_len), device)
        .and_then(|t| t.to_dtype(DType::I64))
        .map_err(candle)?;

    // ---- forward through the frozen body --------------------------------
    // Each example is an independent sequence: the attention cache from the
    // previous step must not be visible to this one.
    model.reset_cache()?;
    let hidden = model.body(&input_ids)?;
    let (_, _, hidden_size) = hidden.dims3().map_err(candle)?;

    // Next-token prediction: position t predicts the label at t+1.
    let shifted = max_len - 1;
    let h_shift = hidden
        .narrow(1, 0, shifted)
        .and_then(|t| t.contiguous())
        .and_then(|t| t.reshape((b * shifted, hidden_size)))
        .map_err(candle)?;
    let targets_shift: Vec<i64> = (0..b)
        .flat_map(|row| labels_flat[row * max_len + 1..(row + 1) * max_len].iter().copied())
        .collect();

    // ---- keep only the positions the loss is actually taken over ---------
    let (keep, targets): (Vec<u32>, Vec<usize>) = targets_shift
        .iter()
        .enumerate()
        .filter(|(_, &l)| l >= 0)
        .map(|(i, &l)| (i as u32, l as usize))
        .unzip();
    let supervised = keep.len();
    if supervised == 0 {
        return Ok(idle(0.0));
    }

    let index = Tensor::from_vec(keep, (supervised,), device).map_err(candle)?;
    let h_sup = h_shift.index_select(&index, 0).map_err(candle)?;

    // ---- chunked backward, folded through the monoid ---------------------
    let adapter = model
        .adapter(model::HEAD_ADAPTER)
        .ok_or_else(|| TrainError::Model("no trainable adapter on the head".into()))?;

    let chunk = cfg.logit_chunk.max(1);
    let step = Step::concat((0..supervised).step_by(chunk).map(|start| {
        let m = chunk.min(supervised - start);
        let h_c = h_sup.narrow(0, start, m).map_err(candle)?;
        let logits = model.head_logits(&h_c)?;

        let rows: Vec<f64> = logits
            .to_dtype(DType::F64)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f64>())
            .map_err(candle)?;

        // optimizr::losses is the single definition of this objective; its
        // gradient is finite-difference checked there, so nothing is
        // re-derived here.
        let mut ce = CrossEntropy::new(vocab).with_ignore_index(vocab);
        if cfg.label_smoothing > 0.0 {
            ce = ce
                .with_label_smoothing(cfg.label_smoothing as f64)
                .map_err(|e| TrainError::Dataset(format!("optimizr::losses: {e}")))?;
        }
        let (value, grad) = ce
            .value_and_grad(&rows, &targets[start..start + m])
            .map_err(|e| TrainError::Dataset(format!("optimizr::losses: {e}")))?;

        // CrossEntropy divides by the positions in *its* call; undo that here
        // so the monoid sums comparable quantities and `finish` normalises
        // once, globally.
        let weight = m as f64;
        let g = Tensor::from_vec(
            grad.into_iter().map(|x| (x * weight) as f32).collect::<Vec<f32>>(),
            (m, vocab),
            device,
        )
        .map_err(candle)?;

        let lg = adapter.backward(&h_c, &g).map_err(candle)?;
        Ok(Step {
            loss: LossSum::of(value, m),
            grads: Grads::of(format!("{}.a", model::HEAD_ADAPTER), lg.a)
                .combine(Grads::of(format!("{}.b", model::HEAD_ADAPTER), lg.b))?,
        })
    }))?;

    let (loss, grads) = step.finish()?;
    let (grads, grad_norm) = grads.clipped(cfg.max_grad_norm as f64)?;

    // ---- update ----------------------------------------------------------
    optimizer.step();
    let names: Vec<String> = model.adapters().keys().cloned().collect();
    let updated = names.iter().try_fold(0usize, |acc, name| {
        let (ka, kb) = (format!("{name}.a"), format!("{name}.b"));
        match (grads.get(&ka).cloned(), grads.get(&kb).cloned()) {
            (Some(ga), Some(gb)) => {
                let adapter = model
                    .adapters_mut()
                    .get_mut(name)
                    .expect("name came from this map");
                optimizer.update_at(&ka, &mut adapter.a, &ga, lr).map_err(candle)?;
                optimizer.update_at(&kb, &mut adapter.b, &gb, lr).map_err(candle)?;
                Ok::<usize, TrainError>(acc + 2)
            }
            // No gradient reached this adapter: it is not in the forward
            // pass, so leaving it untouched is the correct outcome.
            _ => Ok(acc),
        }
    })?;

    Ok(StepReport {
        loss: loss as f32,
        grad_norm,
        supervised,
        updated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::TrainingExample;
    use std::collections::HashMap;

    /// Build a Qwen3 small enough to train in a unit test, write it out as
    /// safetensors, and load it through the real `TrainableModel::load`.
    ///
    /// Synthetic weights, real code path: the loader, the body, the head and
    /// the adapter are the ones used in production. Returns `None` when the
    /// repository has no tokenizer to point at, which is the only piece that
    /// cannot be synthesised cheaply.
    fn tiny_model(dir: &std::path::Path) -> Option<TrainableModel> {
        let tokenizer = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/tokenizers/qwen3-tokenizer.json"
        ));
        if !tokenizer.exists() {
            eprintln!("skipping: no tokenizer at {}", tokenizer.display());
            return None;
        }

        const V: usize = 64;
        const H: usize = 16;
        const I: usize = 32;
        const HEADS: usize = 2;
        const KV: usize = 1;
        const HEAD_DIM: usize = 8;

        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({
                "vocab_size": V, "hidden_size": H, "intermediate_size": I,
                "num_hidden_layers": 1, "num_attention_heads": HEADS,
                "num_key_value_heads": KV, "head_dim": HEAD_DIM,
                "attention_bias": false, "max_position_embeddings": 128,
                "max_window_layers": 1, "tie_word_embeddings": true,
                "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
                "use_sliding_window": false, "sliding_window": null,
                "hidden_act": "silu"
            })
            .to_string(),
        )
        .unwrap();

        let d = Device::Cpu;
        let rand = |dims: (usize, usize)| Tensor::rand(-0.08f32, 0.08, dims, &d).unwrap();
        let ones = |n: usize| Tensor::ones((n,), DType::F32, &d).unwrap();

        let mut w: HashMap<String, Tensor> = HashMap::new();
        w.insert("model.embed_tokens.weight".into(), rand((V, H)));
        w.insert("model.norm.weight".into(), ones(H));
        let l = "model.layers.0";
        w.insert(format!("{l}.input_layernorm.weight"), ones(H));
        w.insert(format!("{l}.post_attention_layernorm.weight"), ones(H));
        w.insert(format!("{l}.self_attn.q_proj.weight"), rand((HEADS * HEAD_DIM, H)));
        w.insert(format!("{l}.self_attn.k_proj.weight"), rand((KV * HEAD_DIM, H)));
        w.insert(format!("{l}.self_attn.v_proj.weight"), rand((KV * HEAD_DIM, H)));
        w.insert(format!("{l}.self_attn.o_proj.weight"), rand((H, HEADS * HEAD_DIM)));
        w.insert(format!("{l}.self_attn.q_norm.weight"), ones(HEAD_DIM));
        w.insert(format!("{l}.self_attn.k_norm.weight"), ones(HEAD_DIM));
        w.insert(format!("{l}.mlp.gate_proj.weight"), rand((I, H)));
        w.insert(format!("{l}.mlp.up_proj.weight"), rand((I, H)));
        w.insert(format!("{l}.mlp.down_proj.weight"), rand((H, I)));

        candle_core::safetensors::save(&w, dir.join("model.safetensors")).unwrap();

        let cfg = ModelConfig::new(dir, &tokenizer);
        Some(TrainableModel::load(&cfg, &LoraConfig { rank: 4, alpha: 8, ..Default::default() }).unwrap())
    }

    fn example() -> TrainingExample {
        // A prompt of 4 masked tokens, then 5 supervised ones.
        let input_ids: Vec<u32> = vec![3, 9, 14, 7, 21, 33, 5, 18, 2];
        let labels: Vec<u32> = input_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| if i < 4 { IGNORE_LABEL } else { id })
            .collect();
        TrainingExample { input_ids, labels }
    }

    fn step_cfg() -> StepConfig {
        StepConfig {
            max_seq_len: 32,
            label_smoothing: 0.0,
            max_grad_norm: 1.0,
            logit_chunk: 3, // deliberately not a divisor of the position count
        }
    }

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("trainer-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn a_step_moves_the_adapter_and_lowers_the_loss() {
        let dir = tmpdir("descent");
        let Some(mut model) = tiny_model(&dir) else { return };
        let mut opt = AdamWOptimizer::new(0.0, 0.0);
        let batch = vec![example()];
        let cfg = step_cfg();

        let before = model.adapter(model::HEAD_ADAPTER).unwrap().b.clone();
        let first = training_step(&batch, &mut model, &mut opt, 0.05, &cfg, &Device::Cpu).unwrap();
        assert_eq!(first.updated, 2, "both A and B must be updated");
        assert_eq!(first.supervised, 5, "only the unmasked positions count");
        assert!(first.loss > 0.0 && first.loss.is_finite(), "{}", first.loss);
        assert!(first.grad_norm > 0.0, "a gradient must have reached the adapter");

        let after = model.adapter(model::HEAD_ADAPTER).unwrap().b.clone();
        let moved = (&after - &before)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(moved > 0.0, "B must change; it did not");

        // Keep stepping on the same example: the loss has to come down.
        let mut last = first.loss;
        for _ in 0..12 {
            last = training_step(&batch, &mut model, &mut opt, 0.05, &cfg, &Device::Cpu)
                .unwrap()
                .loss;
        }
        assert!(
            last < first.loss,
            "loss did not decrease: {} -> {last}",
            first.loss
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn every_saved_adapter_received_a_gradient() {
        // The previous implementation created 112 adapters, applied none of
        // them, and still wrote them all out as a "trained" checkpoint.
        let dir = tmpdir("saved");
        let Some(mut model) = tiny_model(&dir) else { return };
        let mut opt = AdamWOptimizer::new(0.0, 0.0);
        let report =
            training_step(&vec![example()], &mut model, &mut opt, 0.01, &step_cfg(), &Device::Cpu)
                .unwrap();

        let saved = model.adapters().len();
        assert_eq!(
            report.updated,
            2 * saved,
            "{saved} adapter(s) would be saved but only {} tensor(s) were updated",
            report.updated
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_fully_masked_batch_is_a_no_op() {
        let dir = tmpdir("masked");
        let Some(mut model) = tiny_model(&dir) else { return };
        let mut opt = AdamWOptimizer::new(0.0, 0.0);
        let batch = vec![TrainingExample {
            input_ids: vec![1, 2, 3, 4],
            labels: vec![IGNORE_LABEL; 4],
        }];
        let before = model.adapter(model::HEAD_ADAPTER).unwrap().b.clone();
        let report =
            training_step(&batch, &mut model, &mut opt, 0.1, &step_cfg(), &Device::Cpu).unwrap();

        assert_eq!(report.supervised, 0);
        assert_eq!(report.updated, 0);
        assert_eq!(report.loss, 0.0);
        let after = model.adapter(model::HEAD_ADAPTER).unwrap().b.clone();
        let delta = (&after - &before)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(delta, 0.0, "nothing supervised, nothing to learn");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_chunk_size_does_not_change_the_result() {
        // The monoid is what makes this true: chunk boundaries must not be
        // observable in the loss or the gradient norm.
        let dir = tmpdir("chunk");
        let Some(mut model) = tiny_model(&dir) else { return };
        let batch = vec![example()];

        let run = |model: &mut TrainableModel, chunk: usize| {
            let mut opt = AdamWOptimizer::new(0.0, 0.0);
            let cfg = StepConfig { logit_chunk: chunk, ..step_cfg() };
            training_step(&batch, model, &mut opt, 0.0, &cfg, &Device::Cpu).unwrap()
        };
        // lr = 0, so neither call perturbs the parameters the other sees.
        let one = run(&mut model, 1);
        let all = run(&mut model, 64);
        let odd = run(&mut model, 3);

        assert!((one.loss - all.loss).abs() < 1e-5, "{} vs {}", one.loss, all.loss);
        assert!((odd.loss - all.loss).abs() < 1e-5);
        assert!((one.grad_norm - all.grad_norm).abs() < 1e-4);
        assert!((odd.grad_norm - all.grad_norm).abs() < 1e-4);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn default_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.mode, TrainMode::Lora);
        assert_eq!(config.epochs, 3);
    }
}
