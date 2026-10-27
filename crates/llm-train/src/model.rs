//! Model loading and LoRA-augmented forward pass for training.
//!
//! This module provides:
//! - Loading Qwen3 base model weights
//! - LoRA adapter injection into attention layers
//! - Forward pass with gradient tracking for training

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};
use candle_transformers::models::qwen3::{Config as Qwen3Config, Model as Qwen3Model};

use crate::error::TrainError;
use crate::lora::{LoraAdapter, LoraConfig};

/// The one module this crate can actually adapt today.
///
/// LoRA belongs on the attention projections, but `candle-transformers`
/// decoder layers take no hook and expose no internals, so adapting them
/// would mean re-implementing Qwen3's block here. The output projection is
/// owned by this crate, so it can carry a real adapter that is in the forward
/// pass and receives a real gradient.
///
/// Before this, 112 adapters were created for `q/k/v/o_proj`, never applied,
/// never differentiated — and then written out as a "trained" checkpoint.
/// `TrainableModel` now holds exactly the adapters it can train, and
/// `trainer::tests` asserts that every saved adapter received a gradient.
pub const HEAD_ADAPTER: &str = "lm_head";

/// Configuration for model loading.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Path to model directory (config.json + safetensors).
    pub model_dir: PathBuf,
    /// Path to tokenizer.json.
    pub tokenizer_path: PathBuf,
    /// Use f16 (default) or f32 for weights.
    pub dtype: DType,
}

impl ModelConfig {
    pub fn new(model_dir: impl Into<PathBuf>, tokenizer_path: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            tokenizer_path: tokenizer_path.into(),
            dtype: DType::F32, // Use F32 for training (gradients)
        }
    }
}

/// LoRA-augmented Qwen3 model for training.
pub struct TrainableModel {
    /// Base Qwen3 body (frozen during LoRA training), *without* the causal-LM
    /// head.
    ///
    /// `ModelForCausalLM::forward` narrows to the last position before
    /// applying `lm_head` — correct for generation, useless for training: the
    /// trainer got logits of sequence length 1, hit its `seq_len <= 1` guard
    /// and returned a loss of exactly 0.0 for every step. Holding the body
    /// and the head separately gives logits at every position, which is what
    /// a next-token loss needs.
    base_model: Qwen3Model,
    /// The causal-LM head, applied to all positions.
    lm_head: Linear,
    /// The loaded weights, kept so the body can be rebuilt between steps.
    ///
    /// `Qwen3Model`'s KV cache concatenates on every forward and its
    /// `clear_kv_cache` is private to candle-transformers, so there is no way
    /// to reset it from here. Left alone, step 2 attends over step 1's keys —
    /// silently on CPU, where the causal mask is skipped. Rebuilding from the
    /// already-resident tensors is cheap: the weights are `Arc`-shared, so
    /// this clones handles, not data.
    tensors: HashMap<String, Tensor>,
    dtype: DType,
    /// LoRA adapters, keyed by the module they adapt. See [`HEAD_ADAPTER`].
    ///
    /// A map rather than one field: this is the seam a per-layer
    /// implementation would widen, and the save/load format already assumes
    /// several.
    lora_adapters: BTreeMap<String, LoraAdapter>,
    /// Model configuration.
    config: Qwen3Config,
    /// Device (CPU for now).
    device: Device,
    /// Tokenizer for decoding.
    tokenizer: tokenizers::Tokenizer,
}

impl TrainableModel {
    /// Load model weights and initialize LoRA adapters.
    pub fn load(
        model_config: &ModelConfig,
        lora_config: &LoraConfig,
    ) -> Result<Self, TrainError> {
        let device = Device::Cpu;

        // Load model config
        let config_path = model_config.model_dir.join("config.json");
        let raw = std::fs::read(&config_path).map_err(|e| TrainError::Io {
            path: config_path.clone(),
            source: e,
        })?;
        let qwen_config: Qwen3Config = serde_json::from_slice(&raw)
            .map_err(|e| TrainError::Model(format!("invalid config.json: {e}")))?;

        // Load safetensors shards
        let shards = glob_safetensors(&model_config.model_dir)?;
        if shards.is_empty() {
            return Err(TrainError::Model(format!(
                "no safetensors files in {}",
                model_config.model_dir.display()
            )));
        }

        tracing::info!(
            dir = %model_config.model_dir.display(),
            shards = shards.len(),
            layers = qwen_config.num_hidden_layers,
            "loading qwen3 weights for training"
        );

        let tensors = load_shards(&shards, &device)?;
        let dtype = model_config.dtype;
        let mut tensors_typed: HashMap<String, Tensor> = HashMap::new();
        for (k, t) in tensors {
            let t = t.to_dtype(dtype).map_err(|e| TrainError::Candle(e.to_string()))?;
            tensors_typed.insert(k, t);
        }

        let vb = VarBuilder::from_tensors(tensors_typed.clone(), dtype, &device);
        let base_model = Qwen3Model::new(&qwen_config, vb.clone())
            .map_err(|e| TrainError::Candle(e.to_string()))?;
        // Same construction `ModelForCausalLM::new` performs, kept here so the
        // head can be applied to the whole sequence.
        let lm_head = if qwen_config.tie_word_embeddings {
            let w = vb
                .pp("model.embed_tokens")
                .get((qwen_config.vocab_size, qwen_config.hidden_size), "weight")
                .map_err(|e| TrainError::Candle(e.to_string()))?;
            Linear::new(w, None)
        } else {
            linear_no_bias(
                qwen_config.hidden_size,
                qwen_config.vocab_size,
                vb.pp("lm_head"),
            )
            .map_err(|e| TrainError::Candle(e.to_string()))?
        };

        // One adapter, on the module that is actually in the forward pass.
        let head_adapter = LoraAdapter::new(
            qwen_config.hidden_size,
            qwen_config.vocab_size,
            lora_config,
            &device,
        )
        .map_err(|e| TrainError::Candle(e.to_string()))?;
        let lora_adapters = BTreeMap::from([(HEAD_ADAPTER.to_string(), head_adapter)]);

        tracing::info!(
            adapters = lora_adapters.len(),
            rank = lora_config.rank,
            trainable = qwen_config.hidden_size * lora_config.rank
                + lora_config.rank * qwen_config.vocab_size,
            "initialized LoRA adapters"
        );

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(&model_config.tokenizer_path)
            .map_err(|e| TrainError::Model(format!("failed to load tokenizer: {e}")))?;

        Ok(Self {
            base_model,
            lm_head,
            tensors: tensors_typed,
            dtype,
            lora_adapters,
            config: qwen_config,
            device,
            tokenizer,
        })
    }

    /// Drop the KV cache by rebuilding the body.
    ///
    /// Call once per training step. See the note on [`TrainableModel::tensors`].
    pub fn reset_cache(&mut self) -> Result<(), TrainError> {
        let vb = VarBuilder::from_tensors(self.tensors.clone(), self.dtype, &self.device);
        self.base_model = Qwen3Model::new(&self.config, vb)
            .map_err(|e| TrainError::Candle(e.to_string()))?;
        Ok(())
    }

    /// The frozen body: hidden states at every position, `[batch, seq, hidden]`.
    pub fn body(&mut self, input_ids: &Tensor) -> Result<Tensor, TrainError> {
        self.base_model
            .forward(input_ids, 0)
            .map_err(|e| TrainError::Candle(e.to_string()))
    }

    /// Frozen head plus adapter, on flattened hidden states `[n, hidden]`.
    ///
    /// This is the only place the two are combined, so training and
    /// inference cannot drift apart: [`TrainableModel::forward`] routes
    /// through it too.
    pub fn head_logits(&self, hidden_2d: &Tensor) -> Result<Tensor, TrainError> {
        let base = self
            .lm_head
            .forward(hidden_2d)
            .map_err(|e| TrainError::Candle(e.to_string()))?;
        match self.lora_adapters.get(HEAD_ADAPTER) {
            None => Ok(base),
            Some(adapter) => {
                let delta = adapter
                    .forward(hidden_2d)
                    .map_err(|e| TrainError::Candle(e.to_string()))?;
                (base + delta).map_err(|e| TrainError::Candle(e.to_string()))
            }
        }
    }

    /// Logits at every position, `[batch, seq, vocab]`.
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor, TrainError> {
        let hidden = self.body(input_ids)?;
        let (b, s, h) = hidden
            .dims3()
            .map_err(|e| TrainError::Candle(e.to_string()))?;
        let flat = hidden
            .reshape((b * s, h))
            .map_err(|e| TrainError::Candle(e.to_string()))?;
        let logits = self.head_logits(&flat)?;
        logits
            .reshape((b, s, self.config.vocab_size))
            .map_err(|e| TrainError::Candle(e.to_string()))
    }

    /// Hidden states for embedding generation (RAG) — the body output, which
    /// is what an embedding wants. This used to return logits.
    pub fn get_hidden_states(&mut self, input_ids: &Tensor) -> Result<Tensor, TrainError> {
        self.body(input_ids)
    }

    pub fn adapters(&self) -> &BTreeMap<String, LoraAdapter> {
        &self.lora_adapters
    }

    pub fn adapters_mut(&mut self) -> &mut BTreeMap<String, LoraAdapter> {
        &mut self.lora_adapters
    }

    pub fn adapter(&self, name: &str) -> Option<&LoraAdapter> {
        self.lora_adapters.get(name)
    }

    /// Save LoRA adapter weights to a directory.
    pub fn save_lora(&self, dir: &Path) -> Result<(), TrainError> {
        std::fs::create_dir_all(dir).map_err(|e| TrainError::Io {
            path: dir.to_path_buf(),
            source: e,
        })?;

        for (name, adapter) in &self.lora_adapters {
            let a_path = dir.join(format!("{name}_a.safetensors"));
            let b_path = dir.join(format!("{name}_b.safetensors"));

            let mut a_map: HashMap<String, Tensor> = HashMap::new();
            a_map.insert("a".to_string(), adapter.a.clone());
            candle_core::safetensors::save(&a_map, &a_path)
                .map_err(|e| TrainError::Candle(e.to_string()))?;

            let mut b_map: HashMap<String, Tensor> = HashMap::new();
            b_map.insert("b".to_string(), adapter.b.clone());
            candle_core::safetensors::save(&b_map, &b_path)
                .map_err(|e| TrainError::Candle(e.to_string()))?;
        }

        tracing::info!(dir = %dir.display(), "saved LoRA adapters");
        Ok(())
    }

    /// Load LoRA adapter weights from a directory.
    pub fn load_lora(&mut self, dir: &Path) -> Result<(), TrainError> {
        for (name, adapter) in &mut self.lora_adapters {
            let a_path = dir.join(format!("{name}_a.safetensors"));
            let b_path = dir.join(format!("{name}_b.safetensors"));

            if a_path.exists() && b_path.exists() {
                let a_data = std::fs::read(&a_path).map_err(|e| TrainError::Io {
                    path: a_path.clone(),
                    source: e,
                })?;
                let b_data = std::fs::read(&b_path).map_err(|e| TrainError::Io {
                    path: b_path.clone(),
                    source: e,
                })?;

                let a_map = candle_core::safetensors::load_buffer(&a_data, &self.device)
                    .map_err(|e| TrainError::Candle(e.to_string()))?;
                let b_map = candle_core::safetensors::load_buffer(&b_data, &self.device)
                    .map_err(|e| TrainError::Candle(e.to_string()))?;

                if let Some(a) = a_map.get("a") {
                    adapter.a = a.clone();
                }
                if let Some(b) = b_map.get("b") {
                    adapter.b = b.clone();
                }
            }
        }

        tracing::info!(dir = %dir.display(), "loaded LoRA adapters");
        Ok(())
    }

    /// Get the tokenizer reference.
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Get model config.
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    /// Vocab size for loss computation.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

/// Glob all .safetensors files in a directory.
fn glob_safetensors(dir: &Path) -> Result<Vec<PathBuf>, TrainError> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| TrainError::Io { path: dir.to_path_buf(), source: e })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| TrainError::Io { path: dir.to_path_buf(), source: e })?;
    let mut shards: Vec<PathBuf> = entries
        .into_iter()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    shards.sort();
    Ok(shards)
}

/// Load all safetensors shards and merge into a single tensor map.
fn load_shards(shards: &[PathBuf], device: &Device) -> Result<HashMap<String, Tensor>, TrainError> {
    let mut tensors = HashMap::new();
    for shard in shards {
        let data = std::fs::read(shard).map_err(|e| TrainError::Io {
            path: shard.clone(),
            source: e,
        })?;
        let map = candle_core::safetensors::load_buffer(&data, device)
            .map_err(|e| TrainError::Candle(e.to_string()))?;
        tensors.extend(map);
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_config_new() {
        let cfg = ModelConfig::new("/path/to/model", "/path/to/tokenizer.json");
        assert_eq!(cfg.model_dir.to_str().unwrap(), "/path/to/model");
        assert_eq!(cfg.dtype, DType::F32);
    }
}
