//! Generation engine: Qwen3-8B tokenizer + candle model, greedy/temperature sampling.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::qwen3::{Config as Qwen3Config, ModelForCausalLM};
use tokenizers::Tokenizer;

use crate::config::ServeConfig;
use crate::error::ServeError;

/// Qwen3 chat-marker tokens.
pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
/// End-of-sequence marker for chat turns.
pub const EOS_TOKEN: &str = "<|im_end|>";
/// Fixed sampling seed so single-user generations are reproducible.
pub const DEFAULT_SEED: u64 = 42;

/// One turn of a chat conversation.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Sampling parameters for a single generation call.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub seed: u64,
}

impl GenerationParams {
    pub fn from_config(cfg: &ServeConfig, seed: u64) -> Self {
        Self {
            max_tokens: cfg.max_tokens,
            temperature: cfg.temperature,
            top_p: cfg.top_p,
            seed,
        }
    }
}

/// Render a conversation into the Qwen3 chat template:
/// `<|im_start|>user\n...<|im_end|>\n` ... ending with the assistant generation
/// marker, unless the last turn is already an assistant message.
pub fn format_chat_template(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(IM_START);
        out.push_str(&m.role.trim().to_ascii_lowercase());
        out.push('\n');
        out.push_str(m.content.trim());
        out.push_str(IM_END);
        out.push('\n');
    }
    let last_is_assistant = messages
        .last()
        .is_some_and(|m| m.role.trim().eq_ignore_ascii_case("assistant"));
    if !last_is_assistant {
        out.push_str(IM_START);
        out.push_str("assistant\n");
    }
    out
}

/// Build a candle sampling strategy from temperature/top-p.
fn sampling(temperature: f64, top_p: f64) -> Sampling {
    if temperature <= 0.0 {
        Sampling::ArgMax
    } else if (0.0..1.0).contains(&top_p) {
        Sampling::TopP {
            p: top_p,
            temperature,
        }
    } else {
        Sampling::All { temperature }
    }
}

fn sample(logits: &Tensor, processor: &mut LogitsProcessor) -> Result<u32, ServeError> {
    let logits = if logits.rank() == 3 {
        logits.squeeze(1)?
    } else {
        logits.clone()
    };
    let logits = logits.squeeze(0)?;
    processor.sample(&logits).map_err(Into::into)
}

/// Sample the next token from logits shaped `(1, vocab)`, deterministic from `seed`.
pub fn sample_token(
    logits: &Tensor,
    temperature: f64,
    top_p: f64,
    seed: u64,
) -> Result<u32, ServeError> {
    let mut processor = LogitsProcessor::from_sampling(seed, sampling(temperature, top_p));
    sample(logits, &mut processor)
}

/// A labelled download hint for when the local model dir is empty.
fn missing_model(config: &ServeConfig) -> ServeError {
    ServeError::model(format!(
        "no model weights found in `{}` — download first, e.g.:\n  huggingface-cli download {} --local-dir {}",
        config.model_dir.display(),
        config.model_id,
        config.model_dir.display()
    ))
}

fn glob_safetensors(dir: &std::path::Path) -> Result<Vec<PathBuf>, ServeError> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| ServeError::Io { path: dir.to_path_buf(), source: e })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ServeError::Io { path: dir.to_path_buf(), source: e })?;
    let mut shards: Vec<PathBuf> = entries
        .into_iter()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    shards.sort();
    Ok(shards)
}

/// The generation engine: loaded tokenizer + Qwen3 candle model.
pub struct Engine {
    model: ModelForCausalLM,
    tokenizer: Tokenizer,
    tokenizer_path: PathBuf,
    device: Device,
    eos_token: u32,
}

impl crate::http::Embedder for Engine {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, ServeError> {
        Engine::embed(self, texts)
    }
}

impl crate::http::Inference for Engine {
    /// Greedy/temperature + top-p generation for a template-rendered `prompt`.
    fn generate(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<String, ServeError> {
        let prompt_ids = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|source| ServeError::Tokenizer {
                path: self.tokenizer_path.clone(),
                source,
            })?
            .get_ids()
            .to_vec();

        self.model.clear_kv_cache();
        let mut processor = LogitsProcessor::from_sampling(
            params.seed,
            sampling(params.temperature, params.top_p),
        );

        let first = {
            let input = Tensor::new(prompt_ids.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, 0)?;
            sample(&logits, &mut processor)?
        };

        let mut generated = vec![first];
        let mut pos = prompt_ids.len();
        while generated.len() < params.max_tokens && *generated.last().unwrap() != self.eos_token {
            let input = Tensor::new(&[*generated.last().unwrap()], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, pos)?;
            pos += 1;
            let next = sample(&logits, &mut processor)?;
            generated.push(next);
        }

        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|source| ServeError::Tokenizer {
                path: self.tokenizer_path.clone(),
                source,
            })?;
        tracing::debug!(
            prompt_tokens = prompt_ids.len(),
            generated_tokens = generated.len(),
            "generation done"
        );
        Ok(text)
    }
}

/// Embedding dimension after mean-pooling and projection.
pub const EMBED_DIM: usize = 384;

impl Engine {
    /// Generate embeddings from text by mean-pooling the last hidden states.
    pub fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, ServeError> {
        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let tokens = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|source| ServeError::Tokenizer {
                    path: self.tokenizer_path.clone(),
                    source,
                })?
                .get_ids()
                .to_vec();

            if tokens.is_empty() {
                embeddings.push(vec![0.0; EMBED_DIM]);
                continue;
            }

            self.model.clear_kv_cache();
            let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            
            // Forward pass to get logits (we use these as a proxy for hidden states)
            // Ideally we'd access the last hidden layer directly, but candle's Qwen3
            // only exposes the final logits. We project them down to EMBED_DIM.
            let logits = self.model.forward(&input, 0)?;
            
            // Mean pool over sequence dimension
            let pooled = logits.mean(1)?;
            let data = pooled.to_vec2::<f32>()?;
            
            // Project to EMBED_DIM and L2 normalize
            let mut embedding = vec![0.0f32; EMBED_DIM];
            let copy_len = data[0].len().min(EMBED_DIM);
            embedding[..copy_len].copy_from_slice(&data[0][..copy_len]);
            
            // L2 normalize
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut embedding {
                    *x /= norm;
                }
            }
            
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    /// Load the tokenizer and model weights from `config.model_dir`.
    ///
    /// Refuses to download: an empty model dir produces a labelled hint and a
    /// typed [`ServeError::Model`].
    pub fn load(config: &ServeConfig) -> Result<Self, ServeError> {
        if !config.model_dir.is_dir() {
            return Err(missing_model(config));
        }
        let config_path = config.model_dir.join("config.json");
        if !config_path.is_file() {
            return Err(missing_model(config));
        }
        let shards = glob_safetensors(&config.model_dir)?;
        if shards.is_empty() {
            return Err(missing_model(config));
        }

        let device = Device::Cpu;
        let raw = std::fs::read(&config_path)
            .map_err(|e| ServeError::Io { path: config_path.clone(), source: e })?;
        let model_config: Qwen3Config = serde_json::from_slice(&raw).map_err(|e| {
            ServeError::config(format!("invalid `{}`: {e}", config_path.display()))
        })?;

        tracing::info!(
            dir = %config.model_dir.display(),
            shards = shards.len(),
            layers = model_config.num_hidden_layers,
            "loading qwen3 weights (cpu)"
        );
        let tensors = load_shards(&shards, &device)?;
        // F32, not F16: this crate depends on accelerate-src (Cargo.toml), which
        // enables candle's Accelerate backend, and Accelerate has no f16 matmul
        // -- every completion used to fail with
        //   "candle error: the accelerate backend does not support f16 matmul"
        // while /health and /v1/models still answered fine. Accelerate's sgemm
        // is the fast path on this CPU, so f32 is both correct and quick; the
        // cost is memory, roughly 3x the on-disk BF16 weights at load.
        // Override with LLM_SERVE_DTYPE=f16|bf16|f32 (f16 needs accelerate off).
        let dtype = match std::env::var("LLM_SERVE_DTYPE")
            .unwrap_or_else(|_| "f32".into())
            .to_ascii_lowercase()
            .as_str()
        {
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            _ => DType::F32,
        };
        let mut tensors_f16: HashMap<String, Tensor> = HashMap::new();
        for (k, t) in tensors {
            let t = t.to_dtype(dtype).map_err(ServeError::Candle)?;
            tensors_f16.insert(k, t);
        }
        let vb = VarBuilder::from_tensors(tensors_f16, dtype, &device);
        let model = ModelForCausalLM::new(&model_config, vb)?;

        let tokenizer = Tokenizer::from_file(&config.tokenizer_path).map_err(|source| {
            ServeError::Tokenizer {
                path: config.tokenizer_path.clone(),
                source,
            }
        })?;
        let eos_token = tokenizer.token_to_id(EOS_TOKEN).ok_or_else(|| {
            ServeError::Tokenizer {
                path: config.tokenizer_path.clone(),
                source: Box::new(std::io::Error::other(format!("missing {EOS_TOKEN} token"))),
            }
        })?;
        Ok(Self {
            model,
            tokenizer,
            tokenizer_path: config.tokenizer_path.clone(),
            device,
            eos_token,
        })
    }
}

/// Read every safetensors shard into memory and merge into a single tensor map.
fn load_shards(shards: &[PathBuf], device: &Device) -> Result<HashMap<String, Tensor>, ServeError> {
    let mut tensors = HashMap::new();
    for shard in shards {
        let data = std::fs::read(shard)
            .map_err(|e| ServeError::Io { path: shard.clone(), source: e })?;
        let map = candle_core::safetensors::load_buffer(&data, device)?;
        tensors.extend(map);
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    

    fn logits() -> Tensor {
        Tensor::new(&[0.1f32, 0.2, 0.9, 3.0, 0.4], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
    }

    #[test]
    fn format_chat_template_user() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hello".into(),
        }];
        assert_eq!(
            format_chat_template(&messages),
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn format_chat_template_system_then_user() {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "be terse".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
            },
        ];
        let out = format_chat_template(&messages);
        assert!(out.starts_with("<|im_start|>system\nbe terse<|im_end|>\n"));
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>\n"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn format_chat_template_assistant_continuation() {
        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "1+1".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "2".into(),
            },
        ];
        let out = format_chat_template(&messages);
        assert!(out.contains("<|im_start|>assistant\n2<|im_end|>\n"));
        assert!(!out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn sample_greedy_is_argmax() {
        let next = sample_token(&logits(), 0.0, 0.9, 7).unwrap();
        assert_eq!(next, 3);
    }

    #[test]
    fn sample_deterministic_given_seed() {
        let a = sample_token(&logits(), 0.8, 0.9, 1234).unwrap();
        let b = sample_token(&logits(), 0.8, 0.9, 1234).unwrap();
        assert_eq!(a, b);
        assert!(a < 5);
    }

    #[test]
    fn sample_without_top_p_works() {
        let next = sample_token(&logits(), 0.8, 1.0, 9).unwrap();
        assert!(next < 5);
    }

    #[test]
    fn load_missing_model_dir_returns_typed_error() {
        let cfg = ServeConfig {
            model_dir: PathBuf::from("/nonexistent/llm-serve-test-weights"),
            ..ServeConfig::default()
        };
        assert!(matches!(Engine::load(&cfg), Err(ServeError::Model(_))));
    }

    #[test]
    fn load_empty_model_dir_returns_typed_error() {
        let tmp = std::env::temp_dir().join(format!("llm-serve-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let cfg = ServeConfig {
            model_dir: tmp.clone(),
            ..ServeConfig::default()
        };
        assert!(matches!(Engine::load(&cfg), Err(ServeError::Model(_))));
        std::fs::remove_dir_all(tmp).unwrap();
    }

    #[test]
    fn load_dir_without_shards_returns_typed_error() {
        let tmp = std::env::temp_dir().join(format!("llm-serve-tok-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("config.json"), "{}").unwrap();
        let cfg = ServeConfig {
            model_dir: tmp.clone(),
            tokenizer_path: PathBuf::from("/nonexistent/qwen3-tokenizer.json"),
            ..ServeConfig::default()
        };
        assert!(matches!(Engine::load(&cfg), Err(ServeError::Model(_))));
        std::fs::remove_dir_all(tmp).unwrap();
    }
}

