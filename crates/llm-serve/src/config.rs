//! Server configuration: env-driven with sensible defaults, CLI overridable.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::Deserialize;

use crate::error::ServeError;

pub const DEFAULT_MODEL_ID: &str = "Qwen/Qwen3-8B";
pub const DEFAULT_MODEL_DIR: &str = "data/models/qwen3-8b";
pub const DEFAULT_TOKENIZER_PATH: &str = "data/tokenizers/qwen3-tokenizer.json";
pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 8100;
pub const DEFAULT_MAX_TOKENS: usize = 512;
pub const DEFAULT_TEMPERATURE: f64 = 0.7;
pub const DEFAULT_TOP_P: f64 = 0.9;

/// Env var prefix consumed by [`ServeConfig::from_env`].
pub const ENV_PREFIX: &str = "LLM_SERVE_";

/// Configuration for the private inference server.
#[derive(Debug, Clone, Deserialize)]
pub struct ServeConfig {
    /// Hugging Face model id, used for the download hint and `/v1/models`.
    pub model_id: String,
    /// Directory holding `config.json` + `*.safetensors` shards.
    pub model_dir: PathBuf,
    /// Path to the Qwen3 tokenizer file.
    pub tokenizer_path: PathBuf,
    /// Interface to bind.
    pub host: String,
    /// Port to bind.
    pub port: u16,
    /// Default maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Default sampling temperature (0 = greedy).
    pub temperature: f64,
    /// Default nucleus sampling probability.
    pub top_p: f64,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            model_id: DEFAULT_MODEL_ID.to_string(),
            model_dir: PathBuf::from(DEFAULT_MODEL_DIR),
            tokenizer_path: PathBuf::from(DEFAULT_TOKENIZER_PATH),
            host: DEFAULT_HOST.to_string(),
            port: DEFAULT_PORT,
            max_tokens: DEFAULT_MAX_TOKENS,
            temperature: DEFAULT_TEMPERATURE,
            top_p: DEFAULT_TOP_P,
        }
    }
}

/// Optional CLI overrides; `None` keeps the current value.
#[derive(Debug, Default, Clone)]
pub struct ServeOverrides {
    pub model_id: Option<String>,
    pub model_dir: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
}

fn env_str(vars: &HashMap<String, String>, name: &str) -> Option<String> {
    vars.get(name).cloned()
}

fn env_num<T: std::str::FromStr>(
    vars: &HashMap<String, String>,
    name: &str,
) -> Result<Option<T>, ServeError> {
    match vars.get(name) {
        Some(raw) => raw
            .parse::<T>()
            .map(Some)
            .map_err(|_| ServeError::config(format!("invalid `{name}`: `{raw}`"))),
        None => Ok(None),
    }
}

impl ServeConfig {
    /// Build from `LLM_SERVE_*` env vars, falling back to defaults.
    pub fn from_env() -> Result<Self, ServeError> {
        let vars: HashMap<String, String> = std::env::vars()
            .filter(|(k, _)| k.starts_with(ENV_PREFIX))
            .collect();
        Self::from_map(&vars)
    }

    /// Build from a raw map of env vars (testable without touching the process env).
    pub fn from_map(vars: &HashMap<String, String>) -> Result<Self, ServeError> {
        let d = Self::default();
        let name = |field: &str| format!("{ENV_PREFIX}{field}");
        Ok(Self {
            model_id: env_str(vars, &name("MODEL_ID")).unwrap_or(d.model_id),
            model_dir: env_str(vars, &name("MODEL_DIR"))
                .map(PathBuf::from)
                .unwrap_or(d.model_dir),
            tokenizer_path: env_str(vars, &name("TOKENIZER_PATH"))
                .map(PathBuf::from)
                .unwrap_or(d.tokenizer_path),
            host: env_str(vars, &name("HOST")).unwrap_or(d.host),
            port: env_num(vars, &name("PORT"))?.unwrap_or(d.port),
            max_tokens: env_num(vars, &name("MAX_TOKENS"))?.unwrap_or(d.max_tokens),
            temperature: env_num(vars, &name("TEMPERATURE"))?.unwrap_or(d.temperature),
            top_p: env_num(vars, &name("TOP_P"))?.unwrap_or(d.top_p),
        })
    }

    /// Overlay CLI-provided values on top of the current config.
    pub fn apply(&mut self, o: &ServeOverrides) {
        if let Some(v) = &o.model_id {
            self.model_id = v.clone();
        }
        if let Some(v) = &o.model_dir {
            self.model_dir = v.clone();
        }
        if let Some(v) = &o.tokenizer_path {
            self.tokenizer_path = v.clone();
        }
        if let Some(v) = &o.host {
            self.host = v.clone();
        }
        if let Some(v) = o.port {
            self.port = v;
        }
        if let Some(v) = o.max_tokens {
            self.max_tokens = v;
        }
        if let Some(v) = o.temperature {
            self.temperature = v;
        }
        if let Some(v) = o.top_p {
            self.top_p = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (format!("{ENV_PREFIX}{k}"), v.to_string()))
            .collect()
    }

    #[test]
    fn defaults_are_sensible() {
        let cfg = ServeConfig::default();
        assert_eq!(cfg.model_id, "Qwen/Qwen3-8B");
        assert_eq!(cfg.port, 8100);
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.max_tokens, 512);
        assert_eq!(cfg.temperature, 0.7);
        assert_eq!(cfg.top_p, 0.9);
        assert!(cfg.model_dir.ends_with("data/models/qwen3-8b"));
        assert!(cfg.tokenizer_path.ends_with("data/tokenizers/qwen3-tokenizer.json"));
    }

    #[test]
    fn from_map_overrides_all_fields() {
        let vars = map(&[
            ("MODEL_ID", "org/x"),
            ("MODEL_DIR", "/tmp/m"),
            ("TOKENIZER_PATH", "/tmp/tok.json"),
            ("HOST", "0.0.0.0"),
            ("PORT", "9999"),
            ("MAX_TOKENS", "128"),
            ("TEMPERATURE", "1.5"),
            ("TOP_P", "0.95"),
        ]);
        let cfg = ServeConfig::from_map(&vars).unwrap();
        assert_eq!(cfg.model_id, "org/x");
        assert_eq!(cfg.model_dir, PathBuf::from("/tmp/m"));
        assert_eq!(cfg.tokenizer_path, PathBuf::from("/tmp/tok.json"));
        assert_eq!(cfg.host, "0.0.0.0");
        assert_eq!(cfg.port, 9999);
        assert_eq!(cfg.max_tokens, 128);
        assert_eq!(cfg.temperature, 1.5);
        assert_eq!(cfg.top_p, 0.95);
    }

    #[test]
    fn from_map_partial_keeps_defaults() {
        let cfg = ServeConfig::from_map(&map(&[("PORT", "9000")])).unwrap();
        assert_eq!(cfg.port, 9000);
        assert_eq!(cfg.max_tokens, ServeConfig::default().max_tokens);
        assert_eq!(cfg.temperature, ServeConfig::default().temperature);
    }

    #[test]
    fn from_map_invalid_number_errors() {
        let err = ServeConfig::from_map(&map(&[("PORT", "not-a-port")]))
            .unwrap_err();
        assert!(matches!(err, ServeError::Config(_)));
        let err = ServeConfig::from_map(&map(&[("TEMPERATURE", "hot")])).unwrap_err();
        assert!(matches!(err, ServeError::Config(_)));
    }

    #[test]
    fn apply_overlays_cli_values() {
        let mut cfg = ServeConfig::default();
        cfg.apply(&ServeOverrides {
            port: Some(1234),
            temperature: Some(0.0),
            ..Default::default()
        });
        assert_eq!(cfg.port, 1234);
        assert_eq!(cfg.temperature, 0.0);
        assert_eq!(cfg.top_p, ServeConfig::default().top_p);
    }
}
