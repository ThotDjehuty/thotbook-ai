//! thotbook-ai M5 CLI — private OpenAI-compatible inference server for Qwen3-8B.

use std::path::PathBuf;

use clap::Parser;
use llm_serve::config::{ServeConfig, ServeOverrides};
use llm_serve::error::ServeError;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[derive(Debug, Parser)]
#[command(
    name = "llm-serve",
    about = "thotbook-ai M5: private OpenAI-compatible inference server (Qwen3-8B, candle, CPU)"
)]
struct Cli {
    /// Hugging Face model id (used for the download hint and /v1/models).
    #[arg(long)]
    model_id: Option<String>,

    /// Directory holding config.json + safetensors shards.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Path to the Qwen3 tokenizer file.
    #[arg(long)]
    tokenizer_path: Option<PathBuf>,

    /// Interface to bind.
    #[arg(long)]
    host: Option<String>,

    /// Port to bind.
    #[arg(long)]
    port: Option<u16>,

    /// Maximum number of tokens to generate per request.
    #[arg(long)]
    max_tokens: Option<usize>,

    /// Sampling temperature (0 = greedy).
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability.
    #[arg(long)]
    top_p: Option<f64>,
}

impl From<Cli> for ServeOverrides {
    fn from(cli: Cli) -> Self {
        Self {
            model_id: cli.model_id,
            model_dir: cli.model_dir,
            tokenizer_path: cli.tokenizer_path,
            host: cli.host,
            port: cli.port,
            max_tokens: cli.max_tokens,
            temperature: cli.temperature,
            top_p: cli.top_p,
        }
    }
}

fn main() -> Result<(), ServeError> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let mut config = ServeConfig::from_env()?;
    config.apply(&Cli::parse().into());
    llm_serve::run_server(config)
}
