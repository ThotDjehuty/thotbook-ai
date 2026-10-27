//! thotbook-ai M2 CLI — tokenize lakehouse tables into sharded Arrow IPC files.
//!
//! ```sh
//! llm-tokenize tokenize \
//!   --lakehouse <lakehouse_dir> \
//!   --table datasets.sessions \
//!   [--content-col content] \
//!   [--tokenizer <tokenizer.json>] \
//!   [--out <out_dir>] \
//!   [--tokens-per-shard 1000000]
//! ```

use std::path::{Path, PathBuf};

use arrow::array::RecordBatch;
use clap::{Parser, Subcommand};
use llm_tokenize::error::TokenizeError;
use llm_tokenize::tokenizer::load_tokenizer;
use llm_tokenize::{tokenize_table_with, TokenizeReport};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "llm-tokenize", about = "thotbook-ai M2: tokenize lakehouse corpus into sharded training-ready datasets")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Tokenize a lakehouse table into sharded Arrow IPC token files.
    Tokenize {
        /// Root directory of the Delta lakehouse.
        #[arg(long)]
        lakehouse: PathBuf,

        /// Table to tokenize.
        #[arg(long, default_value = "datasets.sessions")]
        table: String,

        /// Content column (auto-detected: sessions/corpus -> content, equations -> latex).
        #[arg(long)]
        content_col: Option<String>,

        /// Path to tokenizer.json (downloaded from HuggingFace on first use if missing).
        #[arg(long, default_value = llm_tokenize::tokenizer::DEFAULT_TOKENIZER_PATH)]
        tokenizer: PathBuf,

        /// Output directory (default: data/tokenized/<table>).
        #[arg(long)]
        out: Option<PathBuf>,

        /// Tokens per shard.
        #[arg(long, default_value_t = 1_000_000)]
        tokens_per_shard: usize,
    },
}

fn default_content_col(table: &str) -> &'static str {
    match table {
        "datasets.equations" => "latex",
        _ => "content",
    }
}

async fn read_batches(lakehouse: &Path, table: &str) -> Result<Vec<RecordBatch>, TokenizeError> {
    let path = lakehouse.join(table);
    let url = url::Url::from_directory_path(&path).map_err(|_| TokenizeError::Delta {
        table: table.to_string(),
        path: path.clone(),
        detail: "not a valid table directory path".to_string(),
    })?;
    let delta = deltalake::open_table(url).await.map_err(|e| TokenizeError::Delta {
        table: table.to_string(),
        path: path.clone(),
        detail: e.to_string(),
    })?;
    let (_delta, stream) = delta.scan_table().await.map_err(|e| TokenizeError::Delta {
        table: table.to_string(),
        path: path.clone(),
        detail: e.to_string(),
    })?;
    deltalake::datafusion::physical_plan::common::collect(stream).await.map_err(|e| {
        TokenizeError::Delta {
            table: table.to_string(),
            path: path.clone(),
            detail: e.to_string(),
        }
    })
}

fn print_report(table: &str, rows: usize, content_col: &str, tokenizer: &Path, out: &Path, tokens_per_shard: usize, report: &TokenizeReport) {
    println!("=== llm-tokenize report ({table}) ===");
    println!("mode           : RÉEL (Delta lakehouse data present)");
    println!("rows           : {rows}");
    println!("content column : {content_col}");
    println!("total tokens   : {}", report.total_tokens);
    println!("shards         : {}", report.shards);
    println!("shard bytes    : {}", report.bytes);
    println!("tokens/shard   : {tokens_per_shard}");
    println!("tokenizer      : {}", tokenizer.display());
    println!("out dir        : {}", out.display());
}

#[tokio::main]
async fn main() -> Result<(), TokenizeError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("llm_tokenize=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();
    let Command::Tokenize {
        lakehouse,
        table,
        content_col,
        tokenizer,
        out,
        tokens_per_shard,
    } = cli.command;

    let content_col = content_col.unwrap_or_else(|| default_content_col(&table).to_string());
    let out = out.unwrap_or_else(|| PathBuf::from("data/tokenized").join(&table));

    let batches = read_batches(&lakehouse, &table).await?;
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    tracing::info!(table = %table, rows, "read lakehouse table");

    let tok = load_tokenizer(&tokenizer).await?;
    tracing::info!(tokenizer = %tokenizer.display(), "tokenizer ready");

    let schema = batches
        .first()
        .ok_or_else(|| TokenizeError::Delta {
            table: table.clone(),
            path: lakehouse.join(&table),
            detail: "table contains no rows".to_string(),
        })?
        .schema();
    let combined = arrow::compute::concat_batches(&schema, &batches)?;

    let report = tokenize_table_with(
        &combined,
        &content_col,
        &tok,
        &out,
        &table,
        &tokenizer,
        tokens_per_shard,
    )?;

    print_report(&table, rows, &content_col, &tokenizer, &out, tokens_per_shard, &report);
    Ok(())
}
