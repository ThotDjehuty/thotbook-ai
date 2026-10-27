//! thotbook-ai M1 CLI — corpus ingestion.
//!
//! ```sh
//! llm-corpus ingest \
//!   --historia  <historia.db> \
//!   --thotbook  <thotbook.db> \
//!   --lakehouse <lakehouse_dir> \
//!   [--tables sessions,corpus,equations,citations]
//! ```

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use llm_corpus::delta::{append_batch, corpus_batch, equations_batch, citations_batch, sessions_batch};
use llm_corpus::error::CorpusError;
use llm_corpus::schema;
use llm_corpus::sources;
use polarway_lakehouse::{DeltaStore, LakehouseConfig};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "llm-corpus", about = "thotbook-ai M1: ingest SQLite sources into Polarway Delta")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Ingest historia.db + thotbook.db into the lakehouse.
    Ingest {
        /// Path to historia.db (opencode session documents)
        #[arg(long)]
        historia: PathBuf,

        /// Path to thotbook.db (research index)
        #[arg(long)]
        thotbook: PathBuf,

        /// Root directory for Delta tables
        #[arg(long)]
        lakehouse: PathBuf,

        /// Comma-separated subset of: sessions,corpus,equations,citations (default: all)
        #[arg(long, default_value = "sessions,corpus,equations,citations")]
        tables: String,
    },
}

fn parse_tables(spec: &str) -> Result<Vec<&'static str>, CorpusError> {
    spec.split(',')
        .map(|t| match t.trim() {
            "sessions" => Ok(schema::TABLE_SESSIONS),
            "corpus" => Ok(schema::TABLE_CORPUS),
            "equations" => Ok(schema::TABLE_EQUATIONS),
            "citations" => Ok(schema::TABLE_CITATIONS),
            other => Err(CorpusError::UnknownTable(other.to_string())),
        })
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), CorpusError> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("llm_corpus=info".parse().unwrap()))
        .init();

    let cli = Cli::parse();
    let Command::Ingest {
        historia,
        thotbook,
        lakehouse,
        tables,
    } = cli.command;

    if !historia.exists() {
        return Err(CorpusError::MissingDb(historia));
    }
    if !thotbook.exists() {
        return Err(CorpusError::MissingDb(thotbook));
    }

    let table_names = parse_tables(&tables)?;
    let store = DeltaStore::new(LakehouseConfig::new(&lakehouse))
        .await
        .map_err(|e| CorpusError::Delta {
            table: "lakehouse".into(),
            source: e,
        })?;

    tracing::info!(historia = %historia.display(), thotbook = %thotbook.display(), lakehouse = %lakehouse.display(), "starting ingest");

    let want = |t: &'static str| table_names.contains(&t);

    if want(schema::TABLE_SESSIONS) {
        let docs = sources::read_sessions(&historia)?;
        let prepared = sessions_batch(&docs);
        let report = append_batch(&store, schema::TABLE_SESSIONS, schema::sessions_delta_fields(), prepared).await?;
        tracing::info!(rows = report.rows, version = report.version, "wrote sessions");
    }

    if want(schema::TABLE_CORPUS) {
        let (papers, notebooks, arxiv) = (
            sources::read_papers(&thotbook)?,
            sources::read_notebooks(&thotbook)?,
            sources::read_arxiv(&thotbook)?,
        );
        let prepared = corpus_batch(&papers, &notebooks, &arxiv);
        let report = append_batch(&store, schema::TABLE_CORPUS, schema::corpus_delta_fields(), prepared).await?;
        tracing::info!(rows = report.rows, version = report.version, "wrote corpus");
    }

    if want(schema::TABLE_EQUATIONS) {
        let eqs = sources::read_equations(&thotbook)?;
        let prepared = equations_batch(&eqs);
        let report = append_batch(&store, schema::TABLE_EQUATIONS, schema::equations_delta_fields(), prepared).await?;
        tracing::info!(rows = report.rows, version = report.version, "wrote equations");
    }

    if want(schema::TABLE_CITATIONS) {
        let arxiv = sources::read_arxiv(&thotbook)?;
        let prepared = citations_batch(&arxiv);
        let report = append_batch(&store, schema::TABLE_CITATIONS, schema::citations_delta_fields(), prepared).await?;
        tracing::info!(rows = report.rows, version = report.version, "wrote citations");
    }

    tracing::info!("ingest complete");
    Ok(())
}
