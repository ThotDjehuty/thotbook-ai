use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "thotbook", about = "thotbook-ai — private LLM training & serving")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the inference server
    Serve {
        /// Model directory path
        #[arg(long, default_value = "data/models/qwen3-8b")]
        model_dir: PathBuf,
        /// Host to bind
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to bind
        #[arg(long, default_value_t = 8100)]
        port: u16,
    },

    /// Ingest documents into the corpus
    Ingest {
        /// Source directories to scan
        #[arg(required = true)]
        sources: Vec<PathBuf>,
        /// Output directory for chunks
        #[arg(long, default_value = "data/ingest")]
        output: PathBuf,
        /// Max tokens per chunk (whitespace-token count). Keep this well
        /// under the embedding model's real subword-token context: BERT-style
        /// tokenizers produce more tokens per word than whitespace splitting,
        /// especially on LaTeX and code.
        #[arg(long, default_value = "150")]
        max_tokens: usize,
        /// Overlap between consecutive chunks, in whitespace tokens.
        #[arg(long, default_value = "30")]
        overlap_tokens: usize,
    },

    /// Start continuous training
    Train {
        /// Training data path (JSONL of tokenised records)
        #[arg(long, required_unless_present = "owner_shards", conflicts_with = "owner_shards")]
        data: Option<PathBuf>,

        /// Train on one tenant's `polarway-learn` shards instead: a directory
        /// of prompt/completion JSONL written by `thotbook-learn`. Requires
        /// --tenant, and refuses the whole run if any sample in there belongs
        /// to somebody else.
        #[arg(long, requires = "tenant")]
        owner_shards: Option<PathBuf>,

        /// The tenant --owner-shards must belong to.
        #[arg(long)]
        tenant: Option<String>,
        /// Training mode: lora or full
        #[arg(long, default_value = "lora")]
        mode: String,
        /// Number of epochs
        #[arg(long, default_value_t = 3)]
        epochs: usize,
        /// Learning rate
        #[arg(long, default_value = "0.0002")]
        lr: f64,

        /// Base model directory. Defaults to the one in TrainingConfig,
        /// which is not necessarily the one you downloaded.
        #[arg(long)]
        model_dir: Option<PathBuf>,

        /// tokenizer.json to use. Defaults to <model-dir>/tokenizer.json when
        /// --model-dir is given.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
    },

    /// Build or query the RAG index
    Rag {
        #[command(subcommand)]
        action: RagAction,
    },

    /// List available agents
    Agent {
        /// Agent directory path
        #[arg(long, default_value = "sAlvIers")]
        dir: PathBuf,
    },

    /// Check system status
    Status,

    /// Autonomous research orchestration
    Auto {
        #[command(subcommand)]
        action: AutoAction,
    },

    /// Generate a research paper
    Paper {
        /// Research topic
        topic: String,
        /// Output directory
        #[arg(long, default_value = "output/papers")]
        output: PathBuf,
        /// Number of sections
        #[arg(long, default_value_t = 5)]
        sections: usize,
    },
}

#[derive(Subcommand)]
enum AutoAction {
    /// Run autonomous research on a topic
    Research {
        /// Research topic or question
        topic: String,
        /// Maximum iterations
        #[arg(long, default_value_t = 10)]
        max_iter: usize,
    },
    /// Extract training data from a local session-log corpus
    Extract {
        /// Session-log corpus directory
        #[arg(long, default_value = "../session-corpus")]
        session_corpus: PathBuf,
        /// Output JSONL path
        #[arg(long, default_value = "data/training/session_corpus.jsonl")]
        output: PathBuf,
    },
    /// Run self-improvement loop
    Improve {
        /// Training data directory
        #[arg(long, default_value = "data/training")]
        data_dir: PathBuf,
        /// Quality threshold (0-100)
        #[arg(long, default_value_t = 70)]
        threshold: u8,
    },
}

#[derive(Subcommand)]
enum RagAction {
    /// Build the RAG index from ingested documents
    Build {
        /// Directory containing ingested chunks
        #[arg(long, default_value = "data/ingest")]
        input: PathBuf,
    },
    /// Search the RAG index
    Search {
        /// Query text
        query: String,
        /// Number of results
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            model_dir,
            host,
            port,
        } => {
            tracing::info!(?model_dir, %host, %port, "starting inference server");
            // Start from the environment so LLM_SERVE_* is honoured (notably
            // MODEL_ID, which /v1/models reports), then let the explicit CLI
            // flags win. Building the struct literally used to pin model_id to
            // the "Qwen/Qwen3-8B" default whatever was actually loaded.
            let config = llm_serve::config::ServeConfig {
                model_dir,
                host,
                port,
                ..llm_serve::config::ServeConfig::from_env()?
            };
            // Already inside #[tokio::main]: await the async entrypoint rather
            // than run_server, which would nest a second runtime and panic.
            llm_serve::serve(config).await?;
        }

        Commands::Ingest { sources, output, max_tokens, overlap_tokens } => {
            tracing::info!(?sources, ?output, max_tokens, overlap_tokens, "starting ingestion");
            let config = llm_ingest::IngestConfig {
                sources,
                output: Some(output),
                chunk_config: llm_ingest::chunker::ChunkConfig { max_tokens, overlap_tokens },
            };
            let pipeline = llm_ingest::IngestPipeline::new(config);
            let report = pipeline.run()?;
            println!(
                "Ingestion complete: {} files, {} chunks, {} errors",
                report.files_processed, report.chunks_created, report.errors.len()
            );
            match &report.output_path {
                Some(p) => println!("Chunks written to {}", p.display()),
                None => println!("No output directory configured; nothing persisted."),
            }
        }

        Commands::Train {
            data,
            owner_shards,
            tenant,
            mode,
            epochs,
            lr,
            model_dir,
            tokenizer,
        } => {
            tracing::info!(?data, ?owner_shards, %mode, epochs, %lr, "starting training");
            let train_mode = match mode.as_str() {
                "full" => llm_train::trainer::TrainMode::Full,
                _ => llm_train::trainer::TrainMode::Lora,
            };
            let defaults = llm_train::TrainingConfig::default();
            let model_path = model_dir.unwrap_or(defaults.model_dir);
            let tokenizer_path = tokenizer
                .or_else(|| {
                    let beside = model_path.join("tokenizer.json");
                    beside.exists().then_some(beside)
                })
                .unwrap_or(defaults.tokenizer_path);
            let config = llm_train::TrainingConfig {
                mode: train_mode,
                epochs,
                learning_rate: lr as f32,
                model_dir: model_path,
                tokenizer_path,
                ..llm_train::TrainingConfig::default()
            };

            let mut dataset = match (&data, &owner_shards, &tenant) {
                (_, Some(dir), Some(t)) => {
                    let (ds, report) = llm_train::samples::load_owner_dataset(
                        dir,
                        t,
                        &config.tokenizer_path,
                        config.max_seq_len,
                        config.batch_size,
                    )?;
                    println!(
                        "owner shards: {} file(s), {} example(s) for {t} \
                         ({} question-only row(s) skipped, {} empty completion(s), {} truncated)",
                        report.shards,
                        report.samples,
                        report.signal_only,
                        report.empty_completions,
                        report.truncated
                    );
                    if report.samples == 0 {
                        return Err(format!(
                            "no training examples in {} for tenant {t}: {} question-only \
                             row(s) and nothing supervised. Run `make learn` so the \
                             reinforcement pass pairs your questions with the references \
                             the MCP holds.",
                            dir.display(),
                            report.signal_only
                        )
                        .into());
                    }
                    ds
                }
                (Some(path), _, _) => {
                    llm_train::dataset::StreamingDataset::from_jsonl(path, config.batch_size)?
                }
                _ => return Err("pass --data, or --owner-shards with --tenant".into()),
            };

            let mut trainer = llm_train::Trainer::new(config);
            // `train()` returns Model("model not loaded, call load_model()
            // first") without this, so the subcommand could never have run.
            trainer.load_model()?;
            trainer.train(&mut dataset)?;
            println!("Training complete");
        }

        Commands::Rag { action } => match action {
            RagAction::Build { input } => {
                tracing::info!(?input, "building RAG index");
                let chunks = input.join("chunks.jsonl");
                if !chunks.exists() {
                    return Err(format!(
                        "no chunks.jsonl in {} — run: thotbook ingest <sources> --output {}",
                        input.display(),
                        input.display()
                    )
                    .into());
                }
                let mut embedder = llm_rag::OllamaEmbedder::from_env()?;
                let mut store = llm_rag::RagStore::new(&input);
                let n = store.build_from_chunks(&chunks, &mut embedder)?;
                store.save()?;
                println!(
                    "RAG index built: {n} chunks embedded with {} (dim {})",
                    store.embed_model, store.embed_dim
                );
            }
            RagAction::Search { query, top_k } => {
                tracing::info!(%query, top_k, "searching RAG index");
                let mut store = llm_rag::RagStore::new(&std::path::PathBuf::from("data/ingest"));
                store.load()?;
                let mut embedder = llm_rag::OllamaEmbedder::from_env()?;
                let qv = embedder.embed(&query)?;
                let engine = &store.engine;
                let results = engine.search(&qv, &query, &llm_rag::SearchConfig {
                    top_k,
                    ..Default::default()
                });
                for (i, r) in results.iter().enumerate() {
                    println!("{}. [{}] (score: {:.3}) {}", i + 1, r.source, r.score, &r.text[..r.text.len().min(100)]);
                }
            }
        },

        Commands::Agent { dir } => {
            let registry = llm_salviers::AgentRegistry::load_from_dir(&dir)?;
            println!("Loaded {} agents:", registry.agents.len());
            for agent in &registry.agents {
                println!(
                    "  - {} ({})",
                    agent.name,
                    &agent.description[..agent.description.len().min(60)]
                );
            }
        }

        Commands::Status => {
            println!("thotbook-ai v0.3.0");
            println!("Crates: llm-corpus, llm-tokenize, llm-train, llm-rag, llm-serve, llm-cli, llm-ingest, llm-salviers, llm-auto");
            println!("Model: Qwen3-8B (candle, CPU)");
            println!("sAlvIers agents: sAlvIers/");
            println!("Autonomous: task decomposition, self-training, paper generation");
        }

        Commands::Auto { action } => match action {
            AutoAction::Research { topic, max_iter } => {
                tracing::info!(%topic, max_iter, "starting autonomous research");
                
                // Create orchestrator with default config
                let config = llm_auto::OrchestratorConfig::default();
                let mut orchestrator = llm_auto::Orchestrator::new(config);
                
                // Create initial research task
                let task = llm_auto::Task::new(
                    &topic,
                    llm_auto::TaskType::Research,
                    topic.clone(),
                );
                
                // Execute the task
                match orchestrator.execute(task).await {
                    Ok(result) => {
                        println!("\n=== Research Complete ===");
                        println!("Task ID: {}", result.id);
                        println!("Quality score: {:?}", result.quality_score);
                        if let Some(output) = &result.output {
                            println!("\nOutput:\n{}", output);
                        }
                    }
                    Err(e) => {
                        tracing::error!(?e, "research failed");
                        return Err(e.into());
                    }
                }
            }

            AutoAction::Extract { session_corpus, output } => {
                tracing::info!(?session_corpus, ?output, "extracting training data from session corpus");

                // Use TrainingOrchestrator to extract task/response pairs
                let orchestrator = llm_auto::training::TrainingOrchestrator::new(
                    session_corpus.clone(),
                    output.parent().unwrap_or(&PathBuf::from("data")).to_path_buf(),
                );

                // Get session-corpus stats first
                match orchestrator.session_corpus_stats() {
                    Ok(stats) => {
                        println!("Session-log corpus:");
                        println!("  Documents: {}", stats.documents);
                        println!("  Total bytes: {}", stats.total_bytes);

                        // Extract training examples
                        match orchestrator.extract_from_session_corpus() {
                            Ok(count) => {
                                println!("\nExtracted {} training examples to {:?}", count, output);
                            }
                            Err(e) => {
                                tracing::error!(?e, "extraction failed");
                                return Err(e.into());
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!(?e, "failed to get session-corpus stats");
                        return Err(e.into());
                    }
                }
            }

            AutoAction::Improve { data_dir, threshold } => {
                tracing::info!(?data_dir, threshold, "starting self-improvement loop");

                let orchestrator = llm_auto::training::TrainingOrchestrator::new(
                    data_dir.join("session-corpus"),
                    data_dir.clone(),
                );
                
                // Check current training data
                match orchestrator.count_examples() {
                    Ok(count) => {
                        println!("Current training examples: {}", count);
                        if count < 100 {
                            println!("Need at least 100 examples before retraining.");
                            println!("Run 'thotbook auto extract' first to gather training data.");
                        } else {
                            println!("Ready for training. Run 'thotbook train --data {:?}'", 
                                orchestrator.training_data_path());
                        }
                    }
                    Err(e) => {
                        tracing::error!(?e, "failed to count examples");
                        return Err(e.into());
                    }
                }
            }
        },

        Commands::Paper { topic, output, sections } => {
            tracing::info!(%topic, ?output, sections, "generating research paper");
            
            std::fs::create_dir_all(&output)?;
            
            let config = llm_auto::paper::PaperConfig {
                inference_url: "http://127.0.0.1:8100".to_string(),
                output_dir: output.clone(),
                max_sections: sections,
                ..Default::default()
            };
            let mut generator = llm_auto::paper::PaperGenerator::new(config);
            
            match generator.generate(&topic).await {
                Ok(paper) => {
                    // Write LaTeX output
                    let slug = topic.to_lowercase()
                        .chars()
                        .filter(|c| c.is_alphanumeric() || *c == ' ')
                        .collect::<String>()
                        .split_whitespace()
                        .take(3)
                        .collect::<Vec<_>>()
                        .join("_");
                    let tex_path = output.join(format!("{}.tex", slug));
                    std::fs::write(&tex_path, &paper.content)?;
                    
                    println!("\n=== Paper Generated ===");
                    println!("Title: {}", paper.title);
                    println!("Sections: {}", paper.sections.len());
                    println!("Output: {:?}", tex_path);
                    println!("\nTo compile: pdflatex {:?}", tex_path);
                }
                Err(e) => {
                    tracing::error!(?e, "paper generation failed");
                    return Err(e.into());
                }
            }
        }
    }

    Ok(())
}
