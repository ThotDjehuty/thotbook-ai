//! thotbook-ai M5 — private OpenAI-compatible inference server (Qwen3-8B, candle, CPU).
//! Enhanced with RAG context retrieval and agent dispatch.

pub mod agent;
pub mod config;
pub mod engine;
pub mod error;
pub mod http;
pub mod rag;

use config::ServeConfig;
use error::ServeError;

/// Load the engine, bind the listener, and serve until the process stops.
///
/// Call this from an existing runtime. `run_server` is the sync wrapper for
/// callers that have none.
pub async fn serve(config: ServeConfig) -> Result<(), ServeError> {
    let engine = engine::Engine::load(&config)?;
    let state = http::AppState::new(engine, config.clone());

    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| ServeError::http(format!("failed to bind {addr}: {e}")))?;

    // Logged only once the socket is actually accepting: the previous order
    // announced "listening" before the bind, so a failure right after looked
    // like a healthy server that refused every connection.
    tracing::info!(%addr, model = %config.model_id, "llm-serve listening (private, cpu)");

    axum::serve(listener, http::router(state))
        .await
        .map_err(|e| ServeError::Http(e.to_string()))
}

/// Blocking entrypoint: builds a runtime and drives [`serve`].
///
/// Do not call this from inside a runtime - `block_on` panics with "Cannot
/// start a runtime from within a runtime". `llm-cli` is `#[tokio::main]`, so
/// it awaits [`serve`] directly.
pub fn run_server(config: ServeConfig) -> Result<(), ServeError> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| ServeError::http(format!("failed to build tokio runtime: {e}")))?;

    runtime.block_on(serve(config))
}
