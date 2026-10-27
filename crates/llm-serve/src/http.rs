//! OpenAI-compatible HTTP surface (axum 0.7) for the private inference server.

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::config::ServeConfig;
use crate::engine::{format_chat_template, ChatMessage, DEFAULT_SEED, GenerationParams};
use crate::error::ServeError;

/// Backend capable of generating text from a template-rendered prompt.
pub trait Inference {
    fn generate(&mut self, prompt: &str, params: &GenerationParams) -> Result<String, ServeError>;
}

/// Backend capable of generating embeddings.
pub trait Embedder {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, ServeError>;
}

/// Shared application state.
pub struct AppState<E: Inference> {
    engine: Arc<Mutex<E>>,
    config: ServeConfig,
}

impl<E: Inference> Clone for AppState<E> {
    fn clone(&self) -> Self {
        Self {
            engine: Arc::clone(&self.engine),
            config: self.config.clone(),
        }
    }
}

impl<E: Inference> AppState<E> {
    pub fn new(engine: E, config: ServeConfig) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
            config,
        }
    }
}

/// Build the axum router for the OpenAI-compatible surface.
pub fn router<E: Inference + Embedder + Send + Sync + 'static>(state: AppState<E>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions::<E>))
        .route("/v1/embeddings", post(embeddings::<E>))
        .with_state(state)
}

/// OpenAI-compatible chat completion request.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: ResponseMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ResponseMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}

/// OpenAI-compatible embeddings request.
#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub input: EmbeddingInput,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

impl EmbeddingInput {
    pub fn to_vec(self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s],
            EmbeddingInput::Multiple(v) => v,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: &'static str,
    pub param: Option<String>,
    pub code: &'static str,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn list_models<E: Inference + Embedder + Send + Sync>(
    State(state): State<AppState<E>>,
) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelInfo {
            id: state.config.model_id.clone(),
            object: "model",
            owned_by: "thotbook",
        }],
    })
}

async fn embeddings<E: Inference + Embedder + Send + Sync + 'static>(
    State(state): State<AppState<E>>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, ServeError> {
    let texts = req.input.to_vec();
    if texts.is_empty() {
        return Err(ServeError::http("`input` must not be empty"));
    }

    let prompt_tokens: usize = texts.iter().map(|t| t.split_whitespace().count()).sum();
    let engine = Arc::clone(&state.engine);

    let embeddings = tokio::task::spawn_blocking(move || {
        let mut engine = engine.lock().map_err(|_| ServeError::http("engine lock poisoned"))?;
        engine.embed(&texts)
    })
    .await
    .map_err(|e| ServeError::http(format!("embedding task failed: {e}")))??;

    let data: Vec<EmbeddingData> = embeddings
        .into_iter()
        .enumerate()
        .map(|(i, embedding)| EmbeddingData {
            object: "embedding",
            embedding,
            index: i,
        })
        .collect();

    Ok(Json(EmbeddingsResponse {
        object: "list",
        data,
        model: state.config.model_id.clone(),
        usage: EmbeddingUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    }))
}

async fn chat_completions<E: Inference + Send + Sync + 'static>(
    State(state): State<AppState<E>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatCompletion>, ServeError> {
    if req.messages.is_empty() {
        return Err(ServeError::http("`messages` must contain at least one message"));
    }
    let params = GenerationParams {
        max_tokens: req.max_tokens.unwrap_or(state.config.max_tokens),
        temperature: req.temperature.unwrap_or(state.config.temperature),
        top_p: req.top_p.unwrap_or(state.config.top_p),
        seed: DEFAULT_SEED,
    };
    let prompt = format_chat_template(&req.messages);

    let engine = Arc::clone(&state.engine);
    let content = tokio::task::spawn_blocking(move || {
        let mut engine = engine.lock().map_err(|_| ServeError::http("engine lock poisoned"))?;
        engine.generate(&prompt, &params)
    })
    .await
    .map_err(|e| ServeError::http(format!("generation task failed: {e}")))??;

    Ok(Json(ChatCompletion {
        id: format!("chatcmpl-{:016x}", unix_nanos()),
        object: "chat.completion",
        created: unix_now(),
        model: state.config.model_id.clone(),
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant",
                content,
            },
            finish_reason: "stop",
        }],
    }))
}

impl IntoResponse for ServeError {
    fn into_response(self) -> Response {
        let status = match &self {
            ServeError::Http(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(ErrorBody {
            error: ErrorDetail {
                message: self.to_string(),
                r#type: "error",
                param: None,
                code: "error",
            },
        }))
        .into_response()
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn unix_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;

    #[derive(Clone, Default)]
    struct FakeEngine {
        prompts: Arc<Mutex<Vec<String>>>,
    }

    impl Inference for FakeEngine {
        fn generate(
            &mut self,
            prompt: &str,
            _params: &GenerationParams,
        ) -> Result<String, ServeError> {
            self.prompts.lock().unwrap().push(prompt.to_string());
            Ok("42".to_string())
        }
    }

    impl Embedder for FakeEngine {
        fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, ServeError> {
            Ok(texts.iter().map(|_| vec![0.0; 384]).collect())
        }
    }

    #[derive(Clone, Default)]
    struct FailingEngine;

    impl Inference for FailingEngine {
        fn generate(
            &mut self,
            _prompt: &str,
            _params: &GenerationParams,
        ) -> Result<String, ServeError> {
            Err(ServeError::model("boom"))
        }
    }

    impl Embedder for FailingEngine {
        fn embed(&mut self, _texts: &[String]) -> Result<Vec<Vec<f32>>, ServeError> {
            Err(ServeError::model("boom"))
        }
    }

    async fn spawn_server<E: Inference + Embedder + Send + Sync + 'static>(
        engine: E,
    ) -> String {
        let state = AppState::new(engine, ServeConfig::default());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn health_responds() {
        let base = spawn_server(FakeEngine::default()).await;
        let body: serde_json::Value = reqwest::get(format!("{base}/health"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(body["status"], "ok");
    }

    #[tokio::test]
    async fn list_models_responds() {
        let base = spawn_server(FakeEngine::default()).await;
        let body: serde_json::Value = reqwest::get(format!("{base}/v1/models"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(body["object"], "list");
        assert_eq!(body["data"][0]["id"], "Qwen/Qwen3-8B");
        assert_eq!(body["data"][0]["object"], "model");
    }

    #[tokio::test]
    async fn chat_completions_responds() {
        let engine = FakeEngine::default();
        let base = spawn_server(engine.clone()).await;

        let client = reqwest::Client::new();
        let body: serde_json::Value = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "Qwen/Qwen3-8B",
                "messages": [
                    {"role": "user", "content": "what is 6*7?"}
                ],
                "max_tokens": 16,
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["model"], "Qwen/Qwen3-8B");
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(body["choices"][0]["message"]["content"], "42");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");

        let prompts = engine.prompts.lock().unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(
            prompts[0],
            "<|im_start|>user\nwhat is 6*7?<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[tokio::test]
    async fn chat_completions_empty_messages_is_bad_request() {
        let base = spawn_server(FakeEngine::default()).await;
        let resp = reqwest::Client::new()
            .post(format!("{base}/v1/chat/completions"))
            .json(&serde_json::json!({ "messages": [] }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn chat_completions_generation_error_is_500() {
        let base = spawn_server(FailingEngine).await;
        let resp = reqwest::Client::new()
            .post(format!("{base}/v1/chat/completions"))
            .json(&serde_json::json!({
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn engine_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Engine>();
    }
}
