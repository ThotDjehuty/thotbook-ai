//! `llm-train` — LoRA/QLoRA + full fine-tuning pipeline for thotbook-ai.
//!
//! Two-tier training:
//! - **LoRA on Qwen3-8B**: fast adapter training (~0.1% trainable params)
//! - **Full fine-tune on Qwen3-4B**: complete weight updates on smaller model
//!
//! Streaming dataset from Delta tables or Arrow IPC shards, never loads all data into RAM.

pub mod arrow_dataset;
pub mod checkpoint;
pub mod dataset;
pub mod error;
pub mod grads;
pub mod lora;
pub mod loss;
pub mod metrics;
pub mod model;
pub mod optimizer;
pub mod samples;
pub mod scheduler;
pub mod trainer;

pub use error::TrainError;
pub use model::{ModelConfig, TrainableModel};
pub use trainer::{TrainMode, TrainingConfig, Trainer};
