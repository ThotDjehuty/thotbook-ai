//! `llm-auto` — Autonomous orchestration for thotbook-ai.
//!
//! This crate implements:
//! - Task decomposition into subtasks
//! - Sub-agent spawning and coordination
//! - Self-improvement through training on outputs
//! - Research paper generation pipeline
//!
//! The orchestrator runs without human intervention, using only local
//! thotbook-ai weights for inference.

pub mod agent;
pub mod error;
pub mod orchestrator;
pub mod paper;
pub mod task;
pub mod training;

pub use error::AutoError;
pub use orchestrator::{Orchestrator, OrchestratorConfig};
pub use task::{Task, TaskType};
