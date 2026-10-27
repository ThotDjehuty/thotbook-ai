//! Task representation and decomposition.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A task to be executed by the autonomous system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// Task type determining which agent handles it.
    pub task_type: TaskType,
    /// Input context for the task.
    pub input: String,
    /// Expected output format.
    pub output_format: OutputFormat,
    /// Parent task ID (for subtasks).
    pub parent_id: Option<String>,
    /// Priority (higher = more important).
    pub priority: u8,
    /// Current status.
    pub status: TaskStatus,
    /// Output after completion.
    pub output: Option<String>,
    /// Quality score from self-evaluation (0-100).
    pub quality_score: Option<u8>,
}

impl Task {
    pub fn new(description: impl Into<String>, task_type: TaskType, input: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            description: description.into(),
            task_type,
            input: input.into(),
            output_format: OutputFormat::Text,
            parent_id: None,
            priority: 50,
            status: TaskStatus::Pending,
            output: None,
            quality_score: None,
        }
    }

    pub fn with_parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_id = Some(parent_id.into());
        self
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }
}

/// Type of task determining which agent handles it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    /// Literature review and arxiv search.
    Research,
    /// Mathematical derivations and proofs.
    Mathematics,
    /// Code implementation and testing.
    Coding,
    /// Writing and LaTeX formatting.
    Writing,
    /// Quality evaluation and feedback.
    Critique,
    /// Continuous learning orchestration.
    Training,
    /// Task decomposition.
    Decomposition,
    /// Research paper generation.
    PaperGeneration,
}

/// Expected output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Plain text.
    Text,
    /// Markdown.
    Markdown,
    /// LaTeX.
    LaTeX,
    /// JSON.
    Json,
    /// Python code.
    Python,
    /// Rust code.
    Rust,
}

/// Task execution status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Not started.
    Pending,
    /// Currently executing.
    Running,
    /// Completed successfully.
    Completed,
    /// Failed with error.
    Failed,
    /// Cancelled.
    Cancelled,
}

/// Result of task decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecomposedTask {
    /// Original task.
    pub original: Task,
    /// Subtasks to execute.
    pub subtasks: Vec<Task>,
    /// Execution order (task IDs).
    pub execution_order: Vec<String>,
    /// Dependencies (task_id -> depends_on task_ids).
    pub dependencies: std::collections::HashMap<String, Vec<String>>,
}

impl DecomposedTask {
    pub fn new(original: Task) -> Self {
        Self {
            original,
            subtasks: Vec::new(),
            execution_order: Vec::new(),
            dependencies: std::collections::HashMap::new(),
        }
    }

    pub fn add_subtask(&mut self, subtask: Task) {
        self.execution_order.push(subtask.id.clone());
        self.subtasks.push(subtask);
    }

    pub fn add_dependency(&mut self, task_id: impl Into<String>, depends_on: impl Into<String>) {
        self.dependencies
            .entry(task_id.into())
            .or_default()
            .push(depends_on.into());
    }
}
