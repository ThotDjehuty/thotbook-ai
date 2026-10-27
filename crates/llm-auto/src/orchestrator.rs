//! Main orchestrator for autonomous task execution.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::agent::AgentRegistry;
use crate::error::AutoError;
use crate::task::{DecomposedTask, Task, TaskStatus, TaskType};
use crate::training::TrainingOrchestrator;

/// Configuration for the orchestrator.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// URL of the thotbook-ai inference server.
    pub inference_url: String,
    /// Path to the session-log corpus directory for training data.
    pub session_corpus_dir: PathBuf,
    /// Path to lakehouse for training data.
    pub lakehouse_dir: PathBuf,
    /// Quality threshold for adding outputs to training (0-100).
    pub quality_threshold: u8,
    /// Enable automatic retraining.
    pub auto_retrain: bool,
    /// Minimum samples before retraining.
    pub retrain_min_samples: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            inference_url: "http://127.0.0.1:8100".to_string(),
            session_corpus_dir: PathBuf::from("../session-corpus"),
            lakehouse_dir: PathBuf::from("data/lakehouse"),
            quality_threshold: 70,
            auto_retrain: true,
            retrain_min_samples: 100,
        }
    }
}

/// The main orchestrator for autonomous task execution.
pub struct Orchestrator {
    config: OrchestratorConfig,
    agents: AgentRegistry,
    training: TrainingOrchestrator,
    /// Completed tasks with their outputs.
    completed_tasks: HashMap<String, Task>,
    /// Pending high-quality outputs for training.
    training_queue: Vec<TrainingExample>,
}

/// A training example extracted from task outputs.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub input: String,
    pub output: String,
    pub quality_score: u8,
    pub task_type: TaskType,
}

impl Orchestrator {
    pub fn new(config: OrchestratorConfig) -> Self {
        let agents = AgentRegistry::new(&config.inference_url);
        let training = TrainingOrchestrator::new(
            config.session_corpus_dir.clone(),
            config.lakehouse_dir.clone(),
        );

        Self {
            config,
            agents,
            training,
            completed_tasks: HashMap::new(),
            training_queue: Vec::new(),
        }
    }

    /// Execute a task autonomously.
    pub async fn execute(&mut self, mut task: Task) -> Result<Task, AutoError> {
        tracing::info!(task_id = %task.id, description = %task.description, "starting task");

        // Decompose complex tasks
        let decomposed = self.decompose_task(&task).await?;

        if decomposed.subtasks.is_empty() {
            // Simple task - execute directly
            task = self.execute_simple_task(task).await?;
        } else {
            // Complex task - execute subtasks in order
            task = self.execute_decomposed_task(decomposed).await?;
        }

        // Evaluate quality
        let quality = self.evaluate_quality(&task).await?;
        task.quality_score = Some(quality);

        // Add to training queue if quality is high enough
        if quality >= self.config.quality_threshold {
            if let Some(output) = &task.output {
                self.training_queue.push(TrainingExample {
                    input: task.input.clone(),
                    output: output.clone(),
                    quality_score: quality,
                    task_type: task.task_type,
                });
                tracing::info!(
                    task_id = %task.id,
                    quality = quality,
                    "added to training queue"
                );
            }
        }

        // Check if we should retrain
        if self.config.auto_retrain && self.training_queue.len() >= self.config.retrain_min_samples {
            self.trigger_retraining().await?;
        }

        self.completed_tasks.insert(task.id.clone(), task.clone());
        Ok(task)
    }

    /// Decompose a complex task into subtasks.
    async fn decompose_task(&self, task: &Task) -> Result<DecomposedTask, AutoError> {
        let mut decomposed = DecomposedTask::new(task.clone());

        // Use the decomposer agent
        let agent = self
            .agents
            .find_agent(TaskType::Decomposition)
            .ok_or_else(|| AutoError::Agent("no decomposer agent".to_string()))?;

        let decompose_task = Task::new(
            "Decompose this task into subtasks",
            TaskType::Decomposition,
            format!("Task: {}\n\nInput: {}", task.description, task.input),
        );

        let response = agent.execute(&decompose_task).await?;

        // Parse the JSON response
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&response) {
            if let Some(subtasks) = json["subtasks"].as_array() {
                for (i, st) in subtasks.iter().enumerate() {
                    let description = st["description"].as_str().unwrap_or("subtask");
                    let task_type_str = st["task_type"].as_str().unwrap_or("Writing");
                    let task_type = match task_type_str {
                        "Research" => TaskType::Research,
                        "Mathematics" => TaskType::Mathematics,
                        "Coding" => TaskType::Coding,
                        "Critique" => TaskType::Critique,
                        _ => TaskType::Writing,
                    };

                    let subtask = Task::new(description, task_type, &task.input)
                        .with_parent(&task.id)
                        .with_priority((100 - i as u8).max(1));

                    // Check dependencies
                    let subtask_id = subtask.id.clone();
                    if let Some(deps) = st["depends_on"].as_array() {
                        for dep in deps {
                            if let Some(dep_idx) = dep.as_u64() {
                                if let Some(dep_task) = decomposed.subtasks.get(dep_idx as usize) {
                                    let dep_id = dep_task.id.clone();
                                    decomposed.add_dependency(&subtask_id, &dep_id);
                                }
                            }
                        }
                    }

                    decomposed.add_subtask(subtask);
                }
            }
        }

        Ok(decomposed)
    }

    /// Execute a simple (non-decomposed) task.
    async fn execute_simple_task(&self, mut task: Task) -> Result<Task, AutoError> {
        task.status = TaskStatus::Running;

        let agent = self
            .agents
            .find_agent(task.task_type)
            .ok_or_else(|| AutoError::Agent(format!("no agent for {:?}", task.task_type)))?;

        match agent.execute(&task).await {
            Ok(output) => {
                task.output = Some(output);
                task.status = TaskStatus::Completed;
            }
            Err(e) => {
                tracing::error!(task_id = %task.id, error = %e, "task failed");
                task.status = TaskStatus::Failed;
                return Err(e);
            }
        }

        Ok(task)
    }

    /// Execute a decomposed task by running subtasks in order.
    async fn execute_decomposed_task(
        &mut self,
        mut decomposed: DecomposedTask,
    ) -> Result<Task, AutoError> {
        let mut completed: HashMap<String, Task> = HashMap::new();
        let mut context = String::new();

        // Execute subtasks in order, respecting dependencies
        for task_id in &decomposed.execution_order {
            // Check if dependencies are satisfied
            if let Some(deps) = decomposed.dependencies.get(task_id) {
                for dep_id in deps {
                    if !completed.contains_key(dep_id) {
                        return Err(AutoError::Decomposition(format!(
                            "dependency {} not satisfied for task {}",
                            dep_id, task_id
                        )));
                    }
                }
            }

            // Find the subtask
            let subtask = decomposed
                .subtasks
                .iter_mut()
                .find(|t| &t.id == task_id)
                .ok_or_else(|| AutoError::Decomposition(format!("subtask {} not found", task_id)))?;

            // Add context from previous subtasks
            if !context.is_empty() {
                subtask.input = format!("{}\n\n## Previous context\n{}", subtask.input, context);
            }

            // Execute
            let result = self.execute_simple_task(subtask.clone()).await?;
            if let Some(output) = &result.output {
                context.push_str(&format!("\n\n### {}\n{}", result.description, output));
            }
            completed.insert(result.id.clone(), result);
        }

        // Combine outputs into final result
        let mut final_output = String::new();
        for task_id in &decomposed.execution_order {
            if let Some(task) = completed.get(task_id) {
                if let Some(output) = &task.output {
                    final_output.push_str(&format!("\n\n## {}\n{}", task.description, output));
                }
            }
        }

        decomposed.original.output = Some(final_output);
        decomposed.original.status = TaskStatus::Completed;
        Ok(decomposed.original)
    }

    /// Evaluate the quality of a completed task.
    async fn evaluate_quality(&self, task: &Task) -> Result<u8, AutoError> {
        let output = task
            .output
            .as_ref()
            .ok_or_else(|| AutoError::Agent("no output to evaluate".to_string()))?;

        let critic = self
            .agents
            .find_agent(TaskType::Critique)
            .ok_or_else(|| AutoError::Agent("no critic agent".to_string()))?;

        let eval_task = Task::new(
            "Evaluate the quality of this output",
            TaskType::Critique,
            format!(
                "Task: {}\n\nOutput to evaluate:\n{}",
                task.description, output
            ),
        );

        let response = critic.execute(&eval_task).await?;

        // Extract quality score from response
        // Look for patterns like "Score: 85" or "Quality: 90/100"
        let score = extract_score(&response).unwrap_or(50);
        Ok(score)
    }

    /// Trigger retraining on accumulated high-quality outputs.
    async fn trigger_retraining(&mut self) -> Result<(), AutoError> {
        tracing::info!(
            samples = self.training_queue.len(),
            "triggering retraining"
        );

        // Convert training examples to the format expected by llm-train
        let examples: Vec<_> = self
            .training_queue
            .drain(..)
            .map(|ex| (ex.input, ex.output))
            .collect();

        self.training
            .add_examples(examples)
            .map_err(|e| AutoError::Training(e.to_string()))?;

        // Note: Actual retraining is resource-intensive and should be scheduled
        // rather than run synchronously. This marks the data as ready for training.
        tracing::info!("training data queued for next training run");

        Ok(())
    }

    /// Get the number of pending training examples.
    pub fn pending_training_samples(&self) -> usize {
        self.training_queue.len()
    }

    /// Get completed tasks.
    pub fn completed_tasks(&self) -> &HashMap<String, Task> {
        &self.completed_tasks
    }
}

/// Extract a numeric score from text.
fn extract_score(text: &str) -> Option<u8> {
    // Look for patterns like "85", "85/100", "Score: 85"
    let patterns = [
        r"(?i)score[:\s]+(\d+)",
        r"(?i)quality[:\s]+(\d+)",
        r"(?i)rating[:\s]+(\d+)",
        r"(\d+)\s*/\s*100",
        r"(?i)is\s+(\d+)",
    ];

    for pattern in patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            if let Some(cap) = re.captures(text) {
                if let Some(m) = cap.get(1) {
                    if let Ok(score) = m.as_str().parse::<u8>() {
                        return Some(score.min(100));
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_score_patterns() {
        assert_eq!(extract_score("Score: 85"), Some(85));
        assert_eq!(extract_score("Quality: 90/100"), Some(90));
        assert_eq!(extract_score("Rating: 75"), Some(75));
        assert_eq!(extract_score("The score is 80"), Some(80));
    }
}
