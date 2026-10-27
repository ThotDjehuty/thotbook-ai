//! Sub-agent definitions and spawning.

use serde::{Deserialize, Serialize};

use crate::error::AutoError;
use crate::task::{Task, TaskType};

/// Configuration for a sub-agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Agent name.
    pub name: String,
    /// Task types this agent handles.
    pub task_types: Vec<TaskType>,
    /// System prompt for the agent.
    pub system_prompt: String,
    /// Maximum tokens for response.
    pub max_tokens: usize,
    /// Temperature for sampling.
    pub temperature: f64,
}

impl AgentConfig {
    pub fn researcher() -> Self {
        Self {
            name: "researcher".to_string(),
            task_types: vec![TaskType::Research],
            system_prompt: r#"You are a research specialist. Your task is to:
1. Search for relevant papers and resources
2. Summarize key findings
3. Identify gaps in the literature
4. Suggest research directions

Output should be well-structured markdown with citations."#.to_string(),
            max_tokens: 2048,
            temperature: 0.3,
        }
    }

    pub fn mathematician() -> Self {
        Self {
            name: "mathematician".to_string(),
            task_types: vec![TaskType::Mathematics],
            system_prompt: r#"You are a mathematician specializing in:
- Stochastic calculus and SDEs
- Rough path theory
- Mean field games
- Topological data analysis

For every claim:
1. State the theorem precisely
2. Provide a rigorous proof or proof sketch
3. Write all equations in LaTeX
4. Give examples where helpful

Output should be LaTeX with proper theorem/proof environments."#.to_string(),
            max_tokens: 4096,
            temperature: 0.1,
        }
    }

    pub fn coder() -> Self {
        Self {
            name: "coder".to_string(),
            task_types: vec![TaskType::Coding],
            system_prompt: r#"You are a software engineer specializing in:
- Rust and Python
- Numerical computing
- Machine learning implementations
- High-performance computing

For every task:
1. Understand the mathematical specification
2. Implement clean, tested code
3. Include docstrings and comments
4. Write unit tests

Output should be well-documented code."#.to_string(),
            max_tokens: 4096,
            temperature: 0.2,
        }
    }

    pub fn writer() -> Self {
        Self {
            name: "writer".to_string(),
            task_types: vec![TaskType::Writing, TaskType::PaperGeneration],
            system_prompt: r#"You are an academic writer specializing in:
- Research papers in quantitative finance
- Mathematical physics exposition
- Technical documentation

For every task:
1. Use clear, precise language
2. Structure content logically
3. Format in LaTeX for papers
4. Include proper citations

Output should be publication-ready LaTeX or markdown."#.to_string(),
            max_tokens: 8192,
            temperature: 0.4,
        }
    }

    pub fn critic() -> Self {
        Self {
            name: "critic".to_string(),
            task_types: vec![TaskType::Critique],
            system_prompt: r#"You are a research critic and quality evaluator.

For every piece of work:
1. Check mathematical correctness
2. Evaluate clarity and exposition
3. Identify missing citations
4. Suggest improvements
5. Rate quality on a scale of 0-100

Be constructive but rigorous. Output should be structured feedback."#.to_string(),
            max_tokens: 2048,
            temperature: 0.2,
        }
    }

    pub fn trainer() -> Self {
        Self {
            name: "trainer".to_string(),
            task_types: vec![TaskType::Training],
            system_prompt: r#"You are the training orchestrator.

Your role is to:
1. Evaluate which outputs should be added to training data
2. Decide when to trigger retraining
3. Monitor model quality over time
4. Manage the self-improvement loop

Output should be JSON with training decisions."#.to_string(),
            max_tokens: 1024,
            temperature: 0.1,
        }
    }

    pub fn decomposer() -> Self {
        Self {
            name: "decomposer".to_string(),
            task_types: vec![TaskType::Decomposition],
            system_prompt: r#"You are a task decomposition specialist.

Given a complex task, break it down into:
1. Independent subtasks that can run in parallel
2. Sequential subtasks with dependencies
3. Assign each subtask to the appropriate agent type

Output should be JSON with the following structure:
{
  "subtasks": [
    {"description": "...", "task_type": "Research|Mathematics|Coding|Writing|Critique", "depends_on": []}
  ]
}"#.to_string(),
            max_tokens: 2048,
            temperature: 0.2,
        }
    }
}

/// A sub-agent that can execute tasks.
pub struct Agent {
    pub config: AgentConfig,
    inference_url: String,
}

impl Agent {
    pub fn new(config: AgentConfig, inference_url: impl Into<String>) -> Self {
        Self {
            config,
            inference_url: inference_url.into(),
        }
    }

    /// Execute a task and return the output.
    pub async fn execute(&self, task: &Task) -> Result<String, AutoError> {
        let _prompt = format!(
            "{}\n\n## Task\n{}\n\n## Input\n{}",
            self.config.system_prompt, task.description, task.input
        );

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/v1/chat/completions", self.inference_url))
            .json(&serde_json::json!({
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": format!("## Task\n{}\n\n## Input\n{}", task.description, task.input)}
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }))
            .send()
            .await?;

        let json: serde_json::Value = response.json().await?;
        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| AutoError::Inference("no content in response".to_string()))?;

        Ok(content.to_string())
    }
}

/// Registry of available agents.
pub struct AgentRegistry {
    agents: Vec<Agent>,
}

impl AgentRegistry {
    pub fn new(inference_url: impl Into<String>) -> Self {
        let url = inference_url.into();
        Self {
            agents: vec![
                Agent::new(AgentConfig::researcher(), &url),
                Agent::new(AgentConfig::mathematician(), &url),
                Agent::new(AgentConfig::coder(), &url),
                Agent::new(AgentConfig::writer(), &url),
                Agent::new(AgentConfig::critic(), &url),
                Agent::new(AgentConfig::trainer(), &url),
                Agent::new(AgentConfig::decomposer(), &url),
            ],
        }
    }

    /// Find an agent that can handle the given task type.
    pub fn find_agent(&self, task_type: TaskType) -> Option<&Agent> {
        self.agents
            .iter()
            .find(|a| a.config.task_types.contains(&task_type))
    }
}
