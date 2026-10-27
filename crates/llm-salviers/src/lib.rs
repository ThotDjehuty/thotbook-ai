//! `llm-salviers` — Agent dispatch system for thotbook-ai.
//!
//! Loads agent `.agent.md` files, embeds their descriptions, and performs
//! semantic matching to dispatch queries to the appropriate agent.
//! Named after the `sAlvIers/` folder containing all agent definitions.

pub mod error;
pub mod registry;

pub use error::SalviersError;
pub use registry::{AgentDef, AgentRegistry};
