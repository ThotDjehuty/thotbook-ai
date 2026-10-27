use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::embed_http::OllamaEmbedder;
use crate::error::RagError;
use crate::hnsw::HnswIndex;
use crate::search::SearchEngine;

/// Everything the engine needs that the HNSW binary does not carry.
///
/// The vector index stores only vectors. Without this sidecar a reloaded store
/// can rank documents but cannot say what any of them are, and BM25 has no
/// inverted index to score against — so the previous save/load pair silently
/// produced a searchable index that returned empty text.
#[derive(Serialize, Deserialize, Default)]
struct StoreSidecar {
    doc_texts: Vec<String>,
    doc_ids: Vec<String>,
    doc_sources: Vec<String>,
    chunk_indices: Vec<usize>,
    embed_model: String,
    embed_dim: usize,
}

/// One line of `chunks.jsonl`, as written by `llm-ingest`.
#[derive(Deserialize)]
struct IngestedChunkLine {
    doc_id: String,
    source: String,
    chunk: ChunkLine,
}

#[derive(Deserialize)]
struct ChunkLine {
    text: String,
    chunk_idx: usize,
}

/// Persistent RAG store: saves/loads the HNSW index and metadata to disk.
pub struct RagStore {
    pub index_path: PathBuf,
    pub engine: SearchEngine,
    pub embed_model: String,
    pub embed_dim: usize,
}

impl RagStore {
    pub fn new(index_dir: &Path) -> Self {
        Self {
            index_path: index_dir.join("rag_index.bin"),
            engine: SearchEngine::new(),
            embed_model: String::new(),
            embed_dim: 0,
        }
    }

    /// Embed every chunk in `chunks.jsonl` and build the index.
    ///
    /// Runs a preflight embed first: a wrong model name otherwise surfaces
    /// thousands of chunks later, or not at all.
    pub fn build_from_chunks(
        &mut self,
        chunks_path: &Path,
        embedder: &mut OllamaEmbedder,
    ) -> Result<usize, RagError> {
        let dim = embedder.preflight()?;
        tracing::info!(model = embedder.model(), dim, "embedding backend ready");

        let body = std::fs::read_to_string(chunks_path).map_err(|e| RagError::Io {
            path: chunks_path.to_path_buf(),
            source: e,
        })?;

        let total = body.lines().filter(|l| !l.trim().is_empty()).count();
        let mut indexed = 0usize;
        let mut skipped = 0usize;

        for (i, line) in body.lines().filter(|l| !l.trim().is_empty()).enumerate() {
            let rec: IngestedChunkLine = match serde_json::from_str(line) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(line = i + 1, error = %e, "skipping malformed chunk");
                    skipped += 1;
                    continue;
                }
            };
            if rec.chunk.text.trim().is_empty() {
                skipped += 1;
                continue;
            }
            let embedding = embedder.embed(&rec.chunk.text)?;
            self.engine.add(
                &rec.doc_id,
                rec.chunk.chunk_idx,
                &rec.chunk.text,
                &rec.source,
                &embedding,
            );
            indexed += 1;
            if indexed % 250 == 0 {
                tracing::info!(indexed, total, "embedding progress");
            }
        }

        self.embed_model = embedder.model().to_string();
        self.embed_dim = embedder.dim().unwrap_or(dim);
        tracing::info!(indexed, skipped, "index built");
        Ok(indexed)
    }

    /// Save the index, its sidecar metadata, and a human-readable summary.
    pub fn save(&self) -> Result<(), RagError> {
        let data = self.engine.index.read().serialize();
        std::fs::write(&self.index_path, &data).map_err(|e| RagError::Io {
            path: self.index_path.clone(),
            source: e,
        })?;

        let sidecar = StoreSidecar {
            doc_texts: self.engine.doc_texts.clone(),
            doc_ids: self.engine.doc_ids.clone(),
            doc_sources: self.engine.doc_sources.clone(),
            chunk_indices: self.engine.chunk_indices.clone(),
            embed_model: self.embed_model.clone(),
            embed_dim: self.embed_dim,
        };
        let side_path = self.index_path.with_extension("docs.json");
        std::fs::write(&side_path, serde_json::to_vec(&sidecar).map_err(|e| {
            RagError::Embedding(format!("serialising sidecar: {e}"))
        })?)
        .map_err(|e| RagError::Io {
            path: side_path,
            source: e,
        })?;

        let meta_path = self.index_path.with_extension("meta.json");
        let meta = serde_json::json!({
            "doc_count": self.engine.doc_texts.len(),
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
            "vectors": self.engine.index.read().len(),
        });
        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap())
            .map_err(|e| RagError::Io {
                path: meta_path,
                source: e,
            })?;

        Ok(())
    }

    /// Load the index from disk.
    pub fn load(&mut self) -> Result<(), RagError> {
        if !self.index_path.exists() {
            return Err(RagError::NotLoaded(format!(
                "index file not found: {}",
                self.index_path.display()
            )));
        }

        let data = std::fs::read(&self.index_path).map_err(|e| RagError::Io {
            path: self.index_path.clone(),
            source: e,
        })?;

        if let Some(index) = HnswIndex::deserialize(&data) {
            self.engine.index = std::sync::Arc::new(parking_lot::RwLock::new(index));
        }

        // Restore the text and BM25 state. Loading vectors alone yields an
        // index that ranks documents it cannot name.
        let side_path = self.index_path.with_extension("docs.json");
        if side_path.exists() {
            let raw = std::fs::read(&side_path).map_err(|e| RagError::Io {
                path: side_path.clone(),
                source: e,
            })?;
            let side: StoreSidecar = serde_json::from_slice(&raw)
                .map_err(|e| RagError::Embedding(format!("reading sidecar: {e}")))?;

            // Rebuilding through add() regenerates the inverted index and BM25
            // averages, so they cannot drift from the texts.
            let vectors: Vec<Vec<f32>> = {
                let idx = self.engine.index.read();
                (0..side.doc_texts.len()).map(|i| idx.vector(i).unwrap_or_default()).collect()
            };
            let mut engine = SearchEngine::new();
            for i in 0..side.doc_texts.len() {
                engine.add(
                    &side.doc_ids[i],
                    side.chunk_indices[i],
                    &side.doc_texts[i],
                    &side.doc_sources[i],
                    &vectors[i],
                );
            }
            self.engine = engine;
            self.embed_model = side.embed_model;
            self.embed_dim = side.embed_dim;
        }

        Ok(())
    }

    /// Check if an index exists on disk.
    pub fn exists(&self) -> bool {
        self.index_path.exists()
    }
}
