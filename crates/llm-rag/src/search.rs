use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::hnsw::HnswIndex;

/// Configuration for RAG search.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Number of vector search results.
    pub vector_k: usize,
    /// Number of BM25 keyword results.
    pub bm25_k: usize,
    /// Final number of results after fusion.
    pub top_k: usize,
    /// Reciprocal Rank Fusion constant.
    pub rrf_k: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            vector_k: 5,
            bm25_k: 5,
            top_k: 5,
            rrf_k: 60,
        }
    }
}

/// A single search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub doc_id: String,
    pub chunk_idx: usize,
    pub text: String,
    pub score: f32,
    pub source: String,
}

/// BM25 scoring parameters.
pub struct Bm25Params {
    avg_dl: f32,
    k1: f32,
    b: f32,
}

/// Hybrid search engine: vector similarity + BM25 keyword matching.
pub struct SearchEngine {
    pub index: Arc<RwLock<HnswIndex>>,
    pub doc_texts: Vec<String>,
    pub doc_ids: Vec<String>,
    pub doc_sources: Vec<String>,
    pub chunk_indices: Vec<usize>,
    /// Inverted index: term → (doc_index, term_frequency)
    pub inverted_index: HashMap<String, Vec<(usize, usize)>>,
    /// Document lengths
    pub doc_lengths: Vec<usize>,
    pub bm25_params: Bm25Params,
}

impl Default for SearchEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchEngine {
    pub fn new() -> Self {
        Self {
            index: Arc::new(RwLock::new(HnswIndex::new(16, 8))),
            doc_texts: Vec::new(),
            doc_ids: Vec::new(),
            doc_sources: Vec::new(),
            chunk_indices: Vec::new(),
            inverted_index: HashMap::new(),
            doc_lengths: Vec::new(),
            bm25_params: Bm25Params {
                avg_dl: 0.0,
                k1: 1.2,
                b: 0.75,
            },
        }
    }

    /// Add a document chunk to the index.
    pub fn add(&mut self, doc_id: &str, chunk_idx: usize, text: &str, source: &str, embedding: &[f32]) {
        let idx = self.doc_texts.len();

        // Store metadata
        self.doc_texts.push(text.to_string());
        self.doc_ids.push(doc_id.to_string());
        self.doc_sources.push(source.to_string());
        self.chunk_indices.push(chunk_idx);

        // Tokenize and update inverted index
        let tokens: Vec<String> = text
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        self.doc_lengths.push(tokens.len());

        let mut term_count: HashMap<String, usize> = HashMap::new();
        for token in &tokens {
            *term_count.entry(token.clone()).or_insert(0) += 1;
        }
        for (term, count) in term_count {
            self.inverted_index
                .entry(term)
                .or_default()
                .push((idx, count));
        }

        // Add to HNSW
        self.index.write().insert(idx, embedding.to_vec());

        // Update BM25 params
        let total_len: usize = self.doc_lengths.iter().sum();
        self.bm25_params.avg_dl = total_len as f32 / self.doc_texts.len() as f32;
    }

    /// Hybrid search: vector + BM25 with Reciprocal Rank Fusion.
    pub fn search(&self, query_embedding: &[f32], query_text: &str, config: &SearchConfig) -> Vec<SearchResult> {
        // Vector search
        let vector_results = self.index.read().search(query_embedding, config.vector_k);
        let mut vector_ranks: HashMap<usize, f32> = HashMap::new();
        for (rank, (id, _)) in vector_results.iter().enumerate() {
            vector_ranks.insert(*id, 1.0 / (config.rrf_k as f32 + rank as f32 + 1.0));
        }

        // BM25 search
        let query_terms: Vec<String> = query_text
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let n = self.doc_texts.len() as f32;
        let mut bm25_scores: HashMap<usize, f32> = HashMap::new();

        for term in &query_terms {
            if let Some(postings) = self.inverted_index.get(term) {
                let df = postings.len() as f32;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

                for &(doc_idx, tf) in postings {
                    let dl = self.doc_lengths[doc_idx] as f32;
                    let avg_dl = self.bm25_params.avg_dl;
                    let k1 = self.bm25_params.k1;
                    let b = self.bm25_params.b;

                    let tf_norm = (tf as f32 * (k1 + 1.0))
                        / (tf as f32 + k1 * (1.0 - b + b * dl / avg_dl));
                    let score = idf * tf_norm;

                    *bm25_scores.entry(doc_idx).or_insert(0.0) += score;
                }
            }
        }

        // Convert BM25 scores to ranks for RRF
        let mut bm25_ranked: Vec<(usize, f32)> = bm25_scores.into_iter().collect();
        bm25_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut bm25_ranks: HashMap<usize, f32> = HashMap::new();
        for (rank, (id, _)) in bm25_ranked.iter().enumerate().take(config.bm25_k) {
            bm25_ranks.insert(*id, 1.0 / (config.rrf_k as f32 + rank as f32 + 1.0));
        }

        // Combine with RRF
        let mut combined: HashMap<usize, f32> = HashMap::new();
        for (&id, &rank) in &vector_ranks {
            *combined.entry(id).or_insert(0.0) += rank;
        }
        for (&id, &rank) in &bm25_ranks {
            *combined.entry(id).or_insert(0.0) += rank;
        }

        let mut results: Vec<(usize, f32)> = combined.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results
            .into_iter()
            .take(config.top_k)
            .map(|(id, score)| SearchResult {
                doc_id: self.doc_ids[id].clone(),
                chunk_idx: self.chunk_indices[id],
                text: self.doc_texts[id].clone(),
                score,
                source: self.doc_sources[id].clone(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm25_basic_search() {
        const EMBED_DIM: usize = 384;
        let mut engine = SearchEngine::new();
        engine.add("doc1", 0, "quantum mechanics wave function", "test.pdf", &vec![0.0; EMBED_DIM]);
        engine.add("doc2", 0, "classical mechanics Newton", "test.pdf", &vec![0.0; EMBED_DIM]);
        engine.add("doc3", 0, "quantum field theory operators", "test.pdf", &vec![0.0; EMBED_DIM]);

        let query_emb = vec![0.0; EMBED_DIM];
        let results = engine.search(&query_emb, "quantum", &SearchConfig::default());
        assert!(!results.is_empty());
        // "quantum" appears in doc1 and doc3, should rank higher
    }
}
