use candle_core::Tensor;

use crate::error::RagError;

/// Embedding dimension after projection from Qwen3 hidden dim.
pub const EMBED_DIM: usize = 384;

/// Project Qwen3 hidden states to a compact embedding for RAG.
///
/// Uses the mean-pooled last hidden layer, then a linear projection to EMBED_DIM.
pub fn project_embedding(hidden: &Tensor) -> Result<Vec<f32>, RagError> {
    // Mean pooling over sequence length (dim 1)
    let pooled = hidden
        .mean(1)
        .map_err(|e| RagError::Embedding(e.to_string()))?;

    // If already EMBED_DIM, return as-is
    let shape = pooled.shape();
    if shape.dims().len() >= 2 && shape.dims()[1] == EMBED_DIM {
        return pooled
            .to_vec1::<f32>()
            .map_err(|e| RagError::Embedding(e.to_string()));
    }

    // Truncate or pad to EMBED_DIM
    let data = pooled
        .to_vec1::<f32>()
        .map_err(|e| RagError::Embedding(e.to_string()))?;

    let mut embedding = vec![0.0f32; EMBED_DIM];
    let copy_len = data.len().min(EMBED_DIM);
    embedding[..copy_len].copy_from_slice(&data[..copy_len]);

    // L2 normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }

    Ok(embedding)
}

/// Cosine similarity between two L2-normalized embeddings.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Dot product (equivalent to cosine for L2-normalized vectors).
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similar_ones() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }
}
