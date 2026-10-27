/// Sliding-window text chunker with overlap.
///
/// Chunks text into fixed-size token windows with configurable overlap.
/// Uses whitespace tokenization (fast, no model dependency) for chunk sizing.

#[derive(Debug, Clone)]
pub struct ChunkConfig {
    pub max_tokens: usize,
    pub overlap_tokens: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            overlap_tokens: 128,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Chunk {
    pub text: String,
    pub start_offset: usize,
    pub end_offset: usize,
    pub chunk_idx: usize,
}

/// Split text into overlapping chunks by whitespace token count.
pub fn chunk_text(text: &str, config: &ChunkConfig) -> Vec<Chunk> {
    if text.is_empty() || config.max_tokens == 0 {
        return vec![];
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= config.max_tokens {
        return vec![Chunk {
            text: text.to_string(),
            start_offset: 0,
            end_offset: text.len(),
            chunk_idx: 0,
        }];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    let mut idx = 0;

    while start < words.len() {
        let end = (start + config.max_tokens).min(words.len());
        let chunk_words = &words[start..end];
        let chunk_text = chunk_words.join(" ");

        let start_offset = text.find(chunk_words[0]).unwrap_or(0);
        let end_offset = start_offset + chunk_text.len();

        chunks.push(Chunk {
            text: chunk_text,
            start_offset,
            end_offset,
            chunk_idx: idx,
        });

        idx += 1;
        let step = config
            .max_tokens
            .saturating_sub(config.overlap_tokens)
            .max(1);
        start += step;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_single_chunk() {
        let chunks = chunk_text("hello world", &ChunkConfig::default());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_idx, 0);
    }

    #[test]
    fn empty_text_no_chunks() {
        let chunks = chunk_text("", &ChunkConfig::default());
        assert!(chunks.is_empty());
    }

    #[test]
    fn long_text_produces_overlap() {
        let text = (0..1000)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let config = ChunkConfig {
            max_tokens: 100,
            overlap_tokens: 20,
        };
        let chunks = chunk_text(&text, &config);
        assert!(chunks.len() > 1);
        assert!(chunks[1].start_offset < chunks[0].end_offset);
    }
}
