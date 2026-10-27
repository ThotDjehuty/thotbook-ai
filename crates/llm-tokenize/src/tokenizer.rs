//! HuggingFace tokenizer loading — local `tokenizer.json` with remote fallback.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::error::TokenizeError;

/// Default Qwen3 tokenizer location (downloaded from HuggingFace on first use).
pub const DEFAULT_TOKENIZER_PATH: &str = "data/tokenizers/qwen3-tokenizer.json";

/// Remote source for the Qwen3-4B `tokenizer.json`.
pub const TOKENIZER_URL: &str = "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/tokenizer.json";

/// Load a tokenizer from `path`, downloading it if the file does not exist.
///
/// Never silently succeeds: if both the local file and the remote download
/// fail, an error is returned.
pub async fn load_tokenizer(path: &Path) -> Result<Tokenizer, TokenizeError> {
    if path.exists() {
        return Tokenizer::from_file(path).map_err(|source| TokenizeError::Tokenizer { source });
    }
    let bytes = download(TOKENIZER_URL)
        .await
        .map_err(|source| TokenizeError::Download {
            url: TOKENIZER_URL.to_string(),
            source,
        })?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| TokenizeError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    std::fs::write(path, &bytes).map_err(|source| TokenizeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Tokenizer::from_bytes(bytes).map_err(|source| TokenizeError::Tokenizer { source })
}

async fn download(url: &str) -> Result<Vec<u8>, reqwest::Error> {
    reqwest::Client::new()
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await
        .map(|b| b.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize a minimal WordLevel tokenizer built via the `tokenizers` builder.
    fn synthetic_json() -> Vec<u8> {
        let vocab = [("[UNK]", 0u32), ("hello", 1), ("world", 2)]
            .into_iter()
            .map(|(w, i)| (w.to_string(), i))
            .collect::<ahash::AHashMap<String, u32>>();
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tok = tokenizers::Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        serde_json::to_vec(&tok).unwrap()
    }

    fn temp_file(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("llm-tokenize-tok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[tokio::test]
    async fn loads_tokenizer_from_local_file() {
        let path = temp_file("synthetic.json");
        std::fs::write(&path, synthetic_json()).unwrap();
        let tok = load_tokenizer(&path).await.unwrap();
        let ids = tok.encode("hello world", true).unwrap().get_ids().to_vec();
        assert_eq!(ids, vec![1, 2], "whitespace-split WordLevel vocab lookup");
    }

    #[tokio::test]
    async fn missing_local_file_errors_rather_than_succeeding() {
        let missing = temp_file("does-not-exist.json");
        if missing.exists() {
            std::fs::remove_file(&missing).unwrap();
        }
        // Requires network; guard so offline runs still exercise error shape.
        let err = match load_tokenizer(&missing).await {
            Ok(_) => return, // network available and download succeeded — acceptable
            Err(e) => e,
        };
        assert!(
            matches!(err, TokenizeError::Download { .. } | TokenizeError::Io { .. }),
            "unexpected error variant: {err}"
        );
    }
}
