use std::path::Path;

use crate::error::TrainError;

/// Save a training checkpoint.
pub fn save_checkpoint(
    path: &Path,
    step: usize,
    loss: f32,
    adapter_weights: &[(String, Vec<f32>)],
) -> Result<(), TrainError> {
    let checkpoint = serde_json::json!({
        "step": step,
        "loss": loss,
        "adapters": adapter_weights.iter().map(|(name, weights)| {
            serde_json::json!({
                "name": name,
                "weights": weights,
            })
        }).collect::<Vec<_>>(),
    });

    std::fs::write(path, serde_json::to_string_pretty(&checkpoint).unwrap()).map_err(|e| {
        TrainError::Io {
            path: path.to_path_buf(),
            source: e,
        }
    })?;

    tracing::info!(step, loss, path = %path.display(), "checkpoint saved");
    Ok(())
}

/// Load a training checkpoint.
pub fn load_checkpoint(path: &Path) -> Result<(usize, f32), TrainError> {
    let data = std::fs::read_to_string(path).map_err(|e| TrainError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;

    let checkpoint: serde_json::Value =
        serde_json::from_str(&data).map_err(|e| TrainError::Checkpoint(e.to_string()))?;

    let step = checkpoint["step"]
        .as_u64()
        .ok_or_else(|| TrainError::Checkpoint("missing step".to_string()))? as usize;
    let loss = checkpoint["loss"]
        .as_f64()
        .ok_or_else(|| TrainError::Checkpoint("missing loss".to_string()))? as f32;

    Ok((step, loss))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("ckpt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("checkpoint.json");

        save_checkpoint(&path, 100, 0.5, &[("q_proj".into(), vec![0.1; 64])]).unwrap();
        let (step, loss) = load_checkpoint(&path).unwrap();
        assert_eq!(step, 100);
        assert!((loss - 0.5).abs() < 1e-6);

        std::fs::remove_dir_all(dir).unwrap();
    }
}
