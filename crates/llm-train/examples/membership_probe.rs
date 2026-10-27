//! Real, small-scale LoRA training probe for the membership-inference
//! privacy audit (see docs/source/algorithms/membership_inference.rst).
//!
//! Trains a tiny autoregressive character-level LM with a real LoRA
//! adapter (mirroring `llm_train::lora::LoraAdapter`'s `h' = h + scale*(h@A@B)`
//! update) using candle's autograd and `llm_train::optimizer::AdamWOptimizer`
//! — the same optimizer implementation documented in
//! docs/source/algorithms/training.rst. The base embedding + unembedding
//! are frozen (as in real LoRA); only the A/B adapter matrices train.
//!
//! Not a mock: real forward pass, real backward pass, real AdamW updates.
//! Deliberately tiny (d=32, rank=4, ~80-token vocab) so it runs in seconds
//! on CPU for the companion notebook, instead of requiring the full
//! Qwen3-8B weights.
//!
//! Usage:
//!   cargo run --release --example membership_probe -- \
//!     --member data/member.jsonl --nonmember data/nonmember.jsonl \
//!     --epochs 30 --lr 0.01 --rank 4 --out /tmp/probe_result.json

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor, Var, D};
use candle_nn::ops::{log_softmax, softmax};
use serde::{Deserialize, Serialize};

use llm_train::optimizer::AdamWOptimizer;

const DEFAULT_HIDDEN_DIM: usize = 32;

#[derive(Deserialize)]
struct Example {
    id: String,
    text: String,
}

#[derive(Serialize)]
struct Record {
    id: String,
    split: String,
    loss: f32,
    entropy: f32,
    embedding: Vec<f32>,
}

/// Deterministic splitmix64 PRNG so runs are reproducible without pulling
/// in an extra `rand` dependency.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        // map to roughly N(0, 0.2^2) via a crude Box-Muller-free scaling
        ((z as f64 / u64::MAX as f64) as f32 - 0.5) * 0.4
    }
    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

fn read_jsonl(path: &PathBuf) -> Vec<Example> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {:?}: {e}", path))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("bad jsonl line: {e}")))
        .collect()
}

fn build_vocab(examples: &[&Example]) -> HashMap<char, u32> {
    let mut chars: Vec<char> = examples
        .iter()
        .flat_map(|e| e.text.chars())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    chars.sort();
    chars
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c, i as u32))
        .collect()
}

fn encode(text: &str, vocab: &HashMap<char, u32>) -> Vec<u32> {
    text.chars().filter_map(|c| vocab.get(&c).copied()).collect()
}

/// Forward pass for one sequence: embed -> LoRA-adapt -> unembed -> shifted
/// cross-entropy. Returns (mean_loss, mean_entropy, mean_pooled_adapted_hidden).
fn forward_sequence(
    ids: &[u32],
    embed: &Tensor,   // [V, d], frozen
    unembed: &Tensor, // [d, V], frozen
    lora_a: &Var,      // [d, r]
    lora_b: &Var,      // [r, d]
    scale: f32,
    device: &Device,
) -> candle_core::Result<(Tensor, f32, Vec<f32>)> {
    let t = ids.len();
    let ids_t = Tensor::from_slice(ids, (t,), device)?;
    let h = embed.index_select(&ids_t, 0)?; // [t, d]

    let adapt = h.matmul(lora_a.as_tensor())?.matmul(lora_b.as_tensor())?; // [t, d]
    let adapt = (adapt * scale as f64)?;
    let h_prime = (&h + &adapt)?; // [t, d] — real LoRA residual: h' = h + scale*(h@A@B)

    let logits = h_prime.matmul(unembed)?; // [t, V]
    let log_probs = log_softmax(&logits, D::Minus1)?;
    let probs = softmax(&logits, D::Minus1)?;

    // Shifted next-token prediction: position i predicts token i+1.
    let shift_log_probs = log_probs.narrow(0, 0, t - 1)?; // [t-1, V]
    let targets = Tensor::from_slice(&ids[1..], (t - 1,), device)?;
    let selected = shift_log_probs
        .gather(&targets.unsqueeze(1)?, 1)?
        .squeeze(1)?; // [t-1]
    let loss = selected.neg()?.mean_all()?; // positive NLL, real cross-entropy

    // Mean pooled adapted hidden state — the exact mean-pooling formula
    // documented in docs/source/algorithms/embeddings.rst.
    let pooled = h_prime.mean(0)?; // [d]
    let embedding_vec: Vec<f32> = pooled.to_dtype(DType::F32)?.to_vec1()?;

    // Token-level entropy H(x) = -sum p log p, averaged over positions.
    let entropy_per_pos = (probs.clone() * log_probs.neg()?)?.sum(D::Minus1)?; // [t]
    let entropy = entropy_per_pos.mean_all()?.to_scalar::<f32>()?;

    Ok((loss, entropy, embedding_vec))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| default.to_string())
    };

    let member_path = PathBuf::from(get("--member", "data/member.jsonl"));
    let nonmember_path = PathBuf::from(get("--nonmember", "data/nonmember.jsonl"));
    let epochs: usize = get("--epochs", "30").parse()?;
    let lr: f32 = get("--lr", "0.01").parse()?;
    let rank: usize = get("--rank", "4").parse()?;
    let out_path = get("--out", "/tmp/probe_result.json");
    let hidden_dim: usize = get("--hidden", &DEFAULT_HIDDEN_DIM.to_string()).parse()?;
    // Cap examples per split — used to create a deliberate overfitting regime,
    // the standard way the MIA literature exposes memorization.
    let limit: usize = get("--limit", "0").parse()?;

    let mut members = read_jsonl(&member_path);
    let mut nonmembers = read_jsonl(&nonmember_path);
    if limit > 0 {
        members.truncate(limit);
        nonmembers.truncate(limit);
    }

    let all_refs: Vec<&Example> = members.iter().chain(nonmembers.iter()).collect();
    let vocab = build_vocab(&all_refs);
    let vocab_size = vocab.len().max(2);

    let device = Device::Cpu;
    let mut rng = SplitMix64(0xC0FFEE);

    // Frozen "pretrained" base — fixed, never updated (real LoRA assumption).
    let embed = Tensor::from_vec(
        rng.fill(vocab_size * hidden_dim),
        (vocab_size, hidden_dim),
        &device,
    )?;
    let unembed = Tensor::from_vec(
        rng.fill(hidden_dim * vocab_size),
        (hidden_dim, vocab_size),
        &device,
    )?;

    // Trainable LoRA adapter: A in [d, r], B in [r, d], B initialized to
    // zero so the adapter starts as a no-op (standard LoRA init).
    let lora_a = Var::from_tensor(&Tensor::from_vec(
        rng.fill(hidden_dim * rank),
        (hidden_dim, rank),
        &device,
    )?)?;
    let lora_b = Var::from_tensor(&Tensor::zeros((rank, hidden_dim), DType::F32, &device)?)?;
    let alpha = (2 * rank) as f32;
    let scale = alpha / rank as f32;

    let mut opt = AdamWOptimizer::new(lr, 0.01);

    let train_ids: Vec<Vec<u32>> = members
        .iter()
        .map(|e| encode(&e.text, &vocab))
        .filter(|ids| ids.len() >= 2)
        .collect();

    let mut loss_history = Vec::new();
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        for ids in &train_ids {
            let (loss, _entropy, _emb) =
                forward_sequence(ids, &embed, &unembed, &lora_a, &lora_b, scale, &device)?;
            let grads = loss.backward()?;

            let grad_a = grads.get(&lora_a).expect("grad for lora_a").clone();
            let grad_b = grads.get(&lora_b).expect("grad for lora_b").clone();

            // One timestep per optimiser step, not per parameter: both
            // tensors below belong to the same step.
            opt.step();

            let mut a_tensor = lora_a.as_tensor().clone();
            opt.update_at("lora_a", &mut a_tensor, &grad_a, lr)?;
            lora_a.set(&a_tensor)?;

            let mut b_tensor = lora_b.as_tensor().clone();
            opt.update_at("lora_b", &mut b_tensor, &grad_b, lr)?;
            lora_b.set(&b_tensor)?;

            epoch_loss += loss.to_scalar::<f32>()?;
        }
        epoch_loss /= train_ids.len().max(1) as f32;
        loss_history.push(epoch_loss);
        if epoch % 5 == 0 || epoch == epochs - 1 {
            eprintln!("epoch {epoch:>3}  mean_train_loss = {epoch_loss:.4}");
        }
    }

    // Final (no-grad) evaluation pass over BOTH splits.
    let mut records = Vec::new();
    for (split, examples) in [("member", &members), ("nonmember", &nonmembers)] {
        for ex in examples {
            let ids = encode(&ex.text, &vocab);
            if ids.len() < 2 {
                continue;
            }
            let (loss, entropy, embedding) =
                forward_sequence(&ids, &embed, &unembed, &lora_a, &lora_b, scale, &device)?;
            records.push(Record {
                id: ex.id.clone(),
                split: split.to_string(),
                loss: loss.to_scalar::<f32>()?,
                entropy,
                embedding,
            });
        }
    }

    let output = serde_json::json!({
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "lora_rank": rank,
        "epochs": epochs,
        "learning_rate": lr,
        "train_loss_history": loss_history,
        "records": records,
    });
    fs::write(&out_path, serde_json::to_string_pretty(&output)?)?;
    eprintln!("wrote {} records to {out_path}", records.len());

    Ok(())
}
