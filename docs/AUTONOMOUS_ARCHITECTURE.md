# ThotBook-AI Autonomous Research System

## Vision

A fully autonomous, self-training research LLM capable of:
1. **Self-improvement** — continuous training on its own outputs and a local session-log corpus
2. **Task decomposition** — breaking complex research into subtasks assigned to specialized sub-agents
3. **Research paper generation** — producing publication-ready papers without external model dependencies
4. **Autonomous learning** — ingesting new knowledge and retraining incrementally

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AUTONOMOUS ORCHESTRATOR                             │
│                              (llm-auto crate)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Task Router │  │ Sub-Agent   │  │ Training    │  │ Research Paper      │ │
│  │             │──│ Spawner     │──│ Scheduler   │──│ Generator           │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           ▼                          ▼                          ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  M1: llm-corpus  │      │  M3: llm-train   │      │  M5: llm-serve   │
│  (ingestion)     │─────▶│  (LoRA training) │─────▶│  (inference)     │
└──────────────────┘      └──────────────────┘      └──────────────────┘
         │                         │                         │
         ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  M2: llm-tokenize│      │  M4: llm-rag     │      │  Embeddings API  │
│  (Arrow shards)  │      │  (HNSW + BM25)   │◀────▶│  /v1/embeddings  │
└──────────────────┘      └──────────────────┘      └──────────────────┘
```

## Data Flow

### 1. Continuous Ingestion (M1)
```
session-corpus.db ──┬──▶ llm-corpus ──▶ Delta Lake ──▶ llm-tokenize ──▶ Arrow IPC shards
thotbook.db ────────┘         │
                        ▼
              [Secret redaction: 13 patterns]
```

### 2. Training Pipeline (M3)
```
Arrow IPC shards ──▶ ArrowDataset ──▶ Trainer ──▶ LoRA adapters
                           │
                           ▼
                    [Qwen3-4B base model]
                           │
                           ▼
                    [Forward pass with LoRA]
                           │
                           ▼
                    [Cross-entropy loss]
                           │
                           ▼
                    [Backward + AdamW update]
                           │
                           ▼
                    [Checkpoint LoRA weights]
```

### 3. Self-Improvement Loop
```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   1. Generate research output (M5)                              │
│            │                                                    │
│            ▼                                                    │
│   2. Evaluate quality (self-critique)                          │
│            │                                                    │
│            ▼                                                    │
│   3. If quality > threshold:                                    │
│      - Add to training corpus (M1)                             │
│      - Retrain on improved data (M3)                           │
│            │                                                    │
│            ▼                                                    │
│   4. Deploy new weights (M5 hot-reload)                        │
│            │                                                    │
│            └──────────────────────────────────────────────────┐ │
│                                                               │ │
└───────────────────────────────────────────────────────────────┘ │
                                                                  │
              (loop continues)◀───────────────────────────────────┘
```

## Sub-Agent System

### Agent Types
| Agent | Specialization | Model |
|-------|----------------|-------|
| `researcher` | Literature review, arxiv search | thotbook-ai |
| `mathematician` | Proofs, equations, derivations | thotbook-ai |
| `coder` | Implementation, testing | thotbook-ai |
| `writer` | LaTeX, prose, citations | thotbook-ai |
| `critic` | Quality evaluation, feedback | thotbook-ai |
| `trainer` | Continuous learning orchestration | thotbook-ai |

### Task Decomposition Protocol
```
User Query: "Write a paper on rough volatility"
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR: Decompose into subtasks                       │
│                                                             │
│ 1. [researcher] Search arxiv for rough volatility papers    │
│ 2. [mathematician] Derive key equations (Volterra, BSDE)    │
│ 3. [coder] Implement numerical simulations                  │
│ 4. [writer] Draft LaTeX sections                            │
│ 5. [critic] Review and suggest improvements                 │
│ 6. [trainer] Learn from this research session               │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 2: Embeddings & RAG (Priority: HIGH)
1. **Add /v1/embeddings endpoint** — expose hidden state mean pooling
2. **Wire llm-rag to llm-serve** — real vector retrieval
3. **Build HNSW index from corpus** — on startup or on-demand

### Phase 3: Autonomous Orchestration (Priority: HIGH)
1. **Create llm-auto crate** — task router, sub-agent spawner
2. **Implement self-critique** — quality evaluation prompts
3. **Training scheduler** — periodic retrase 1: Core Training (Priority: HIGH)
1. **Fix M3 trainer.rs** — implement real forward/backward with LoRA
2. **Add ArrowDataset** — read from M2's tokenized shards
3. **Integrate Qwen3-4B** — smaller model for faster training on CPU

### Phaining on new data

### Phase 4: Research Paper Pipeline (Priority: MEDIUM)
1. **LaTeX template system** — standard research paper format
2. **Citation manager** — arxiv integration
3. **Figure generation** — TikZ/matplotlib from equations

## Model Requirements

### Training (Qwen3-4B)
- ~8GB RAM for model + gradients
- LoRA rank 8-16 reduces trainable params to ~20M
- CPU training feasible but slow (~100 tokens/sec)

### Inference (Qwen3-8B)
- ~16GB RAM for f16 weights
- CPU inference ~10 tokens/sec
- Already implemented in llm-serve

## Session-Log Corpus Integration

A local session-log corpus of configurable size (a directory of Markdown
documents, each with a `## Goal` / `## Summary` section pair) provides:
- **Training data** — technical writing, code, problem-solving
- **Evaluation data** — past solutions for self-critique
- **Knowledge base** — RAG retrieval for current tasks

### Continuous Training Schedule
```
Every 24h (or on-demand):
1. Run M1 to ingest new session-log documents
2. Run M2 to tokenize incremental data
3. Run M3 for 1 epoch LoRA fine-tuning
4. Evaluate on held-out set
5. If improved: swap M5 model weights
```

## Success Criteria

1. **Autonomous**: Runs without human intervention for research tasks
2. **Self-improving**: Measurable quality improvement over time
3. **No external deps**: All inference uses local thotbook-ai weights
4. **Research-grade**: Outputs valid LaTeX with correct equations and citations
