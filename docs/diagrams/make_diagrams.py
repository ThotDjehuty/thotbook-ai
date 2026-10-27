"""Professional architecture diagrams for thotbook-ai and Polarway.

Replaces the ASCII box-drawing schematics in READMEs, Sphinx docs and papers
with a single consistent visual system: phase-grouped, rounded-node flow
diagrams in the HFThot house palette.

Emits both SVG (README / Sphinx, crisp at any zoom) and PDF (pdflatex) so one
source of truth feeds every surface.

Run:  python3 make_diagrams.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent

# -- HFThot house palette (from website assets/diagrams.css) -----------------
# THEME="light" targets README / Sphinx / papers; "dark" targets hfthot-lab.eu,
# whose pages are navy-on-gold. Same geometry, swapped tokens.
THEME = "light"

_LIGHT = dict(NAVY="#22304a", INK="#16233a", MUTED="#6b7a90",
              PAGE="#f7f8fa", NODE="#ffffff",
              GOLD="#b8860b", TEAL="#178f7a", BLUE="#2f6fb5",
              RED="#b5504d", PURPLE="#6a4c93",
              TINT={"#b8860b": "#fdf8ec", "#178f7a": "#edf8f5",
                    "#2f6fb5": "#eef4fb", "#b5504d": "#fbeeee",
                    "#6a4c93": "#f2eef8"})

_DARK = dict(NAVY="#8fa3bf", INK="#f4d03f", MUTED="#a8b3c4",
             PAGE="#0f0f1e", NODE="#1a1a2e",
             GOLD="#d4af37", TEAL="#50e3c2", BLUE="#4a90e2",
             RED="#e2726e", PURPLE="#a98bd4",
             TINT={"#d4af37": "#1d1a12", "#50e3c2": "#0f1f1c",
                   "#4a90e2": "#101827", "#e2726e": "#1f1314",
                   "#a98bd4": "#171327"})


def _pal():
    return _DARK if THEME == "dark" else _LIGHT


def _apply_theme():
    globals().update(_pal())


_apply_theme()   # bind tokens at import so module-level defaults are valid


def _canvas(w: float, h: float, title: str, subtitle: str = ""):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(PAGE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(3.2, 94, title, fontsize=17, fontweight="bold", color=INK,
            va="top", ha="left")
    if subtitle:
        ax.text(3.2, 87.5, subtitle, fontsize=9.5, color=MUTED,
                va="top", ha="left")
    return fig, ax


def group(ax, x, y, w, h, label, color):
    """Dashed phase container with an uppercase coloured caption."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                                fc=TINT.get(color, "#ffffff"), ec=color,
                                lw=1.3, ls=(0, (5, 3)), zorder=1))
    # letter-spacing emulated by inserting thin spaces — matplotlib has no
    # letterspacing kwarg, and the wide-tracked caption is part of the look.
    spaced = "\u2009".join(label.upper())
    ax.text(x + w / 2, y + h + 2.6, spaced, fontsize=8.4,
            fontweight="bold", color=color, ha="center", va="bottom")


def node(ax, x, y, w, h, title, sub="", accent=None, bold=True):
    accent = accent or NAVY
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.4",
                                fc=NODE, ec=accent, lw=1.6, zorder=3))
    cy = y + h / 2 + (1.6 if sub else 0)
    ax.text(x + w / 2, cy, title, fontsize=9.6,
            fontweight="bold" if bold else "normal",
            color=INK, ha="center", va="center", zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 3.0, sub, fontsize=7.6, color=MUTED,
                ha="center", va="center", zorder=4)


def arrow(ax, x1, y1, x2, y2, color=None, rad=0.0, lw=1.7, ls="-"):
    color = color or NAVY
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, lw=lw, color=color,
                                 linestyle=ls, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))


def save(fig, name: str):
    if THEME == "dark":
        name = f"{name}_dark"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight",
                    facecolor=fig.get_facecolor(),
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote {name}.svg / .pdf / .png")


# ===========================================================================
# 1. thotbook-ai pipeline  (replaces the M1..M6 ASCII schematic)
# ===========================================================================
def thotbook_pipeline():
    fig, ax = _canvas(13.6, 6.2, "thotbook-ai — Pipeline Architecture",
                      "Private corpus to OpenAI-compatible endpoint, entirely on CPU")

    # --- band 1: processing phases (group 52..78, nodes 58..71) -----------
    group(ax, 4, 52, 27, 26, "Ingestion", TEAL)
    group(ax, 34, 52, 27, 26, "Training", GOLD)
    group(ax, 64, 52, 32, 26, "Serving", BLUE)

    node(ax, 6.5, 58, 9.5, 13, "M1", "corpus\ningest", TEAL)
    node(ax, 19, 58, 9.5, 13, "M2", "tokenize\nsharded", TEAL)
    node(ax, 36.5, 58, 9.5, 13, "M3", "train\nLoRA", GOLD)
    node(ax, 49, 58, 9.5, 13, "M4", "RAG\nHNSW", GOLD)
    node(ax, 66.5, 58, 12, 13, "M5", "serve\nOpenAI API", BLUE)
    node(ax, 81, 58, 12.5, 13, "M7", "llm-auto\norchestration", BLUE)

    for x1, x2 in [(16, 19), (28.5, 36.5), (46, 49), (58.5, 66.5), (78.5, 81)]:
        arrow(ax, x1, 64.5, x2, 64.5)

    # --- band 2 (left): M6 return path, dipping below the phase groups ----
    ax.add_patch(FancyArrowPatch((51, 57.5), (11.5, 57.5), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.4, color=TEAL,
                                 linestyle=(0, (4, 2)), zorder=2,
                                 connectionstyle="arc3,rad=-0.32"))
    ax.text(31, 39.5, "M6  citation crawler — reference closure",
            fontsize=8, color=TEAL, ha="center")

    # --- band 2 (right): storage exchange, clear of every caption ---------
    ax.annotate("", xy=(78, 31), xytext=(78, 51.5),
                arrowprops=dict(arrowstyle="<->", lw=1.6, color=PURPLE))
    ax.text(80.5, 43.5, "reads / writes", fontsize=8.2, color=PURPLE,
            ha="left", va="center")
    ax.text(80.5, 38.5, "every stage is a\nversioned table", fontsize=7.6,
            color=MUTED, ha="left", va="center", style="italic")

    # --- band 3: storage substrate (group 5..26, caption at 28.6) ---------
    group(ax, 4, 5, 92, 21, "Lakehouse substrate", PURPLE)
    for i, (lbl, sub) in enumerate([("SQLite", "sources"),
                                    ("Apache Arrow", "zero-copy IPC"),
                                    ("Delta Lake", "versioned + time travel"),
                                    ("HNSW index", "vector retrieval")]):
        node(ax, 7 + i * 22, 10, 18, 11, lbl, sub, PURPLE)
    for i in range(3):
        arrow(ax, 25 + i * 22, 15.5, 29 + i * 22, 15.5, PURPLE)
    save(fig, "thotbook_pipeline")


# ===========================================================================
# 2. thotbook-ai verification loop (PINN / privacy audit)
# ===========================================================================
def thotbook_verification():
    fig, ax = _canvas(12.6, 4.4, "thotbook-ai — Verification & Audit Loop",
                      "Retrieval answers what a document says; the solver checks whether it is true")

    group(ax, 4, 42, 43, 30, "Document understanding", TEAL)
    group(ax, 53, 42, 43, 30, "Numerical verification", GOLD)

    node(ax, 7, 52, 16, 14, "llm-ingest", "PDF · Markdown\nagent files", TEAL)
    node(ax, 27, 52, 16, 14, "llm-rag", "HNSW + BM25\nhybrid retrieval", TEAL)
    node(ax, 56, 52, 17, 14, "PINN solver", "thotgrad autodiff\nAdamW → L-BFGS", GOLD)
    node(ax, 77, 52, 16, 14, "Reference", "exact form or\nRadau 1e-12", GOLD)

    arrow(ax, 23, 59, 27, 59, TEAL)
    arrow(ax, 43, 59, 56, 59, NAVY)
    arrow(ax, 73, 59, 77, 59, GOLD)
    ax.text(49.5, 62.5, "governing\nequation", fontsize=7.6, color=MUTED, ha="center")

    group(ax, 4, 8, 92, 22, "Privacy gate before release", RED)
    for i, (lbl, sub) in enumerate([("Loss threshold", "Yeom 2018"),
                                    ("LiRA", "Carlini 2022"),
                                    ("VAE scorer", "agent-agnostic"),
                                    ("Calibration", "Platt + ECE"),
                                    ("Federated", "4 defence families")]):
        node(ax, 7 + i * 17.6, 12, 15, 11, lbl, sub, RED)
    ax.text(50, 33.5, "a model that memorises its corpus is blocked from release",
            fontsize=8.2, color=RED, ha="center", style="italic")
    save(fig, "thotbook_verification")


# ===========================================================================
# 3. Polarway architecture
# ===========================================================================
def polarway_architecture():
    fig, ax = _canvas(13.4, 5.8, "Polarway — System Architecture",
                      "Railway-oriented dataframes over a transactional lakehouse")

    group(ax, 3, 62, 30, 24, "Ingestion", TEAL)
    node(ax, 5.5, 69, 11.5, 12, "WebSocket", "tick streams", TEAL)
    node(ax, 19.5, 69, 11, 12, "REST / gRPC", "pull + push", TEAL)
    ax.text(18, 65.5, "polarway-sources → Arrow RecordBatch",
            fontsize=7.6, color=TEAL, ha="center")

    group(ax, 37, 62, 24, 24, "Dispatch", PURPLE)
    node(ax, 39.5, 69, 19, 12, "polarway-bus", "fan-out · filter\ndrain buffers", PURPLE)
    ax.text(49, 65.5, "backpressure, not unbounded memory",
            fontsize=7.6, color=PURPLE, ha="center")

    group(ax, 65, 62, 31, 24, "Execution", GOLD)
    node(ax, 67.5, 69, 12, 12, "Polars", "vectorised\nkernels", GOLD)
    node(ax, 82, 69, 11.5, 12, "ROP layer", "Result<T,E>\ncomposable", GOLD)
    ax.text(80.5, 65.5, "failures are values, not exceptions",
            fontsize=7.6, color=GOLD, ha="center")

    arrow(ax, 33, 75, 37, 75)
    arrow(ax, 61, 75, 65, 75)

    group(ax, 3, 30, 93, 22, "Storage — polarway-lakehouse", BLUE)
    for i, (lbl, sub) in enumerate([("Delta log", "ordered commits"),
                                    ("ACID append", "71 ms measured"),
                                    ("Time travel", "version-addressed"),
                                    ("Maintenance", "compact · vacuum"),
                                    ("Audit", "who changed what")]):
        node(ax, 6 + i * 18, 34, 16, 12, lbl, sub, BLUE)
    arrow(ax, 49, 62, 49, 52.5, BLUE, ls=(0, (4, 2)), lw=1.4)

    group(ax, 3, 4, 93, 16, "Serving & operations", RED)
    for i, (lbl, sub) in enumerate([("polarway-grpc", "warm engine, multi-language"),
                                    ("polarway-history", "Spark History API compatible"),
                                    ("polarway-distributed", "coordinator · executor · cache")]):
        node(ax, 6 + i * 30.5, 7.5, 28, 9.5, lbl, sub, RED)
    arrow(ax, 49, 30, 49, 20.5, RED, ls=(0, (4, 2)), lw=1.4)
    save(fig, "polarway_architecture")


# ===========================================================================
# 4. Polarway low-latency streaming path (HFT)
# ===========================================================================
def polarway_streaming():
    fig, ax = _canvas(13.4, 4.6, "Polarway — Low-Latency Streaming Path",
                      "One process, no broker, no JVM: socket to aggregate without leaving Rust")

    group(ax, 3, 46, 93, 30, "In-process hot path", TEAL)
    stages = [("Exchange", "WebSocket\nframe"), ("Decode", "serde → Arrow\nzero-copy"),
              ("Bus", "fan-out\nfiltered"), ("Window", "TWAP / VWAP\nrolling"),
              ("Sink", "Delta append\nor query")]
    for i, (lbl, sub) in enumerate(stages):
        node(ax, 6 + i * 18, 53, 15, 14, lbl, sub, TEAL)
        if i < len(stages) - 1:
            arrow(ax, 21 + i * 18, 60, 24 + i * 18, 60, TEAL)
    ax.text(50, 49, "no serialisation boundary, no broker hop, no GC pause",
            fontsize=8.4, color=TEAL, ha="center", style="italic")

    group(ax, 3, 6, 93, 30, "Conventional stack for the same job", RED)
    conv = [("Exchange", "WebSocket"), ("Kafka", "broker hop\n+ disk"),
            ("Flink / Spark", "JVM task\n+ shuffle"), ("Ser/de", "row ↔ column\nconversion"),
            ("Sink", "warehouse\nwrite")]
    for i, (lbl, sub) in enumerate(conv):
        node(ax, 6 + i * 18, 13, 15, 14, lbl, sub, RED)
        if i < len(conv) - 1:
            arrow(ax, 21 + i * 18, 20, 24 + i * 18, 20, RED)
    ax.text(50, 9, "each hop adds a network round trip, a copy, and a GC surface",
            fontsize=8.4, color=RED, ha="center", style="italic")

    ax.annotate("", xy=(99, 60), xytext=(99, 20),
                arrowprops=dict(arrowstyle="<->", lw=1.3, color=MUTED))
    ax.text(101.5, 40, "fewer\nhops", fontsize=8, color=MUTED, ha="center",
            va="center", rotation=90)
    save(fig, "polarway_streaming")


if __name__ == "__main__":
    for theme in ("light", "dark"):
        globals()["THEME"] = theme
        _apply_theme()
        print(f"generating {theme} diagrams:")
        thotbook_pipeline()
        thotbook_verification()
        polarway_architecture()
        polarway_streaming()
    print("done.")
