"""Minimal dependency-free notebook builder + executor.

Builds an executed Jupyter nbformat-4 notebook from a list of
(markdown | code) cells:

    from nblite import build
    build("05_dirac_spinors.ipynb", [
        ("md",  "# Title\\n prose"),
        ("code", "import numpy as np\\nprint(np.arange(3))"),
    ])

Code cells are executed IN ORDER in one shared namespace, exactly as a
Jupyter kernel would; stdout is captured into stream outputs and any
exception is captured as an error output and then raised so that the
build fails loudly rather than shipping a half-executed notebook.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import pathlib
import platform
import tempfile
import traceback


_PENDING: list[dict] = []


def emit_gif(anim, fps: int = 20, dpi: int = 90) -> None:
    """Queue a matplotlib animation as an inline GIF output.

    A still frame cannot show a wave packet moving or a particle cloud
    settling, so animations are rendered to GIF with Pillow and embedded in
    the notebook the same way figures are. Call it from a cell; the builder
    drains the queue once the cell finishes.

    The animation's figure is closed afterwards so `_capture_figures` does not
    also emit its last frame as a redundant PNG.
    """
    from matplotlib.animation import PillowWriter
    import matplotlib.pyplot as plt

    # PillowWriter writes through Pillow's save(), which needs a real path:
    # handing it a BytesIO raises "argument should be a str or an
    # os.PathLike object ... not 'BytesIO'". So render to a temp file and
    # read the bytes back.
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp.close()
    try:
        anim.save(tmp.name, writer=PillowWriter(fps=fps), dpi=dpi)
        data = pathlib.Path(tmp.name).read_bytes()
    except Exception as exc:
        _PENDING.append({"output_type": "stream", "name": "stderr",
                         "text": [f"animation failed to render: {exc}\n"]})
        return
    finally:
        pathlib.Path(tmp.name).unlink(missing_ok=True)
        fig = getattr(anim, "_fig", None)
        if fig is not None:
            plt.close(fig)
    _PENDING.append({
        "output_type": "display_data",
        "data": {"image/gif": base64.b64encode(data).decode("ascii")},
        "metadata": {},
    })


def _capture_figures() -> list[dict]:
    """Any matplotlib figures a cell left open become image outputs.

    Without this the builder recorded stdout only, so a notebook of plots
    rendered as a notebook of empty cells. Figures are closed after capture so
    one cell's plot cannot leak into the next. Animations queued by
    `emit_gif` are drained first, so they appear before any still figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)  # headless: no window, no main loop
        import matplotlib.pyplot as plt
    except Exception:
        return []

    outs: list[dict] = []
    while _PENDING:                      # animations first, in call order
        outs.append(_PENDING.pop(0))
    for num in plt.get_fignums():
        fig = plt.figure(num)
        if not fig.get_axes():
            plt.close(fig)
            continue
        png = io.BytesIO()
        try:
            fig.savefig(png, format="png", dpi=110, bbox_inches="tight")
        except Exception as exc:  # a broken figure must not sink the build
            outs.append({"output_type": "stream", "name": "stderr",
                         "text": [f"figure {num} failed to render: {exc}\n"]})
            plt.close(fig)
            continue
        outs.append({
            "output_type": "display_data",
            "data": {"image/png": base64.b64encode(png.getvalue()).decode("ascii")},
            "metadata": {},
        })
        plt.close(fig)
    return outs


def build(out_path: str, cells: list[tuple[str, str]]) -> None:
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python",
                              "version": platform.python_version()},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    ns: dict = {"__name__": "__main__"}
    count = 0
    for kind, src in cells:
        # nbformat 4 only knows "markdown", "code" and "raw"; "md" is not a
        # valid cell_type and some viewers render it as an empty cell.
        cell_type = "markdown" if kind in ("md", "markdown") else kind
        # nbformat >= 4.5 requires a unique cell id; omitting it is a
        # validation warning today and a hard error in future versions.
        cell = {"cell_type": cell_type, "metadata": {},
                "id": f"c{len(nb['cells']) + 1:03d}", "source": src}
        if cell_type == "code":
            count += 1
            buf = io.StringIO()
            outs = []
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(src, "<cell %d>" % count, "exec"), ns)
            except Exception as exc:
                # Report which cell died and why before re-raising: the old
                # handler built an unserialisable dict and lost the message.
                print(f"\ncell {count} failed: {type(exc).__name__}: {exc}")
                print("".join(traceback.format_exc()))
                raise
            if buf.getvalue():
                outs.append({"output_type": "stream", "name": "stdout",
                             "text": buf.getvalue().splitlines(True)})
            outs.extend(_capture_figures())
            cell["outputs"] = outs
            cell["execution_count"] = count
        nb["cells"].append(cell)
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print(f"wrote {out_path} ({len(nb['cells'])} cells, {count} executed)")
