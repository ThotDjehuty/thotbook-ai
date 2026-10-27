"""Query several local Ollama models in parallel on physics questions drawn
from the real corpus, and cache the answers for 03_federated_ensemble.ipynb.

Parallel sub-agent design: each model is an independent 'sub-agent'. They run
concurrently in threads (Ollama serialises GPU/CPU work internally, but the
HTTP round-trips and prompt-eval overlap), and their answers are combined by
agreement voting rather than trusting any single model.

Run:  python3 run_ensemble.py
Out:  data/ensemble_results.json
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OLLAMA = "http://127.0.0.1:11434"
DATA = Path(__file__).parent / "data"
MODELS = ["qwen2.5-coder:7b", "devstral:latest", "glm-4.7-flash:latest"]
PARALLEL = False   # see note in main(): 38 GB of models > 32 GB RAM

# Physics questions with objectively checkable short answers, all on topics
# the corpus actually covers (verified in notebook 04 §8).
QUESTIONS = [
    {"q": "In the Wheeler-DeWitt equation for closed FLRW minisuperspace with "
          "cosmological constant Lambda, at which scale factor a does the "
          "classical turning point occur? Answer with a formula only.",
     "key": ["sqrt(3/", "3/lambda", "sqrt(3/lambda)", "(3/lambda)^(1/2)", "√(3/"]},
    {"q": "For the stationary Dirac equation, state the relativistic dispersion "
          "relation linking energy E, momentum k and mass m. Formula only.",
     "key": ["e^2 = k^2 + m^2", "e²=k²+m²", "e^2=k^2+m^2", "e**2 = k**2 + m**2",
             "e2 = k2 + m2"]},
    {"q": "In Brans-Dicke cosmology with dust, what value does the scale-factor "
          "exponent q approach as omega tends to infinity? Number only.",
     "key": ["2/3", "0.667", "0.6667", "two thirds", "2 / 3"]},
    {"q": "What is the name of the parameter p in the Wheeler-DeWitt equation "
          "that encodes the operator-ordering ambiguity? Two words.",
     "key": ["factor ordering", "factor-ordering", "ordering parameter"]},
    {"q": "In LoRA fine-tuning, a weight update is written as the product of "
          "two matrices. If the rank is r and the dimension d, how many "
          "trainable parameters does one adapter have? Formula only.",
     "key": ["2rd", "2*r*d", "2 r d", "2dr", "2*d*r", "r*d + d*r", "2 x r x d"]},
]


def ask(model: str, prompt: str, timeout: int = 600) -> dict:
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0, "num_predict": 80},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        return {"model": model, "answer": out.get("response", "").strip(),
                "seconds": time.time() - t0, "ok": True}
    except Exception as exc:                       # timeout, OOM, model missing
        return {"model": model, "answer": "", "seconds": time.time() - t0,
                "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def graded(answer: str, keys: list[str]) -> bool:
    a = answer.lower().replace(" ", "")
    return any(k.lower().replace(" ", "") in a for k in keys)


def main() -> None:
    try:
        urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=10)
    except Exception as exc:
        print(f"Ollama unreachable ({exc}) — not writing cache.")
        return

    results = []
    for qi, item in enumerate(QUESTIONS):
        print(f"\n[{qi+1}/{len(QUESTIONS)}] {item['q'][:72]}…")
        t0 = time.time()
        if PARALLEL:
            with ThreadPoolExecutor(max_workers=len(MODELS)) as ex:
                answers = list(ex.map(lambda m: ask(m, item["q"]), MODELS))
        else:
            # Sequential: the three models total ~38 GB, which does not fit in
            # this machine's 32 GB RAM. Loading them concurrently thrashes and
            # every request times out; one at a time works.
            answers = [ask(m, item["q"]) for m in MODELS]
        wall = time.time() - t0
        for a in answers:
            a["correct"] = graded(a["answer"], item["key"]) if a["ok"] else False
            flag = "OK " if a["ok"] else "ERR"
            print(f"   {flag} {a['model']:24s} {a['seconds']:6.1f}s "
                  f"correct={a['correct']}  {a['answer'][:56]!r}")
        results.append({"question": item["q"], "keys": item["key"],
                        "answers": answers, "wall_seconds": wall,
                        "serial_seconds": sum(a["seconds"] for a in answers)})

    DATA.mkdir(exist_ok=True)
    json.dump({"models": MODELS, "results": results},
              open(DATA / "ensemble_results.json", "w"), indent=2)
    print(f"\nwrote {DATA/'ensemble_results.json'}")


if __name__ == "__main__":
    main()
