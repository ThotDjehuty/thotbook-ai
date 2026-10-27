#!/usr/bin/env bash
# arxiv_bundle.sh — build the arXiv submission bundle for thotbook_ai_arxiv
# (ArXiv Submission Agent protocol, PART C2 — idempotent rerun). Local-only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TEX=thotbook_ai_arxiv.tex
FRAGS=(frag_info_geometry.tex frag_dirac.tex)
SUB=arxiv_submission

rm -rf "$SUB" "$SUB.tar.gz"
mkdir -p "$SUB"
cp "$TEX" "${FRAGS[@]}" "$SUB/"

# last honest state: recompile INSIDE the bundle to prove self-containment
( cd "$SUB" && pdflatex -interaction=nonstopmode "$TEX" >/dev/null 2>&1
  pdflatex -interaction=nonstopmode "$TEX" >/dev/null 2>&1
  pdflatex -interaction=nonstopmode "$TEX" >/dev/null 2>&1 )
LOG="${TEX%.tex}.log"
if ! ( cd "$SUB" && [ -s "${TEX%.tex}.pdf" ] ); then echo "bundle compile failed"; exit 1; fi
if grep -qE "undefined|multiply|^! " "$SUB/$LOG"; then
  echo "grep gate failed in bundle"; grep -E "undefined|multiply|^! " "$SUB/$LOG" | head; exit 1
fi
echo "in-bundle compile gate: PASS  log: $SUB/$LOG"
( cd "$SUB" && grep "Output written" "$LOG" )

# A1/A2 verification: no absolute paths, no spaces or shell metacharacters
if grep -rn "/Users/\|/home/" "$SUB" --include='*.tex'; then
  echo "A1 FAIL: absolute paths in bundle"; exit 1
fi
bad_files=$(find "$SUB" -name '* *' -o -name '*&*' -o -name '*:*' | head -5)
if [ -n "$bad_files" ]; then echo "A2 FAIL: $bad_files"; exit 1; fi

# C2: ship ONLY what AutoTeX needs. The in-bundle compile above was the proof
# of self-containment; its by-products are scratch and a stale .aux can make
# AutoTeX resolve references against the wrong pass.
rm -f "$SUB"/*.aux "$SUB"/*.log "$SUB"/*.out "$SUB"/*.toc "$SUB"/*.pdf

tar -czf "arxiv_submission.tar.gz" -C "$SUB" .
echo "=== bundle contents ==="
tar -tzf arxiv_submission.tar.gz
echo "=== sha256 ==="
shasum -a 256 "arxiv_submission.tar.gz"
