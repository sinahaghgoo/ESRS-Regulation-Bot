#!/usr/bin/env bash
# ============================================================================
# run_remaining.sh — Execute the remaining evaluation + verification steps
#
# Run this on your local macOS machine with the Python 3.12 venv active.
# Prerequisites:
#   - cd ESRS-Regulation-Bot
#   - source .venv/bin/activate
#   - Ensure .env has OPENAI_API_KEY (and COHERE_API_KEY if using reranker)
#   - Qdrant must be running (or qdrant_storage artifacts must exist)
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "============================================"
echo "Step 1: Run pytest (all unit tests)"
echo "============================================"
PYTHONPATH=. python -m pytest tests/ -v --tb=short
echo ""

echo "============================================"
echo "Step 2: Run evaluation notebooks 05-08"
echo "============================================"
echo "  Tip: Set SMOKE_TEST = False in each notebook"
echo "  for full evaluation, or leave True for a quick check."
echo ""

for nb in \
    notebooks/05_rq1_chunking_evaluation.ipynb \
    notebooks/06_rq2_retrieval_evaluation.ipynb \
    notebooks/07_rq3_generation_evaluation_and_llm_judge.ipynb \
    notebooks/08_best_pipeline_frontend_validation.ipynb; do
    echo "  Running: $nb"
    jupyter nbconvert --to notebook --execute "$nb" \
        --ExecutePreprocessor.timeout=600 \
        --output "$(basename "$nb")" \
        --output-dir notebooks/ 2>&1 | tail -3
    echo "  Done: $nb"
    echo ""
done

echo "============================================"
echo "Step 3: Verify best pipeline manifest"
echo "============================================"
if [ -f config/best_pipeline.yaml ]; then
    echo "  Manifest found: config/best_pipeline.yaml"
    cat config/best_pipeline.yaml
else
    echo "  WARNING: config/best_pipeline.yaml not found!"
    echo "  Notebook 08 may not have run successfully."
fi
echo ""

echo "============================================"
echo "Step 4: Launch Streamlit for verification"
echo "============================================"
echo "  Run manually:  streamlit run streamlit_app/app.py"
echo "  Then verify:"
echo "    - Sidebar shows the selected pipeline name"
echo "    - A test query returns a sensible answer with page references"
echo ""
echo "All automated steps complete!"
