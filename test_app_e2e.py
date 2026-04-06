"""End-to-end test for the Streamlit app pipeline."""
import sys
import time
import os

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

def log(msg):
    print(msg, flush=True)

log("STEP 1: Loading config...")
from rag.config.settings import load_best_pipeline_config, load_esrs_context_decision

config = load_best_pipeline_config()
pipeline_name = config.pop("_best_pipeline_name", "default")
config.pop("_selection_reason", "")
esrs_helps = load_esrs_context_decision()
log(f"  Pipeline: {pipeline_name}")
log(f"  ESRS context helps: {esrs_helps}")

log("STEP 2: Loading pipeline...")
t0 = time.time()
from rag.pipeline.rq1_runner import load_pipeline
log("  Imports done, calling load_pipeline()...")
retriever, cfg = load_pipeline(config)
log(f"  Pipeline loaded in {time.time() - t0:.1f}s")

log("STEP 3: Single company query (Aker BP)...")
query = "How has the company conducted its double materiality assessment?"
from rag.pipeline.rq1_runner import run_query

t0 = time.time()
result = run_query(
    query=query,
    companies=["Aker BP"],
    esrs_standard="",
    retriever=retriever,
    config=cfg,
    use_esrs_context=esrs_helps,
)
log(f"  Query time: {time.time() - t0:.1f}s")

data = result["per_company"]["Aker BP"]
log(f"  Compliance level: {data.get('compliance_level', 'N/A')}")
log(f"  Answer: {data.get('direct_answer', '')[:300]}")
log(f"  Key findings: {len(data.get('key_findings', []))}")
log(f"  Chunks: {len(result['per_company_chunks']['Aker BP'])}")

log("STEP 4: Multi-company query (3 companies)...")
t0 = time.time()
result2 = run_query(
    query=query,
    companies=["Aker BP", "Shell", "Eni"],
    esrs_standard="",
    retriever=retriever,
    config=cfg,
    use_esrs_context=esrs_helps,
)
log(f"  Query time: {time.time() - t0:.1f}s")
for co in result2["companies"]:
    d = result2["per_company"][co]
    log(f"  {co}: level={d.get('compliance_level', '?')}")

log("ALL TESTS PASSED")
