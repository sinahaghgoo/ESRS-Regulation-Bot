#!/usr/bin/env python3
"""Build the redesigned RQ2 notebook programmatically."""
import json, textwrap

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": [],
}

def md(src):
    lines = textwrap.dedent(src).strip().split("\n")
    nb["cells"].append({
        "cell_type": "markdown", "metadata": {},
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
    })

def code(src):
    lines = textwrap.dedent(src).strip().split("\n")
    nb["cells"].append({
        "cell_type": "code", "metadata": {},
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
        "outputs": [], "execution_count": None,
    })

# ── Cell 0: Title ──────────────────────────────────────────────────────
md("""\
    # RQ2 — ESRS Domain Adaptation: Compliance-Assessment Quality

    **Research Question:** *To what extent does full-standard ESRS context injection
    improve evidence-grounded compliance assessment — comprising both compliance-level
    classification and compliance-rationale quality — relative to the same RAG pipeline
    without injected ESRS context?*

    ---

    ## Why this redesign?

    The original RQ2 evaluated ESRS context injection using token-overlap metrics
    (Token-F1 between generated and gold rationales) alongside classification accuracy.
    Token-overlap is a **proxy** that measures lexical similarity, not substantive
    agreement.  Two rationales can share many tokens yet contradict each other on the
    core fulfillment judgment, or use entirely different words yet agree perfectly.

    The redesigned RQ2 directly measures what the thesis claims:

    1. **Deterministic compliance-level metrics** — correct Low / Medium / High prediction
    2. **LLM-judge rationale evaluation** — substantive alignment with gold rationale
       (G-Eval style, Claude Sonnet 4.5 judge)
    3. **Joint compliance-assessment metric** — *primary RQ2 endpoint* requiring both
       correct level AND substantively aligned rationale
""")

# ── Cell 1: Setup header ──────────────────────────────────────────────
md("## 1. Setup and Imports")

# ── Cell 2: Imports ───────────────────────────────────────────────────
code("""\
    import sys, os, json, re, hashlib, warnings, logging, math
    from pathlib import Path
    from collections import Counter, defaultdict
    from dataclasses import dataclass, asdict, field
    from statistics import mean as _mean, stdev as _stdev
    from typing import Any

    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    sys.path.insert(0, str(PROJECT_ROOT))
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)

    # Load .env robustly (comment-tolerant)
    with open(PROJECT_ROOT / ".env") as _f:
        for _line in _f:
            _line = _line.strip()
            if "=" in _line and not _line.startswith("#") and not _line.startswith("="):
                _k, _, _v = _line.partition("=")
                os.environ[_k.strip()] = _v.strip()

    from rag.config.settings import (
        GOLDEN_STANDARD_PATH, RQ2_RESULTS_DIR, ESRS_JSON_DIR,
        BEST_PIPELINE_PATH, RQ2_RESULTS_VERSION,
    )

    CONDITIONS = ["full_system", "no_esrs_context"]
    NB_CACHE_DIR = RQ2_RESULTS_DIR / "nb_cache"
    LABELS = ["Low", "Medium", "High"]
    LABEL_ORD = {"Low": 0, "Medium": 1, "High": 2}

    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 120, "savefig.dpi": 150})

    print(f"Project root : {PROJECT_ROOT}")
    print(f"RQ2 version  : {RQ2_RESULTS_VERSION}")
""")

# ── Cell 3 ────────────────────────────────────────────────────────────
md("""\
    ## 2. Load Best RQ1 Pipeline Configuration

    Both conditions share the identical retrieval pipeline from RQ1. The only
    experimental variable is whether ESRS regulatory context is injected.
""")

code("""\
    best_config = {}
    with open(BEST_PIPELINE_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, _, v = line.partition(":")
                best_config[k.strip()] = v.strip()

    pipeline_name = best_config.pop("_best_pipeline_name", "unknown")
    selection_reason = best_config.pop("_selection_reason", "")
    print(f"Best RQ1 pipeline : {pipeline_name}")
    print(f"Selection reason  : {selection_reason}")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
""")

# ── Cell 5 ────────────────────────────────────────────────────────────
md("## 3. Load Golden Standard and Generation Cache")

code("""\
    raw_golden = json.loads(GOLDEN_STANDARD_PATH.read_text(encoding="utf-8"))

    golden_lookup = {}
    samples = []
    for q in raw_golden["questions"]:
        for pc in q["per_company"]:
            key = (q["id"], pc["company"])
            entry = dict(
                question_id=q["id"], question=q["question"],
                esrs_standard=q.get("esrs_standard", ""),
                esrs_disclosure_id=q.get("esrs_disclosure_id", ""),
                company=pc["company"],
                ground_truth_answer=pc.get("ground_truth_answer", ""),
                fulfillment_level=pc.get("fulfillment_level", ""),
                fulfillment_rationale=pc.get("fulfillment_rationale", ""),
                evidence=pc.get("evidence", []),
            )
            golden_lookup[key] = entry
            samples.append(entry)

    n_total = len(samples)
    level_dist = Counter(s["fulfillment_level"] for s in samples)
    print(f"Golden-standard samples : {n_total}")
    print(f"  Questions : {len(raw_golden['questions'])}")
    print(f"  Companies : {len(set(s['company'] for s in samples))}")
    print(f"  Standards : {len(set(s['esrs_standard'] for s in samples))}")
    print(f"  Labels    : {dict(level_dist)}")
""")

code("""\
    # Verify cached generation results exist
    for condition in CONDITIONS:
        cdir = NB_CACHE_DIR / condition
        n = len(list(cdir.glob("*.json"))) if cdir.exists() else 0
        print(f"{condition}: {n} cached generation files")
        if n == 0:
            raise RuntimeError(f"No cache for {condition}. Run generation first.")
""")

# ── Cell 8: Build paired table ────────────────────────────────────────
md("""\
    ## 4. Build Paired Sample Table

    For each of the 204 golden-standard samples, match the cached generation
    result from each condition with its gold reference.
""")

code("""\
    def _cache_key(question, company, condition, esrs_context_len):
        blob = f"{question}|{company}|{condition}|{esrs_context_len}"
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    # ESRS context cache → lengths
    ctx_cache_path = RQ2_RESULTS_DIR / "context_cache.json"
    ctx_cache = json.loads(ctx_cache_path.read_text("utf-8")) if ctx_cache_path.exists() else {}

    def _ctx_len(qid):
        for v in ctx_cache.values():
            if isinstance(v, dict) and v.get("question_id") == qid:
                return len(v.get("context_text", ""))
        return 0

    # Index cache files by their stem (hash)
    cache_idx = {}
    for cond in CONDITIONS:
        cache_idx[cond] = {}
        for cf in (NB_CACHE_DIR / cond).glob("*.json"):
            cache_idx[cond][cf.stem] = json.loads(cf.read_text("utf-8"))

    @dataclass
    class Paired:
        question_id: str; question: str; esrs_standard: str
        esrs_disclosure_id: str; company: str
        gold_level: str; gold_rationale: str
        pred_level_full: str; pred_rationale_full: str
        pred_level_no_esrs: str; pred_rationale_no_esrs: str

    def _norm_level(v):
        if v is None: return ""
        return {"low": "Low", "medium": "Medium", "high": "High"}.get(
            str(v).strip().lower(), "")

    paired = []
    missing = []
    for s in samples:
        ctx_l = _ctx_len(s["question_id"])
        fk = _cache_key(s["question"], s["company"], "with_esrs_context", ctx_l)
        nk = _cache_key(s["question"], s["company"], "without_esrs_context", 0)
        fd = cache_idx["full_system"].get(fk)
        nd = cache_idx["no_esrs_context"].get(nk)
        if fd is None or nd is None:
            missing.append((s["question_id"], s["company"], fd is not None, nd is not None))
            continue
        paired.append(Paired(
            question_id=s["question_id"], question=s["question"],
            esrs_standard=s["esrs_standard"],
            esrs_disclosure_id=s["esrs_disclosure_id"],
            company=s["company"],
            gold_level=s["fulfillment_level"],
            gold_rationale=s["fulfillment_rationale"],
            pred_level_full=_norm_level(fd.get("compliance_level")),
            pred_rationale_full=fd.get("rationale", ""),
            pred_level_no_esrs=_norm_level(nd.get("compliance_level")),
            pred_rationale_no_esrs=nd.get("rationale", ""),
        ))

    print(f"Paired samples : {len(paired)} / {n_total}")
    if missing:
        print(f"Missing (cache key mismatch): {len(missing)}")
        for qid, co, hf, hn in missing[:5]:
            print(f"  {qid} / {co}  full={hf}  no_esrs={hn}")
""")

# ── Deterministic metrics ─────────────────────────────────────────────
md("""\
    ## 5. Deterministic Compliance-Level Metrics

    Evaluate the first assessment component: does the system predict the
    correct Low / Medium / High fulfillment level?

    All metrics are implemented without `sklearn` for zero-dependency
    reproducibility.
""")

code("""\
    # ── sklearn-free metric implementations ──────────────────────────
    def _cm3(gold, pred):
        cm = [[0]*3 for _ in range(3)]
        for g, p in zip(gold, pred):
            gi = LABELS.index(g) if g in LABELS else -1
            pi = LABELS.index(p) if p in LABELS else -1
            if gi >= 0 and pi >= 0: cm[gi][pi] += 1
        return cm

    def exact_accuracy(g, p):
        return sum(a == b for a, b in zip(g, p)) / len(g) if g else 0.0

    def macro_f1(g, p):
        cm = _cm3(g, p); f1s = []
        for i in range(3):
            tp = cm[i][i]
            fp = sum(cm[j][i] for j in range(3)) - tp
            fn = sum(cm[i][j] for j in range(3)) - tp
            pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
            f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0)
        return _mean(f1s)

    def balanced_accuracy(g, p):
        cm = _cm3(g, p)
        recs = [cm[i][i]/sum(cm[i]) if sum(cm[i]) else 0 for i in range(3)]
        return _mean([r for r in recs if sum(cm[recs.index(r)]) > 0]) if recs else 0.0

    def weighted_kappa(g, p):
        n = len(g); k = 3
        if n == 0: return 0.0
        O = _cm3(g, p)
        rs = [sum(O[i]) for i in range(k)]
        cs = [sum(O[j][i] for j in range(k)) for i in range(k)]
        E = [[rs[i]*cs[j]/n for j in range(k)] for i in range(k)]
        W = [[abs(i-j)/(k-1) for j in range(k)] for i in range(k)]
        num = sum(W[i][j]*O[i][j] for i in range(k) for j in range(k))
        den = sum(W[i][j]*E[i][j] for i in range(k) for j in range(k))
        return 1 - num/den if den else (1.0 if num == 0 else 0.0)

    def ordinal_err(gl, pl):
        go = LABEL_ORD.get(gl); po = LABEL_ORD.get(pl)
        return abs(go - po) if go is not None and po is not None else 2

    def off_by_one(g, p):
        return sum(ordinal_err(a, b) <= 1 for a, b in zip(g, p)) / len(g) if g else 0.0

    def mean_ord_err(g, p):
        return _mean([ordinal_err(a, b) for a, b in zip(g, p)]) if g else 0.0

    def _level_metrics(ps, cond):
        g = [s.gold_level for s in ps]
        p = [s.pred_level_full if cond == "full_system" else s.pred_level_no_esrs for s in ps]
        v = [(a, b) for a, b in zip(g, p) if a in LABELS and b in LABELS]
        gv, pv = [x[0] for x in v], [x[1] for x in v]
        return dict(n_scored=len(v), n_invalid=len(g)-len(v),
            exact_accuracy=exact_accuracy(gv, pv), macro_f1=macro_f1(gv, pv),
            balanced_accuracy=balanced_accuracy(gv, pv), weighted_kappa=weighted_kappa(gv, pv),
            off_by_one=off_by_one(gv, pv), mean_ord_err=mean_ord_err(gv, pv),
            confusion_matrix=_cm3(gv, pv))

    lm = {c: _level_metrics(paired, c) for c in CONDITIONS}
    for c in CONDITIONS:
        m = lm[c]
        print(f"\\n{c}  (scored {m['n_scored']}, invalid {m['n_invalid']})")
        for k in ["exact_accuracy","macro_f1","balanced_accuracy","weighted_kappa","off_by_one","mean_ord_err"]:
            print(f"  {k:22s}: {m[k]:.3f}")

    print("\\n--- Deltas (full − no_esrs) ---")
    for k in ["exact_accuracy","macro_f1","balanced_accuracy","weighted_kappa"]:
        print(f"  Δ {k}: {lm['full_system'][k] - lm['no_esrs_context'][k]:+.3f}")
""")

# ── Majority baseline ─────────────────────────────────────────────────
md("## 6. Majority-Class Baseline")

code("""\
    all_gold = [s.gold_level for s in paired if s.gold_level in LABELS]
    maj = max(LABELS, key=all_gold.count)
    maj_pred = [maj] * len(all_gold)
    print(f"Majority label: {maj}  ({all_gold.count(maj)}/{len(all_gold)})")
    print(f"  exact_accuracy : {exact_accuracy(all_gold, maj_pred):.3f}")
    print(f"  macro_f1       : {macro_f1(all_gold, maj_pred):.3f}")
    print(f"  balanced_acc   : {balanced_accuracy(all_gold, maj_pred):.3f}")
    for c in CONDITIONS:
        d = lm[c]["macro_f1"] - macro_f1(all_gold, maj_pred)
        print(f"  {c} macro-F1 above majority: {d:+.3f}")
""")

# ── Judge ─────────────────────────────────────────────────────────────
md("""\
    ## 7. LLM-Judge Rationale Evaluation

    The second assessment component: does the generated rationale express the
    same **substantive** fulfillment judgment as the gold rationale?

    We use Claude Sonnet 4.5 as a G-Eval-style judge.  The judge ignores
    style / wording and focuses on substantive agreement, penalising
    contradiction, material omission, and support for a different level.
""")

code("""\
    from rag.evaluation.rq2_judge import (
        JudgeInput, JudgeOutput, judge_sample, run_judge_batch,
        get_judge_version, JUDGE_CACHE_DIR,
    )
    print(f"Judge version  : {get_judge_version()}")
    n_cached = len(list(JUDGE_CACHE_DIR.glob('*.json'))) if JUDGE_CACHE_DIR.exists() else 0
    print(f"Existing cache : {n_cached} files")
""")

code("""\
    # Build judge batches
    judge_batches = {}
    for cond in CONDITIONS:
        batch = []
        for s in paired:
            gl, gr = s.pred_level_full, s.pred_rationale_full
            if cond == "no_esrs_context":
                gl, gr = s.pred_level_no_esrs, s.pred_rationale_no_esrs
            batch.append(dict(
                question=s.question, esrs_standard=s.esrs_standard,
                esrs_disclosure_id=s.esrs_disclosure_id,
                gold_fulfillment_level=s.gold_level,
                gold_fulfillment_rationale=s.gold_rationale,
                generated_compliance_level=gl,
                generated_compliance_rationale=gr,
                company=s.company,
            ))
        judge_batches[cond] = batch
        print(f"{cond}: {len(batch)} samples")
    print(f"Total judge evaluations: {sum(len(b) for b in judge_batches.values())}")
""")

code("""\
    # Run judge (cached results reused automatically)
    judge_results = {}
    for cond in CONDITIONS:
        print(f"\\nJudging {cond} …")
        res = run_judge_batch(judge_batches[cond], cond, dry_run=False)
        judge_results[cond] = res
        nc = sum(1 for r in res if r.get("was_cached"))
        print(f"  done — {len(res)} total, {nc} cached, {len(res)-nc} API")
        vc = Counter(r.get("rationale_verdict","?") for r in res)
        print(f"  verdicts: {dict(vc)}")
""")

# ── Judge aggregation ─────────────────────────────────────────────────
md("## 8. Aggregate Judge Metrics")

code("""\
    def agg_judge(res):
        vs = [r["rationale_verdict"] for r in res]
        ss = [r["rationale_score"] for r in res]
        n = len(res)
        return dict(
            n=n,
            mean_score=_mean(ss) if ss else 0,
            std_score=_stdev(ss) if len(ss)>1 else 0,
            aligned=sum(v=="aligned" for v in vs)/n if n else 0,
            partial=sum(v=="partially_aligned" for v in vs)/n if n else 0,
            misaligned=sum(v=="misaligned" for v in vs)/n if n else 0,
            aligned_or_partial=sum(v in ("aligned","partially_aligned") for v in vs)/n if n else 0,
            error_tags=dict(Counter(t for r in res for t in r.get("error_tags",[]) if t!="none")),
        )

    ja = {c: agg_judge(judge_results[c]) for c in CONDITIONS}
    for c in CONDITIONS:
        m = ja[c]
        print(f"\\n{c}")
        print(f"  mean_score        : {m['mean_score']:.3f} ± {m['std_score']:.3f}")
        print(f"  aligned           : {m['aligned']:.3f}")
        print(f"  partially_aligned : {m['partial']:.3f}")
        print(f"  misaligned        : {m['misaligned']:.3f}")
        print(f"  aligned+partial   : {m['aligned_or_partial']:.3f}")
        if m["error_tags"]:
            print(f"  error_tags: {m['error_tags']}")

    print("\\n--- Deltas ---")
    for k in ["mean_score","aligned","misaligned"]:
        print(f"  Δ {k}: {ja['full_system'][k] - ja['no_esrs_context'][k]:+.3f}")
""")

# ── Joint metrics ─────────────────────────────────────────────────────
md("""\
    ## 9. Joint Compliance-Assessment Metrics — PRIMARY RQ2 ENDPOINT

    The joint metric requires **both** correct level AND substantively aligned
    rationale.

    - **Joint strict**: level correct AND verdict == "aligned"
    - **Joint lenient**: level correct AND verdict ∈ {aligned, partially_aligned}
      ← **primary endpoint**
    - **Joint score**: 0.5 × level_correct + 0.5 × rationale_score
""")

code("""\
    def joint(ps, jr, cond):
        js, jl, scores = 0, 0, []
        for s, r in zip(ps, jr):
            pl = s.pred_level_full if cond == "full_system" else s.pred_level_no_esrs
            lm_ = (pl == s.gold_level)
            v = r.get("rationale_verdict", "misaligned")
            rs = r.get("rationale_score", 0.0)
            if lm_ and v == "aligned": js += 1
            if lm_ and v in ("aligned","partially_aligned"): jl += 1
            scores.append(0.5 * (1.0 if lm_ else 0.0) + 0.5 * rs)
        n = len(ps)
        return dict(strict=js/n, lenient=jl/n,
                    score=_mean(scores), score_std=_stdev(scores) if len(scores)>1 else 0,
                    n_strict=js, n_lenient=jl)

    jm = {c: joint(paired, judge_results[c], c) for c in CONDITIONS}
    for c in CONDITIONS:
        m = jm[c]
        print(f"\\n{c}")
        print(f"  strict  : {m['strict']:.3f}  ({m['n_strict']}/{len(paired)})")
        print(f"  lenient : {m['lenient']:.3f}  ({m['n_lenient']}/{len(paired)})  ← PRIMARY")
        print(f"  score   : {m['score']:.3f} ± {m['score_std']:.3f}")

    delta_primary = jm["full_system"]["lenient"] - jm["no_esrs_context"]["lenient"]
    print(f"\\n{'='*60}")
    print(f"PRIMARY RQ2 ENDPOINT — Joint Lenient Match Rate")
    print(f"  Full system     : {jm['full_system']['lenient']:.3f}")
    print(f"  No ESRS context : {jm['no_esrs_context']['lenient']:.3f}")
    print(f"  Δ               : {delta_primary:+.3f}")
    print(f"{'='*60}")
""")

# ── Visualisation ─────────────────────────────────────────────────────
md("## 10. Aggregate Comparison Visualisation")

code("""\
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    w = 0.35

    # Panel 1 — deterministic level metrics
    ax = axes[0]
    dk = ["exact_accuracy","macro_f1","balanced_accuracy","weighted_kappa"]
    dl = ["Exact\\nAcc","Macro\\nF1","Bal.\\nAcc","Wtd\\nKappa"]
    fv = [lm["full_system"][k] for k in dk]
    nv = [lm["no_esrs_context"][k] for k in dk]
    x = np.arange(len(dk))
    ax.bar(x-w/2, fv, w, label="Full System", color="#2D6A4F")
    ax.bar(x+w/2, nv, w, label="No ESRS Ctx", color="#E76F51")
    ax.set_xticks(x); ax.set_xticklabels(dl)
    ax.set_ylim(0, 1.05); ax.set_title("Deterministic Level Metrics")
    ax.legend(fontsize=9); ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Panel 2 — judge rationale
    ax = axes[1]
    rk = ["mean_score","aligned","aligned_or_partial"]
    rl = ["Mean\\nScore","Aligned\\nRate","Aligned+\\nPartial"]
    fv = [ja["full_system"][k] for k in rk]
    nv = [ja["no_esrs_context"][k] for k in rk]
    x = np.arange(len(rk))
    ax.bar(x-w/2, fv, w, label="Full System", color="#2D6A4F")
    ax.bar(x+w/2, nv, w, label="No ESRS Ctx", color="#E76F51")
    ax.set_xticks(x); ax.set_xticklabels(rl)
    ax.set_ylim(0, 1.05); ax.set_title("LLM-Judge Rationale Metrics")
    ax.legend(fontsize=9); ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Panel 3 — joint (primary)
    ax = axes[2]
    jk = ["strict","lenient","score"]
    jl = ["Joint\\nStrict","Joint Lenient\\n(PRIMARY)","Joint\\nScore"]
    fv = [jm["full_system"][k] for k in jk]
    nv = [jm["no_esrs_context"][k] for k in jk]
    x = np.arange(len(jk))
    ax.bar(x-w/2, fv, w, label="Full System", color="#2D6A4F")
    ax.bar(x+w/2, nv, w, label="No ESRS Ctx", color="#E76F51")
    ax.set_xticks(x); ax.set_xticklabels(jl)
    ax.set_ylim(0, 1.05); ax.set_title("Joint Compliance-Assessment Metrics")
    ax.legend(fontsize=9); ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()
    plt.savefig(RQ2_RESULTS_DIR / "rq2_aggregate_comparison.png", bbox_inches="tight")
    plt.show()
    print("Saved rq2_aggregate_comparison.png")
""")

# ── Confusion matrices ────────────────────────────────────────────────
md("## 11. Confusion Matrices")

code("""\
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cond in zip(axes, CONDITIONS):
        cm = np.array(lm[cond]["confusion_matrix"])
        rs = cm.sum(axis=1, keepdims=True)
        pct = np.divide(cm, rs, where=rs>0, out=np.zeros_like(cm, dtype=float))*100
        ax.imshow(pct, cmap="YlGn", vmin=0, vmax=100)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Gold")
        title = "Full System" if cond == "full_system" else "No ESRS Context"
        ax.set_title(f"Confusion Matrix — {title}")
        for i in range(3):
            for j in range(3):
                clr = "white" if pct[i][j] > 60 else "black"
                ax.text(j, i, f"{cm[i][j]}\\n({pct[i][j]:.0f}%)",
                        ha="center", va="center", fontsize=10, color=clr)
    plt.tight_layout()
    plt.savefig(RQ2_RESULTS_DIR / "rq2_confusion_matrices.png", bbox_inches="tight")
    plt.show()
""")

# ── Per-standard ──────────────────────────────────────────────────────
md("""\
    ## 12. Per-ESRS-Standard Breakdown

    Which of the 12 standards benefit most from explicit ESRS context injection?
""")

code("""\
    all_stds = sorted(set(s.esrs_standard for s in paired))
    rows = []
    for std in all_stds:
        idx = [i for i, s in enumerate(paired) if s.esrs_standard == std]
        sp = [paired[i] for i in idx]
        for cond in CONDITIONS:
            g = [s.gold_level for s in sp]
            p = [s.pred_level_full if cond=="full_system" else s.pred_level_no_esrs for s in sp]
            v = [(a,b) for a,b in zip(g,p) if a in LABELS and b in LABELS]
            gv,pv = [x[0] for x in v],[x[1] for x in v]
            ea = exact_accuracy(gv,pv)
            sj = [judge_results[cond][i] for i in idx]
            ms = _mean([r["rationale_score"] for r in sj]) if sj else 0
            jl = 0
            for si, jr in zip(sp, sj):
                pl = si.pred_level_full if cond=="full_system" else si.pred_level_no_esrs
                if pl==si.gold_level and jr["rationale_verdict"] in ("aligned","partially_aligned"):
                    jl += 1
            rows.append(dict(Standard=std,
                Condition="Full" if cond=="full_system" else "No ESRS",
                N=len(sp), ExactAcc=ea, RatScore=ms,
                JointLenient=jl/len(sp) if sp else 0))

    df_std = pd.DataFrame(rows)
    print(df_std.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
""")

code("""\
    # Per-standard joint-lenient delta chart
    fig, ax = plt.subplots(figsize=(14, 6))
    f_jl = {r["Standard"]: r["JointLenient"] for _, r in df_std[df_std.Condition=="Full"].iterrows()}
    n_jl = {r["Standard"]: r["JointLenient"] for _, r in df_std[df_std.Condition=="No ESRS"].iterrows()}
    deltas = [(s, f_jl.get(s,0)-n_jl.get(s,0)) for s in all_stds]
    deltas.sort(key=lambda x: x[1], reverse=True)
    stds_s = [d[0] for d in deltas]; ds = [d[1] for d in deltas]
    colors = ["#2D6A4F" if d>0 else ("#E76F51" if d<0 else "#999") for d in ds]
    y = np.arange(len(stds_s))
    ax.barh(y, ds, color=colors, height=0.7)
    ax.set_yticks(y); ax.set_yticklabels(stds_s)
    ax.set_xlabel("Δ Joint Lenient (Full − No ESRS)")
    ax.set_title("Per-Standard Impact of ESRS Context on Compliance Assessment")
    ax.axvline(0, color="black", lw=0.8)
    for i,(d,s) in enumerate(zip(ds,stds_s)):
        ax.text(d+(0.005 if d>=0 else -0.005), i, f"{d:+.2f}",
                va="center", ha="left" if d>=0 else "right", fontsize=9)
    plt.tight_layout()
    plt.savefig(RQ2_RESULTS_DIR / "rq2_per_standard.png", bbox_inches="tight")
    plt.show()
""")

# ── Per-company ───────────────────────────────────────────────────────
md("## 13. Per-Company Analysis")

code("""\
    all_cos = sorted(set(s.company for s in paired))
    co_rows = []
    for co in all_cos:
        idx = [i for i,s in enumerate(paired) if s.company==co]
        sp = [paired[i] for i in idx]
        for cond in CONDITIONS:
            g = [s.gold_level for s in sp]
            p = [s.pred_level_full if cond=="full_system" else s.pred_level_no_esrs for s in sp]
            v = [(a,b) for a,b in zip(g,p) if a in LABELS and b in LABELS]
            gv,pv = [x[0] for x in v],[x[1] for x in v]
            sj = [judge_results[cond][i] for i in idx]
            jl = sum(1 for si,jr in zip(sp,sj)
                     if (si.pred_level_full if cond=="full_system" else si.pred_level_no_esrs)==si.gold_level
                     and jr["rationale_verdict"] in ("aligned","partially_aligned"))
            co_rows.append(dict(Company=co,
                Condition="Full" if cond=="full_system" else "No ESRS",
                ExactAcc=exact_accuracy(gv,pv),
                JointLenient=jl/len(sp) if sp else 0))

    df_co = pd.DataFrame(co_rows)
    piv = df_co.pivot(index="Company", columns="Condition", values="JointLenient")
    piv["Delta"] = piv.get("Full",0) - piv.get("No ESRS",0)
    piv = piv.sort_values("Delta", ascending=False)
    print(piv.to_string(float_format=lambda x: f"{x:.3f}"))
""")

code("""\
    fig, ax = plt.subplots(figsize=(12, 8))
    cos = piv.index.tolist(); ds = piv["Delta"].tolist()
    colors = ["#2D6A4F" if d>0 else ("#E76F51" if d<0 else "#999") for d in ds]
    y = np.arange(len(cos))
    ax.barh(y, ds, color=colors, height=0.7)
    ax.set_yticks(y); ax.set_yticklabels(cos)
    ax.set_xlabel("Δ Joint Lenient Match Rate")
    ax.set_title("Per-Company Impact of ESRS Context Injection")
    ax.axvline(0, color="black", lw=0.8)
    for i,d in enumerate(ds):
        ax.text(d+(0.005 if d>=0 else -0.005), i, f"{d:+.2f}",
                va="center", ha="left" if d>=0 else "right", fontsize=9)
    plt.tight_layout()
    plt.savefig(RQ2_RESULTS_DIR / "rq2_per_company.png", bbox_inches="tight")
    plt.show()
""")

# ── Statistical inference ─────────────────────────────────────────────
md("""\
    ## 14. Statistical Inference

    Question-level paired sign test preserving the nested
    companies-within-questions structure.
""")

code("""\
    def sign_test(deltas):
        pos = sum(d>0 for d in deltas)
        neg = sum(d<0 for d in deltas)
        n = pos + neg
        if n == 0: return 1.0
        z = (pos - n/2) / (math.sqrt(n)/2)
        return 2*(1 - 0.5*(1 + math.erf(abs(z)/math.sqrt(2))))

    def cohens_d(deltas):
        if len(deltas)<2: return 0.0
        sd = _stdev(deltas)
        return _mean(deltas)/sd if sd else 0.0

    def boot_ci(vals, n_boot=10000, ci=0.95, seed=42):
        rng = np.random.RandomState(seed)
        ms = [float(np.mean(rng.choice(vals, len(vals), replace=True))) for _ in range(n_boot)]
        lo = np.percentile(ms, (1-ci)/2*100)
        hi = np.percentile(ms, (1+ci)/2*100)
        return float(lo), float(hi)

    qids = sorted(set(s.question_id for s in paired))
    q_full, q_no = {}, {}
    for qid in qids:
        idx = [i for i,s in enumerate(paired) if s.question_id==qid]
        sp = [paired[i] for i in idx]
        jf, jn = 0, 0
        for i_ in idx:
            s = paired[i_]
            jrf = judge_results["full_system"][i_]
            jrn = judge_results["no_esrs_context"][i_]
            if s.pred_level_full==s.gold_level and jrf["rationale_verdict"] in ("aligned","partially_aligned"):
                jf += 1
            if s.pred_level_no_esrs==s.gold_level and jrn["rationale_verdict"] in ("aligned","partially_aligned"):
                jn += 1
        q_full[qid] = jf/len(sp) if sp else 0
        q_no[qid] = jn/len(sp) if sp else 0

    q_deltas = [q_full[q] - q_no[q] for q in qids]
    p_val = sign_test(q_deltas)
    d_eff = cohens_d(q_deltas)
    ci_lo, ci_hi = boot_ci(q_deltas)
    abs_d = abs(d_eff)
    mag = "negligible" if abs_d<0.2 else "small" if abs_d<0.5 else "medium" if abs_d<0.8 else "large"

    print("Question-level paired analysis (Joint Lenient)")
    print(f"  N questions : {len(qids)}")
    print(f"  Mean Δ      : {_mean(q_deltas):+.3f}")
    if len(q_deltas)>1: print(f"  Std Δ       : {_stdev(q_deltas):.3f}")
    print(f"  Positive: {sum(d>0 for d in q_deltas)}, Negative: {sum(d<0 for d in q_deltas)}, Zero: {sum(d==0 for d in q_deltas)}")
    print(f"  Sign-test p : {p_val:.4f}")
    print(f"  Cohen's d   : {d_eff:.3f} ({mag})")
    print(f"  95% CI      : [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Significant : {'YES' if p_val<0.05 else 'NO'} (α = 0.05)")
""")

# ── Export ────────────────────────────────────────────────────────────
md("## 15. Per-Sample Results Export")

code("""\
    rows_out = []
    for i, s in enumerate(paired):
        jf = judge_results["full_system"][i]
        jn = judge_results["no_esrs_context"][i]
        rows_out.append(dict(
            question_id=s.question_id, company=s.company,
            esrs_standard=s.esrs_standard, gold_level=s.gold_level,
            pred_full=s.pred_level_full, pred_no_esrs=s.pred_level_no_esrs,
            level_match_full=(s.pred_level_full==s.gold_level),
            level_match_no_esrs=(s.pred_level_no_esrs==s.gold_level),
            verdict_full=jf["rationale_verdict"],
            score_full=jf["rationale_score"],
            verdict_no_esrs=jn["rationale_verdict"],
            score_no_esrs=jn["rationale_score"],
            joint_lenient_full=(s.pred_level_full==s.gold_level and
                jf["rationale_verdict"] in ("aligned","partially_aligned")),
            joint_lenient_no_esrs=(s.pred_level_no_esrs==s.gold_level and
                jn["rationale_verdict"] in ("aligned","partially_aligned")),
            error_tags_full="; ".join(jf.get("error_tags",[])),
            error_tags_no_esrs="; ".join(jn.get("error_tags",[])),
        ))
    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(RQ2_RESULTS_DIR / "rq2_per_sample_results.csv", index=False)
    print(f"Saved {len(df_out)} rows → rq2_per_sample_results.csv")

    # Biggest improvements
    df_out["Δ_score"] = df_out["score_full"] - df_out["score_no_esrs"]
    top5 = df_out.nlargest(5, "Δ_score")[
        ["question_id","company","gold_level","pred_full","pred_no_esrs",
         "verdict_full","verdict_no_esrs","Δ_score"]]
    print("\\nTop 5 samples where ESRS injection most improved rationale:")
    print(top5.to_string(index=False))
""")

# ── Save aggregate ────────────────────────────────────────────────────
md("## 16. Save Aggregate Results")

code("""\
    summary = dict(
        rq2_version=RQ2_RESULTS_VERSION,
        judge_version=get_judge_version(),
        n_paired=len(paired),
        best_pipeline=pipeline_name,
        gold_label_dist=dict(Counter(s.gold_level for s in paired)),
        majority_baseline=dict(label=maj,
            exact_accuracy=exact_accuracy(all_gold, maj_pred),
            macro_f1=macro_f1(all_gold, maj_pred)),
        level_metrics={c: {k:v for k,v in lm[c].items() if k!="confusion_matrix"} for c in CONDITIONS},
        judge_metrics=ja,
        joint_metrics=jm,
        primary_endpoint=dict(
            metric="joint_lenient_match_rate",
            full_system=jm["full_system"]["lenient"],
            no_esrs_context=jm["no_esrs_context"]["lenient"],
            delta=delta_primary),
        statistical=dict(
            test="paired_sign_test_question_level",
            n_questions=len(qids), p_value=p_val,
            cohens_d=d_eff, magnitude=mag,
            ci_95=[ci_lo, ci_hi],
            significant=p_val<0.05),
    )
    out_path = RQ2_RESULTS_DIR / "rq2_aggregate_results.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), "utf-8")
    print(f"Saved → {out_path}")
""")

# ── Interpretation ────────────────────────────────────────────────────
md("## 17. Interpretation")

code("""\
    print("=" * 70)
    print("RQ2 FINDINGS — Effect of ESRS Context Injection")
    print("=" * 70)
    print()
    print(f"PRIMARY ENDPOINT: Joint Lenient Match Rate")
    print(f"  Full system     : {jm['full_system']['lenient']:.1%}")
    print(f"  No ESRS context : {jm['no_esrs_context']['lenient']:.1%}")
    print(f"  Δ               : {delta_primary:+.1%}")
    print()
    if p_val < 0.05:
        print(f"  → SIGNIFICANT at α = 0.05  (p = {p_val:.4f}, d = {d_eff:.2f} [{mag}])")
    else:
        print(f"  → NOT significant at α = 0.05  (p = {p_val:.4f}, d = {d_eff:.2f} [{mag}])")
    print(f"  95% CI : [{ci_lo:.3f}, {ci_hi:.3f}]")
    print()
    print("COMPONENT BREAKDOWN")
    for k in ["exact_accuracy","macro_f1"]:
        d = lm["full_system"][k] - lm["no_esrs_context"][k]
        print(f"  Level {k:18s}: {lm['full_system'][k]:.3f} vs {lm['no_esrs_context'][k]:.3f} (Δ {d:+.3f})")
    for k in ["mean_score","aligned"]:
        d = ja["full_system"][k] - ja["no_esrs_context"][k]
        print(f"  Judge {k:18s}: {ja['full_system'][k]:.3f} vs {ja['no_esrs_context'][k]:.3f} (Δ {d:+.3f})")
    print()
    print("PER-STANDARD HIGHLIGHTS")
    for std in all_stds:
        fv = f_jl.get(std,0); nv_ = n_jl.get(std,0); d_ = fv - nv_
        m_ = "↑" if d_>0.05 else ("↓" if d_<-0.05 else "≈")
        print(f"  {std:10s}: {fv:.2f} vs {nv_:.2f} (Δ {d_:+.2f}) {m_}")
    print()
    print("=" * 70)
    if delta_primary > 0:
        print("CONCLUSION: ESRS context injection IMPROVES compliance assessment.")
    elif delta_primary < 0:
        print("CONCLUSION: ESRS context injection does NOT improve assessment;")
        print("parametric knowledge appears sufficient.")
    else:
        print("CONCLUSION: No measurable effect of ESRS context injection.")
    print("=" * 70)
""")

code('print("Notebook complete.")')

# ── Write ─────────────────────────────────────────────────────────────
target = "notebooks/10_rq2_esrs_context_ablation.ipynb"
with open(target, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {len(nb['cells'])} cells → {target}")
