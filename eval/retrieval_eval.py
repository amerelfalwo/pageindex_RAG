#!/usr/bin/env python3
"""Judge-free retrieval evaluation + offline parameter search for the Vectorless RAG pipeline.

No LLM is called. The pipeline's deterministic stages (BM25, weighted RRF + nucleus cutoff,
paragraph pruning, intent routing) are imported from ../app and replayed offline against the
indexed tree of the benchmark paper. The only non-deterministic stage, the LLM tree search, is
*replayed* from the node ids the live server logged (`[HYBRID-RETRIEVAL] llm_nodes=...`) during
`run_eval.py run`.

Metrics (per answerable question, against verbatim gold evidence quotes in blind_draft.json):
  evidence_recall   share of gold quotes that appear in the final context handed to the generator
  full_hit_rate     share of questions where *all* gold quotes are in the context
  page_recall       share of gold pages present in the context's page markers
  ctx_chars         context size (token-cost proxy);  density = gold chars / ctx chars
  index_coverage    share of gold quotes present in ANY node text (ceiling set by ingestion)

Usage:  uv run python eval/retrieval_eval.py [--refresh-tree] [--no-grid]
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
import statistics as st
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

# keep the app's logger from attaching a file handler (it would append to ../app.log)
logging.getLogger("VectorlessRAG").addHandler(logging.NullHandler())
sys.path.insert(0, str(ROOT / "app"))
from core.bm25 import BM25NodeIndex  # noqa: E402
from core.intent_classifier import intent_classifier  # noqa: E402
from core.rag_pipeline import VectorlessRAG  # noqa: E402

logging.disable(logging.CRITICAL)

BASE = "http://localhost:7860"
CLIENT = "rag-eval-blind"
DOC_NAME = "ieee_roadmap.pdf"
MARK = re.compile(r"---\s*\[PAGE\s+(\d+)\]\s*---", re.I)

# Parameters of the pipeline as shipped (rag_pipeline.py / bm25.py)
BASELINE = dict(k1=1.5, b=0.75, text_chars=3000, tw=3, sw=2, bm_k=5,
                w_llm=0.6, top_k=3, top_p=0.75, min_p=0.18)


# ----------------------------------------------------------------------------- data
def load_tree(refresh: bool) -> list:
    p = OUT / "tree.json"
    if refresh or not p.exists():
        doc_id = json.loads((OUT / "doc.json").read_text())["doc_id"]
        req = urllib.request.Request(f"{BASE}/documents/{doc_id}/tree", headers={"X-Client-ID": CLIENT})
        p.write_text(urllib.request.urlopen(req, timeout=30).read().decode())
    return json.loads(p.read_text())["tree"]


def flatten(tree: list) -> list[dict]:
    out: list[dict] = []

    def walk(ns):
        for n in ns:
            if isinstance(n, dict):
                if n.get("node_id"):
                    out.append(n)
                walk(n.get("nodes") or [])

    walk(tree)
    return out


def parse_llm_nodes(bench: list[dict]) -> dict[str, list[str]]:
    """Map benchmark id -> node ids returned by the live LLM tree search (from app.log)."""
    log = (ROOT / "app.log").read_text(errors="ignore").splitlines()
    qi = re.compile(r"\[QUERY-INTENT\] query='(.*?)' intent=")
    hy = re.compile(r"\[HYBRID-RETRIEVAL\] doc='%s' llm_nodes=\[(.*?)\] bm25_nodes=\[(.*?)\] fused=\[(.*?)\]" % re.escape(DOC_NAME))
    pairs: dict[str, tuple[list, list, list]] = {}
    pending = None
    for line in log:
        m = qi.search(line)
        if m:
            pending = m.group(1)
            continue
        m = hy.search(line)
        if m and pending is not None:
            ids = lambda s: re.findall(r"'(\w+)'", s)  # noqa: E731
            pairs[pending] = (ids(m.group(1)), ids(m.group(2)), ids(m.group(3)))  # last occurrence wins
            pending = None
    res, logged = {}, {}
    for b in bench:
        key = b["question"][:50]
        if key in pairs:
            res[b["id"]] = pairs[key][0]
            logged[b["id"]] = pairs[key]
    return res, logged


# ----------------------------------------------------------------------- text match
def norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def quote_in(quote: str, ctx_norm: str) -> bool:
    q = norm(quote)
    if q in ctx_norm:
        return True
    toks = q.split()
    grams = [" ".join(toks[i:i + 4]) for i in range(max(1, len(toks) - 3))]
    return sum(g in ctx_norm for g in grams) / len(grams) >= 0.9


# ------------------------------------------------------------------------- pipeline
class TunableBM25(BM25NodeIndex):
    """Same Okapi BM25 as the app but with the field boosts / text window exposed."""

    def __init__(self, nodes, k1, b, text_chars, tw, sw):
        self._cfg = (text_chars, tw, sw)
        super().__init__(nodes, k1=k1, b=b)

    def _build_index(self, nodes):
        from collections import Counter
        text_chars, tw, sw = self._cfg
        self.node_ids, self.tokenized_corpus, self.doc_lens, self.doc_freqs = [], [], [], Counter()
        for node in nodes:
            nid = node.get("node_id")
            if not nid:
                continue
            text = node.get("text") or node.get("content") or ""
            text = text if text_chars is None else text[:text_chars]
            toks = (self.tokenize(node.get("title") or "") * tw + self.tokenize(node.get("summary") or "") * sw
                    + self.tokenize(text)) or ["document"]
            self.node_ids.append(nid)
            self.tokenized_corpus.append(toks)
            self.doc_lens.append(len(toks))
            self.doc_freqs.update(set(toks))
        self.N = len(self.node_ids)
        self.avgdl = sum(self.doc_lens) / self.N if self.N else 1.0


class Sim:
    def __init__(self, tree: list):
        self.tree = tree
        self.nodes = flatten(tree)
        self.map = {n["node_id"]: n for n in self.nodes}
        self._idx: dict[tuple, TunableBM25] = {}

    def bm25(self, query: str, k1, b, text_chars, tw, sw, bm_k):
        key = (k1, b, text_chars, tw, sw)
        if key not in self._idx:
            self._idx[key] = TunableBM25(self.nodes, k1, b, text_chars, tw, sw)
        return [nid for nid, sc in self._idx[key].search(query, top_k=bm_k) if sc > 0]

    def context(self, query: str, llm_ids: list[str], cfg: dict, prune: bool = True) -> list[dict]:
        intent = intent_classifier.classify(query, doc_count=1)
        is_global = intent["is_global_or_summary"]
        ql = query.lower()
        multi = ("multi" in ql or "between" in ql or "connect" in ql
                 or ("and" in ql and ("relat" in ql or "compar" in ql or "synthes" in ql)))
        bm = self.bm25(query, cfg["k1"], cfg["b"], cfg["text_chars"], cfg["tw"], cfg["sw"], cfg["bm_k"])
        w = cfg["w_llm"]
        fused = VectorlessRAG.reciprocal_rank_fusion(
            ranked_lists=[(llm_ids, w), (bm, 1.0 - w)], top_k=cfg["top_k"],
            top_p=cfg["top_p"], min_p=cfg["min_p"], is_multi_hop=multi or is_global)
        nodes = [dict(self.map[nid]) for nid, _ in fused if nid in self.map]
        injected: list[dict] = []
        if is_global:
            d = VectorlessRAG._create_fallback_dossier_from_tree(self.tree, DOC_NAME)
            injected.append(VectorlessRAG._create_dossier_node(d, DOC_NAME, "doc"))
            injected += VectorlessRAG._extract_root_chapter_nodes(self.tree, DOC_NAME, "doc")
        ids = {n["node_id"] for n in injected}
        allnodes = injected + [n for n in nodes if n["node_id"] not in ids]
        return VectorlessRAG.prune_sub_nodes(query, allnodes, deep_analysis=False) if prune else allnodes


def score_item(item: dict, ctx_nodes: list[dict]) -> dict:
    text = "\n".join((n.get("text") or n.get("content") or "") for n in ctx_nodes)
    cn = norm(text)
    pages = set()
    for n in ctx_nodes:
        pages |= {int(p) for p in (n.get("active_pages") or []) if str(p).isdigit()}
        pages |= {int(x) for x in MARK.findall(n.get("text") or "")}
    found = [quote_in(e["quote"], cn) for e in item["evidence"]]
    gt = set(item["ground_truth_pages"])
    gold_chars = sum(len(e["quote"]) for e in item["evidence"])
    return {
        "evidence_recall": sum(found) / len(found),
        "full_hit": all(found),
        "page_recall": len(gt & pages) / len(gt),
        "ctx_chars": len(text),
        "density": gold_chars / max(len(text), 1),
        "nodes": [n["node_id"] for n in ctx_nodes if not n.get("is_dossier")],
    }


def evaluate(sim: Sim, items, llm, cfg, prune=True) -> list[dict]:
    return [dict(id=it["id"], category=it["category"], **score_item(
        it, sim.context(it["question"], llm.get(it["id"], []), cfg, prune))) for it in items]


def summarize(rows: list[dict]) -> dict:
    m = lambda k: round(st.mean(r[k] for r in rows), 4)  # noqa: E731
    return {"n": len(rows), "evidence_recall": m("evidence_recall"), "full_hit_rate": m("full_hit"),
            "page_recall": m("page_recall"), "ctx_chars": round(st.mean(r["ctx_chars"] for r in rows)),
            "density": m("density")}


# ------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-tree", action="store_true")
    ap.add_argument("--no-grid", action="store_true")
    args = ap.parse_args()

    bench = json.loads((HERE / "data" / "blind_draft.json").read_text())
    items = [b for b in bench if b["answerable"]]
    sim = Sim(load_tree(args.refresh_tree))
    llm, logged = parse_llm_nodes(bench)
    items = [b for b in items if b["id"] in llm]
    print(f"replaying {len(items)} answerable questions with logged LLM tree-search results")

    # ---- harness validity: do offline BM25 / fusion reproduce the live server's logged nodes?
    agree_bm = agree_fused = 0
    for b in items:
        l_ids, l_bm, l_fused = logged[b["id"]]
        bm = sim.bm25(b["question"], 1.5, 0.75, 3000, 3, 2, 5)
        fused = [n for n, _ in VectorlessRAG.reciprocal_rank_fusion(
            ranked_lists=[(l_ids, 0.6), (bm, 0.4)], top_k=3, top_p=0.75, min_p=0.18, is_multi_hop=False)]
        agree_bm += bm == l_bm
        agree_fused += fused == l_fused or set(fused) == set(l_fused)
    print(f"harness check: BM25 identical to live log on {agree_bm}/{len(items)}; "
          f"fused set identical on {agree_fused}/{len(items)} (multi-hop flag may differ)")

    # ---- tree-search health & ingestion ceiling
    empty = sum(1 for b in items if not llm[b["id"]])
    all_text = norm(" ".join((n.get("text") or "") for n in sim.nodes))
    cov = [quote_in(e["quote"], all_text) for b in items for e in b["evidence"]]
    print(f"LLM tree search returned NO nodes for {empty}/{len(items)} questions")
    print(f"index coverage (gold quote present in some node text): {sum(cov)}/{len(cov)} = {sum(cov)/len(cov):.1%}")

    # ---- baseline + ablations
    report: dict = {"harness": {"bm25_agree": agree_bm, "fused_agree": agree_fused, "n": len(items)},
                    "llm_tree_search_empty": empty, "index_coverage": sum(cov) / len(cov)}
    variants = {
        "baseline (shipped)": BASELINE,
        "BM25 only": {**BASELINE, "w_llm": 0.0},
        "LLM tree search only": {**BASELINE, "w_llm": 1.0},
    }
    print("\n== baseline & ablations ==")
    for name, cfg in variants.items():
        rows = evaluate(sim, items, llm, cfg)
        report[name] = summarize(rows)
        print(f"{name:<24}", report[name])
    pre = summarize(evaluate(sim, items, llm, BASELINE, prune=False))
    print(f"{'baseline, NO pruning':<24}", pre, "  <- ceiling lost to paragraph pruning")
    report["baseline_no_pruning"] = pre

    base_rows = evaluate(sim, items, llm, BASELINE)
    by_cat = defaultdict(list)
    for r in base_rows:
        by_cat[r["category"]].append(r)
    print("\n== baseline by category ==")
    report["baseline_by_category"] = {}
    for c, rs in sorted(by_cat.items()):
        report["baseline_by_category"][c] = summarize(rs)
        print(f"{c:<10}", report["baseline_by_category"][c])
    misses = [r for r in base_rows if r["evidence_recall"] < 1]
    print(f"\nquestions with incomplete evidence: {[ (r['id'], round(r['evidence_recall'],2)) for r in misses]}")
    report["baseline_misses"] = [(r["id"], r["evidence_recall"], r["nodes"]) for r in misses]

    if not args.no_grid:
        grid = dict(
            k1=[1.2, 1.5], b=[0.5, 0.75], text_chars=[3000, 6000, None], tw=[3, 1], sw=[2],
            bm_k=[5, 8], w_llm=[0.0, 0.4, 0.6, 0.8], top_k=[3, 4, 5, 6], top_p=[0.75, 0.9, None], min_p=[0.18])
        keys = list(grid)
        combos = list(itertools.product(*[grid[k] for k in keys]))
        print(f"\n== grid search over {len(combos)} configs (train = even positions, validate = odd positions) ==")
        train = [b for i, b in enumerate(items) if i % 2 == 0]
        valid = [b for i, b in enumerate(items) if i % 2 == 1]
        scored = []
        for vals in combos:
            cfg = dict(zip(keys, vals))
            s = summarize(evaluate(sim, train, llm, cfg))
            scored.append((s["evidence_recall"], -s["ctx_chars"], cfg, s))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top = scored[:8]
        base_tr = summarize(evaluate(sim, train, llm, BASELINE))
        base_va = summarize(evaluate(sim, valid, llm, BASELINE))
        print("baseline  train", base_tr, "\n          valid", base_va)
        report["grid"] = []
        for rank, (_, _, cfg, s) in enumerate(top, 1):
            v = summarize(evaluate(sim, valid, llm, cfg))
            diff = {k: cfg[k] for k in keys if cfg[k] != BASELINE[k]}
            print(f"#{rank} train={s['evidence_recall']:.3f}/{s['ctx_chars']}ch  valid={v['evidence_recall']:.3f}/{v['ctx_chars']}ch  changes={diff}")
            report["grid"].append({"changes": diff, "train": s, "valid": v})
        report["baseline_train"], report["baseline_valid"] = base_tr, base_va

        # one-factor-at-a-time sensitivity on ALL questions (more interpretable than the joint grid)
        print("\n== one-factor-at-a-time (all questions) ==")
        report["oat"] = {}
        for k in keys:
            line = []
            for v in grid[k]:
                s = summarize(evaluate(sim, items, llm, {**BASELINE, k: v}))
                line.append(f"{k}={v}: R={s['evidence_recall']:.3f} hit={s['full_hit_rate']:.2f} ch={s['ctx_chars']}")
                report["oat"][f"{k}={v}"] = s
            print(" | ".join(line))

    (OUT / "retrieval_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT / 'retrieval_report.json'}")


if __name__ == "__main__":
    main()
