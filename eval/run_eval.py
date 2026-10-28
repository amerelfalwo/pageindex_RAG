#!/usr/bin/env python3
"""Black-box RAG evaluation against the blind benchmark.

Stages (each checkpoints to eval/results/, so they can be re-run independently):
  upload   POST the benchmark PDF under a dedicated X-Client-ID and wait until indexed
  run      ask every benchmark question (fresh chat each) and store answers + timings
  judge    LLM-judge answers (Groq, model != answering model) + deterministic checks
  report   aggregate metrics -> eval/results/report.json and report.md
  cleanup  delete the evaluation document (and its chats are left owner-scoped)

Usage:  uv run python run_eval.py upload|run|judge|report|cleanup|all
Env:    EVAL_BASE_URL (default http://localhost:7860), EVAL_CLIENT_ID,
        EVAL_JUDGE_MODEL (default llama-3.3-70b-versatile), GROQ_API_KEY (from ../.env)
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import statistics as st
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import httpx
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
load_dotenv(HERE.parent / ".env")

BASE = os.getenv("EVAL_BASE_URL", "http://localhost:7860").rstrip("/")
CLIENT_ID = os.getenv("EVAL_CLIENT_ID", "rag-eval-blind")
HEAD = {"X-Client-ID": CLIENT_ID}
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "openai/gpt-oss-20b")
ASK_INTERVAL_S = float(os.getenv("EVAL_ASK_INTERVAL_S", "4.0"))  # server limit: 20 /ask per min
STAGE_RE = re.compile(r"<!--STAGE:(.*?)-->\n?", re.S)
PAGE_RE = re.compile(r"\b(?:pages?|pp?\.)\s*(\d{1,2}(?:\s*(?:,|and|&|-|–)\s*\d{1,2})*)", re.I)


def log(msg: str) -> None:
    print(msg, flush=True)


def load_bench() -> list[dict]:
    return json.loads((DATA / "blind_draft.json").read_text())


def parse_cited_pages(answer: str) -> list[int]:
    pages: list[int] = []
    for m in PAGE_RE.finditer(answer):
        pages += [int(x) for x in re.findall(r"\d+", m.group(1))]
    return sorted(set(p for p in pages if 1 <= p <= 28))


# ------------------------------------------------------------------ upload
async def upload() -> None:
    pdf = DATA / "ieee_roadmap.pdf"
    doc_file = OUT / "doc.json"
    async with httpx.AsyncClient(base_url=BASE, headers=HEAD, timeout=120) as c:
        with pdf.open("rb") as f:
            r = await c.post("/upload", files={"file": (pdf.name, f, "application/pdf")})
        r.raise_for_status()
        doc = r.json()
        doc_id = doc["doc_id"]
        log(f"uploaded {pdf.name} -> {doc_id} ({doc.get('status')})")
        t0 = time.monotonic()
        while True:
            s = (await c.get(f"/documents/{doc_id}/status")).json()
            log(f"  {s.get('status')} {s.get('progress')}% {s.get('current_stage')}")
            if s.get("status") == "ready":
                break
            if s.get("status") == "failed":
                raise SystemExit(f"indexing failed: {s.get('error')}")
            if time.monotonic() - t0 > 900:
                raise SystemExit("indexing timeout")
            await asyncio.sleep(8)
        doc_file.write_text(json.dumps({"doc_id": doc_id, "status": s}, indent=2))
        log(f"ready: pages={s.get('page_count')} sections={s.get('section_count')}")


# --------------------------------------------------------------------- run
async def ask_one(c: httpx.AsyncClient, doc_id: str, q: str) -> dict:
    payload = {"chat_id": str(uuid.uuid4()), "query": q, "doc_ids": [doc_id]}
    t0 = time.monotonic()
    ttfb = None
    buf = []
    err = None
    try:
        async with c.stream("POST", "/ask", json=payload) as r:
            if r.status_code != 200:
                err = f"HTTP {r.status_code}: {(await r.aread())[:300]!r}"
            else:
                async for chunk in r.aiter_text():
                    if ttfb is None:
                        ttfb = time.monotonic() - t0
                    buf.append(chunk)
    except Exception as e:  # recorded, never swallowed silently
        err = f"{type(e).__name__}: {e}"
    raw = "".join(buf)
    stages = [json.loads(m) for m in STAGE_RE.findall(raw) if m.startswith("{")]
    return {
        "answer": STAGE_RE.sub("", raw).strip(),
        "stages": stages,
        "used_web": any(s.get("stage") == "web_search" for s in stages),
        "used_arxiv": any(s.get("stage") == "arxiv_search" for s in stages),
        "latency_s": round(time.monotonic() - t0, 3),
        "ttfb_s": round(ttfb, 3) if ttfb is not None else None,
        "error": err,
    }


async def run() -> None:
    doc_id = json.loads((OUT / "doc.json").read_text())["doc_id"]
    bench = load_bench()
    path = OUT / "answers.json"
    done = {a["id"]: a for a in json.loads(path.read_text())} if path.exists() else {}
    async with httpx.AsyncClient(base_url=BASE, headers=HEAD, timeout=300) as c:
        for item in bench:
            if item["id"] in done and not done[item["id"]].get("error"):
                continue
            res = await ask_one(c, doc_id, item["question"])
            if res["error"] and "429" in res["error"]:
                log("rate limited; sleeping 60s"); await asyncio.sleep(60)
                res = await ask_one(c, doc_id, item["question"])
            res["id"] = item["id"]
            done[item["id"]] = res
            path.write_text(json.dumps(list(done.values()), indent=2, ensure_ascii=False))
            log(f"{item['id']} {item['category']:<12} {res['latency_s']:>6.1f}s "
                f"web={res['used_web']} arxiv={res['used_arxiv']} err={res['error']}")
            await asyncio.sleep(ASK_INTERVAL_S)


# ------------------------------------------------------------------- judge
JUDGE_SYS = (
    "You are a strict evaluation judge for a RAG system. Reply with a single JSON object only."
)
ANSWERABLE_PROMPT = """Question: {q}

Reference answer (ground truth): {ref}

Key facts (judge each independently against the SYSTEM ANSWER):
{facts}

Source evidence quotes from the paper:
{evidence}

SYSTEM ANSWER:
{ans}

Return JSON:
{{"facts_covered": [true/false per key fact, same order, true only if the system answer states it correctly],
  "contradicts_reference": true/false,
  "n_claims": <number of distinct factual claims in the system answer>,
  "n_supported": <how many of those claims are supported by the evidence quotes or reference answer>,
  "refused": true/false (the system says it cannot find/answer this)}}"""
UNANSWERABLE_PROMPT = """The question below is NOT answerable from the source paper (reason: {why}).

Question: {q}

SYSTEM ANSWER:
{ans}

Return JSON:
{{"refused": true/false (clearly states the information is not in the document / cannot be determined),
  "fabricated": true/false (asserts specific details, numbers or findings as if the paper stated them),
  "note": "<=15 words"}}"""


def groq_client():
    from openai import AsyncOpenAI
    key = os.getenv("GROQ_API_KEY_JUDGE") or os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY_JUDGE missing (expected in ../.env)")
    return AsyncOpenAI(api_key=key, base_url="https://api.groq.com/openai/v1", timeout=90)


async def judge_call(cli, prompt: str) -> dict:
    delay = 2.0
    for attempt in range(6):  # exponential backoff + jitter for transient failures
        try:
            r = await cli.chat.completions.create(
                model=JUDGE_MODEL, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": prompt}],
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            log(f"  judge retry {attempt + 1}: {type(e).__name__}: {str(e)[:120]}")
            await asyncio.sleep(delay + (uuid.uuid4().int % 1000) / 1000)
            delay *= 2
    raise RuntimeError("judge failed after retries")


async def judge() -> None:
    bench = {b["id"]: b for b in load_bench()}
    answers = {a["id"]: a for a in json.loads((OUT / "answers.json").read_text())}
    cli = groq_client()
    path = OUT / "judged.json"
    out = {j["id"]: j for j in json.loads(path.read_text())} if path.exists() else {}
    for qid, b in bench.items():
        if qid in out or qid not in answers:
            continue
        a = answers[qid]
        ans = a["answer"] or "(empty)"
        rec = {"id": qid, "category": b["category"], "answerable": b["answerable"]}
        if b["answerable"]:
            facts = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(b["key_facts"]))
            ev = "\n".join(f'[p{e["page"]}] "{e["quote"]}"' for e in b["evidence"])
            j = await judge_call(cli, ANSWERABLE_PROMPT.format(
                q=b["question"], ref=b["reference_answer"], facts=facts, evidence=ev, ans=ans))
            fc = j.get("facts_covered") or []
            fc = (fc + [False] * len(b["key_facts"]))[: len(b["key_facts"])]
            cited = parse_cited_pages(ans)
            gt = set(b["ground_truth_pages"])
            hit = gt & set(cited)
            rec.update({
                "fact_recall": sum(bool(x) for x in fc) / len(fc),
                "correct": sum(bool(x) for x in fc) / len(fc) >= 0.5 and not j.get("contradicts_reference"),
                "contradicts_reference": bool(j.get("contradicts_reference")),
                "faithfulness": (j.get("n_supported", 0) / j["n_claims"]) if j.get("n_claims") else None,
                "refused": bool(j.get("refused")),
                "cited_pages": cited,
                "citation_recall": len(hit) / len(gt),
                "citation_precision": (len(hit) / len(cited)) if cited else None,
                "any_gt_page_cited": bool(hit),
            })
        else:
            j = await judge_call(cli, UNANSWERABLE_PROMPT.format(
                why=b.get("why_unanswerable", ""), q=b["question"], ans=ans))
            probes = [p for p in b.get("absence_probes", []) if p.lower() in ans.lower()]
            rec.update({
                "refused": bool(j.get("refused")),
                "fabricated": bool(j.get("fabricated")),
                "absence_probe_hits": probes,  # deterministic hallucination signal
                "used_web": a["used_web"], "used_arxiv": a["used_arxiv"],
                "note": j.get("note"),
            })
        out[qid] = rec
        path.write_text(json.dumps(list(out.values()), indent=2, ensure_ascii=False))
        log(f"judged {qid}")
        await asyncio.sleep(1.0)


# ------------------------------------------------------------------ report
def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def report() -> None:
    answers = {a["id"]: a for a in json.loads((OUT / "answers.json").read_text())}
    judged = json.loads((OUT / "judged.json").read_text())
    ans_j = [j for j in judged if j["answerable"]]
    un_j = [j for j in judged if not j["answerable"]]
    by_cat = defaultdict(list)
    for j in ans_j:
        by_cat[j["category"]].append(j)

    def agg(rows):
        return {
            "n": len(rows),
            "answer_correctness": mean([r["correct"] for r in rows]),
            "key_fact_recall": mean([r["fact_recall"] for r in rows]),
            "faithfulness": mean([r["faithfulness"] for r in rows]),
            "citation_page_recall": mean([r["citation_recall"] for r in rows]),
            "citation_page_precision": mean([r["citation_precision"] for r in rows]),
            "any_gt_page_cited": mean([r["any_gt_page_cited"] for r in rows]),
            "false_refusal_rate": mean([r["refused"] for r in rows]),
        }

    lat = [answers[j["id"]]["latency_s"] for j in judged if j["id"] in answers]
    ttfb = [answers[j["id"]]["ttfb_s"] for j in judged if answers.get(j["id"], {}).get("ttfb_s")]
    errs = [a["id"] for a in answers.values() if a["error"]]
    rep = {
        "base_url": BASE, "judge_model": JUDGE_MODEL, "n_items": len(judged),
        "answerable_overall": agg(ans_j),
        "answerable_by_category": {k: agg(v) for k, v in sorted(by_cat.items())},
        "unanswerable": {
            "n": len(un_j),
            "correct_refusal_rate": mean([j["refused"] and not j["fabricated"] for j in un_j]),
            "fabrication_rate": mean([j["fabricated"] for j in un_j]),
            "absence_probe_hallucination_rate": mean([bool(j["absence_probe_hits"]) for j in un_j]),
            "web_fallback_rate": mean([j["used_web"] for j in un_j]),
            "arxiv_fallback_rate": mean([j["used_arxiv"] for j in un_j]),
        },
        "latency_s": {
            "mean": round(st.mean(lat), 2) if lat else None,
            "median": round(st.median(lat), 2) if lat else None,
            "p95": round(sorted(lat)[max(0, int(0.95 * len(lat)) - 1)], 2) if lat else None,
            "mean_ttfb": round(st.mean(ttfb), 2) if ttfb else None,
        },
        "web_or_arxiv_used_on_answerable": mean(
            [answers[j["id"]]["used_web"] or answers[j["id"]]["used_arxiv"] for j in ans_j]),
        "request_errors": errs,
        "caveats": [
            "Retrieval recall/context precision are not exposed by /ask; citation-page recall/precision "
            "(pages the answer cites vs. ground-truth evidence pages) is used as the retrieval proxy.",
            "Faithfulness = share of answer claims supported by the benchmark evidence quotes/reference "
            "(judge-estimated); correct-but-uncited claims count as unsupported.",
            "LLM judge differs from the answering model but is still an LLM; N is small (40).",
        ],
    }
    (OUT / "report.json").write_text(json.dumps(rep, indent=2))

    o = rep["answerable_overall"]
    md = [f"# RAG evaluation report\n", f"Server `{BASE}` · judge `{JUDGE_MODEL}` · {rep['n_items']} items\n",
          "## Answerable questions (N=%d)\n" % o["n"],
          "| Category | N | Correctness | Key-fact recall | Faithfulness | Cite recall | Cite precision | False refusal |",
          "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for name, r in [*rep["answerable_by_category"].items(), ("**Overall**", o)]:
        md.append(f"| {name} | {r['n']} | {pct(r['answer_correctness'])} | {pct(r['key_fact_recall'])} | "
                  f"{pct(r['faithfulness'])} | {pct(r['citation_page_recall'])} | "
                  f"{pct(r['citation_page_precision'])} | {pct(r['false_refusal_rate'])} |")
    u = rep["unanswerable"]
    md += ["\n## Unanswerable questions (N=%d)\n" % u["n"],
           f"- Correct refusal: **{pct(u['correct_refusal_rate'])}**",
           f"- Fabrication: **{pct(u['fabrication_rate'])}**",
           f"- Absence-probe hallucination (deterministic): **{pct(u['absence_probe_hallucination_rate'])}**",
           f"- Web fallback used: {pct(u['web_fallback_rate'])} · arXiv fallback: {pct(u['arxiv_fallback_rate'])}",
           "\n## Latency\n", f"- {rep['latency_s']}",
           f"\nWeb/arXiv tools used on answerable questions: {pct(rep['web_or_arxiv_used_on_answerable'])}",
           f"\nRequest errors: {errs or 'none'}\n", "## Caveats\n", *[f"- {c}" for c in rep["caveats"]]]
    (OUT / "report.md").write_text("\n".join(md))
    log("\n".join(md))


# ----------------------------------------------------------------- cleanup
async def cleanup() -> None:
    doc_id = json.loads((OUT / "doc.json").read_text())["doc_id"]
    async with httpx.AsyncClient(base_url=BASE, headers=HEAD, timeout=60) as c:
        r = await c.delete(f"/documents/{doc_id}")
        log(f"DELETE /documents/{doc_id} -> {r.status_code}")


STAGES = {"upload": upload, "run": run, "judge": judge, "cleanup": cleanup}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "report":
        report()
    elif cmd == "all":
        for s in (upload, run, judge):
            asyncio.run(s())
        report()
    elif cmd in STAGES:
        asyncio.run(STAGES[cmd]())
    else:
        raise SystemExit(__doc__)
