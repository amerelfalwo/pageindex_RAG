"""Validate the blind evaluation set against the source PDF text.

Everything here is deterministic and makes zero LLM calls. The source of
truth is ``data/pages.json`` (PDF text extracted with PyMuPDF).

Usage:
    python validate_blind.py data/blind_draft.json            # validate
    python validate_blind.py data/blind_draft.json --no-quota  # partial draft
    python validate_blind.py data/blind_draft.json --freeze    # write blind_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

CATEGORY_QUOTAS = {
    "Local": 10,
    "Multi-hop": 10,
    "Global": 8,
    "Numerical": 4,
    "Unanswerable": 8,
}
MAX_QUOTE_WORDS = 25
NGRAM_COPY = 7  # a question sharing 7 consecutive words with the paper is a copy
MIN_PAGE_COVERAGE = 0.5
REQUIRED_FIELDS = (
    "id",
    "category",
    "question",
    "answerable",
    "ground_truth_pages",
    "evidence",
    "reference_answer",
    "key_facts",
)
FORBIDDEN_IN_QUESTION = re.compile(
    r"\b(?:sections?|sec\.|pages?|fig\.|figures?|tables?|eq\.|equations?)\s*\(?\d"
    r"|§|\bappendix\b",
    re.IGNORECASE,
)
NUMBER = re.compile(r"\d[\d,.]*\d|\d")


@dataclass(frozen=True)
class Issue:
    level: str  # "ERROR" or "WARN"
    item_id: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.item_id}: {self.message}"


def canon(text: str) -> str:
    """Normalise text so PDF text and LLM-written quotes compare equal."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    for src, dst in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"')):
        text = text.replace(src, dst)
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def load_pages(path: Path) -> dict[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {p["page"]: canon(p["text"]) for p in data["pages"]}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", canon(text))


def copies_source(question: str, pages: dict[int, str], n: int = NGRAM_COPY) -> bool:
    words = _words(question)
    haystack = " ".join(" ".join(_words(t)) for t in pages.values())
    return any(
        " ".join(words[i : i + n]) in haystack for i in range(len(words) - n + 1)
    )


def check_item(item: dict, pages: dict[int, str]) -> list[Issue]:
    iid = str(item.get("id", "<no id>"))
    issues: list[Issue] = []

    def err(msg: str) -> None:
        issues.append(Issue("ERROR", iid, msg))

    def warn(msg: str) -> None:
        issues.append(Issue("WARN", iid, msg))

    missing = [f for f in REQUIRED_FIELDS if f not in item]
    if missing:
        err(f"missing fields: {missing}")
        return issues

    category = item["category"]
    if category not in CATEGORY_QUOTAS:
        err(f"unknown category {category!r}")

    question = item["question"]
    if FORBIDDEN_IN_QUESTION.search(question):
        err("question refers to Section/Page/Figure/Table/Appendix explicitly")
    if copies_source(question, pages):
        err(f"question copies {NGRAM_COPY}+ consecutive words from the paper")

    all_text = " ".join(pages.values())
    answerable = bool(item["answerable"])

    if category == "Unanswerable" and answerable:
        err("category Unanswerable but answerable=true")
    if category != "Unanswerable" and not answerable:
        err("answerable=false only allowed for category Unanswerable")

    if not answerable:
        if item["evidence"] or item["ground_truth_pages"]:
            err("unanswerable item must have empty evidence and pages")
        if not str(item.get("why_unanswerable", "")).strip():
            err("missing why_unanswerable")
        probes = item.get("absence_probes") or []
        if len(probes) < 2:
            err("need >=2 absence_probes (distinctive terms claimed absent)")
        for probe in probes:
            if canon(probe) in all_text:
                err(f"absence probe {probe!r} IS present in the paper")
        return issues

    # ---- answerable items -------------------------------------------------
    evidence = item["evidence"]
    if not evidence:
        err("answerable item has no evidence")
        return issues

    quote_text = []
    pages_seen: set[int] = set()
    for ev in evidence:
        page, quote = ev.get("page"), str(ev.get("quote", ""))
        if page not in pages:
            err(f"evidence page {page!r} outside 1..{max(pages)}")
            continue
        pages_seen.add(page)
        quote_text.append(quote)
        if len(quote.split()) > MAX_QUOTE_WORDS:
            err(f"quote longer than {MAX_QUOTE_WORDS} words (p{page})")
        if canon(quote) not in pages[page]:
            hit = [p for p, t in pages.items() if canon(quote) in t]
            where = f"found on page(s) {hit}" if hit else "not found anywhere"
            err(f"quote not verbatim on page {page}: {where}: {quote[:60]!r}")

    if sorted(set(item["ground_truth_pages"])) != sorted(pages_seen):
        err(
            f"ground_truth_pages {item['ground_truth_pages']} != "
            f"pages of evidence {sorted(pages_seen)}"
        )

    if category == "Local" and len(pages_seen) != 1:
        err("Local question must rely on exactly one page")
    if category == "Multi-hop" and len(pages_seen) < 2:
        err("Multi-hop question must rely on >=2 pages")
    if category == "Global" and len(pages_seen) < 2:
        warn("Global question normally spans >=2 pages")

    facts = item["key_facts"]
    if not (2 <= len(facts) <= 5) or any(not str(f).strip() for f in facts):
        err("key_facts must be 2-5 non-empty strings")

    if category == "Numerical":
        evidence_numbers = set(NUMBER.findall(" ".join(quote_text)))
        unsupported = set(NUMBER.findall(item["reference_answer"])) - evidence_numbers
        if unsupported:
            err(f"numbers in reference_answer not in evidence: {sorted(unsupported)}")

    return issues


def check_dataset(
    items: list[dict], pages: dict[int, str], enforce_quota: bool = True
) -> list[Issue]:
    issues: list[Issue] = []
    ids = [str(i.get("id")) for i in items]
    for dup, count in Counter(ids).items():
        if count > 1:
            issues.append(Issue("ERROR", dup, f"duplicate id ({count}x)"))
    questions = [canon(str(i.get("question", ""))) for i in items]
    for dup, count in Counter(questions).items():
        if count > 1:
            issues.append(Issue("ERROR", "<dataset>", f"duplicate question: {dup[:50]!r}"))

    for item in items:
        issues.extend(check_item(item, pages))

    if enforce_quota:
        counts = Counter(i.get("category") for i in items)
        for cat, quota in CATEGORY_QUOTAS.items():
            if counts.get(cat, 0) != quota:
                issues.append(
                    Issue("ERROR", "<dataset>", f"{cat}: {counts.get(cat, 0)} != {quota}")
                )

    covered = {
        ev.get("page")
        for i in items
        for ev in i.get("evidence", [])
        if ev.get("page") in pages
    }
    if items and len(covered) / len(pages) < MIN_PAGE_COVERAGE:
        issues.append(
            Issue(
                "WARN",
                "<dataset>",
                f"evidence covers only {len(covered)}/{len(pages)} pages; spread questions out",
            )
        )
    return issues


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(dataset_path: Path, pages_path: Path, out_path: Path) -> str:
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists; frozen sets are never overwritten")
    items = json.loads(dataset_path.read_text(encoding="utf-8"))
    payload = {
        "meta": {
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "n_items": len(items),
            "pages_json_sha256": sha256_of(pages_path),
        },
        "items": items,
    }
    body = json.dumps(payload, ensure_ascii=False, indent=1)
    out_path.write_text(body, encoding="utf-8")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--pages", type=Path, default=Path("data/pages.json"))
    parser.add_argument("--no-quota", action="store_true", help="allow partial drafts")
    parser.add_argument("--freeze", action="store_true", help="write data/blind_v1.json")
    args = parser.parse_args(argv)

    pages = load_pages(args.pages)
    items = json.loads(args.dataset.read_text(encoding="utf-8"))
    issues = check_dataset(items, pages, enforce_quota=not args.no_quota)

    for issue in issues:
        print(issue)
    errors = sum(i.level == "ERROR" for i in issues)
    warns = len(issues) - errors
    print(f"\n{len(items)} items | {errors} errors | {warns} warnings")

    if args.freeze:
        if errors:
            print("Refusing to freeze: fix errors first.")
            return 1
        digest = freeze(args.dataset, args.pages, args.dataset.parent / "blind_v1.json")
        print(f"Frozen -> blind_v1.json (sha256 {digest})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
