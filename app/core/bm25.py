import math
import re
import time
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import Counter
from utils.logger import app_logger


STOPWORDS: Set[str] = {
    # English stopwords
    'the', 'and', 'for', 'that', 'this', 'with', 'from', 'what', 'which', 'where',
    'when', 'how', 'why', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'does',
    'did', 'about', 'some', 'any', 'each', 'into', 'over', 'after', 'before', 'such',
    'a', 'an', 'in', 'on', 'at', 'to', 'by', 'of', 'is', 'it', 'its', 'as', 'be',
    # Arabic stopwords
    'في', 'من', 'على', 'إلى', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك', 'التي', 'الذي',
    'ما', 'ماذا', 'كيف', 'لماذا', 'أين', 'متى', 'كم', 'هل', 'هو', 'هي', 'أن', 'إن', 'كان'
}


class BM25NodeIndex:
    """
    Lightweight, in-memory Okapi BM25 index over flattened document nodes.
    Zero external dependencies. Highly optimized for section-level retrieval.
    """
    def __init__(self, nodes: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.node_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.doc_freqs: Counter = Counter()
        self.N: int = 0
        self.avgdl: float = 0.0

        self._build_index(nodes)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        if not text:
            return []
        tokens = re.findall(r'[\w\-]+', text.lower())
        return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]

    def _build_index(self, nodes: List[Dict[str, Any]]):
        self.node_ids = []
        self.tokenized_corpus = []
        self.doc_lens = []
        self.doc_freqs = Counter()

        for node in nodes:
            nid = node.get("node_id")
            if not nid:
                continue

            # Weighted field representation:
            # Title: 3x boost
            # Summary: 2x boost
            # Text: 1x (first 8000 chars to cover full sections)
            title = node.get("title") or ""
            summary = node.get("summary") or ""
            text = (node.get("text") or node.get("content") or "")[:8000]

            title_tokens = self.tokenize(title) * 3
            summary_tokens = self.tokenize(summary) * 2
            text_tokens = self.tokenize(text)

            combined_tokens = title_tokens + summary_tokens + text_tokens
            if not combined_tokens:
                combined_tokens = self.tokenize(title or nid) or ["document"]

            self.node_ids.append(nid)
            self.tokenized_corpus.append(combined_tokens)
            dl = len(combined_tokens)
            self.doc_lens.append(dl)
            self.doc_freqs.update(set(combined_tokens))

        self.N = len(self.node_ids)
        self.avgdl = (sum(self.doc_lens) / self.N) if self.N > 0 else 1.0

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Executes Okapi BM25 scoring over indexed nodes.
        Returns [(node_id, score), ...] sorted descending.
        """
        if self.N == 0:
            return []

        q_tokens = self.tokenize(query)
        if not q_tokens:
            # Fallback to raw tokens if all words were stopwords
            raw_tokens = re.findall(r'[\w\-]+', query.lower())
            q_tokens = [t for t in raw_tokens if len(t) >= 2]
            if not q_tokens:
                return []

        scores: List[Tuple[str, float]] = []

        for idx, doc_tokens in enumerate(self.tokenized_corpus):
            nid = self.node_ids[idx]
            dl = self.doc_lens[idx]
            tf = Counter(doc_tokens)
            score = 0.0

            for q in q_tokens:
                if q in tf:
                    n_q = self.doc_freqs[q]
                    # Robertson-Spärck Jones IDF
                    idf = math.log((self.N - n_q + 0.5) / (n_q + 0.5) + 1.0)
                    f_qd = tf[q]
                    denom = f_qd + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                    score += idf * (f_qd * (self.k1 + 1.0)) / denom

            if score > 0.0:
                scores.append((nid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
