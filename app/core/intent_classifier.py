import re
from typing import Dict, Any, List, Set


class QueryIntent:
    GLOBAL_SUMMARY = "global_summary"
    TAXONOMY_ARCHITECTURE = "taxonomy_architecture"
    CROSS_DOC_SUMMARY = "cross_doc_summary"
    SPECIFIC_FACT = "specific_fact"


class QueryIntentClassifier:
    """
    Ultra-fast deterministic query intent classifier for Atlas Vectorless RAG.
    Zero external dependencies. Operates in < 0.1ms using compiled regular expressions.
    """

    # English Triggers
    _EN_GLOBAL_PATTERNS = [
        r"\b(summar(y|ize|ization)|overview|synopsis|abstract|tl;?dr|recap|outline)\b",
        r"\b(what (is|are) (this|the) (document|paper|file|report|text) about)\b",
        r"\b(what does (this|the) (document|paper|report) discuss)\b",
        r"\b(main (points|topics|themes|takeaways|findings|contributions|ideas|results))\b",
        r"\b(core (findings|themes|insights|contributions))\b",
        r"\b(executive summary|high-?level (view|overview|summary))\b",
        r"\b(table of contents|general (idea|summary|overview))\b",
        r"\b(big picture|conclusions?)\b",
    ]

    _EN_TAXONOMY_ARCH_PATTERNS = [
        r"\b(taxonom(y|ies)|architectur(e|al)|framework|system design|components?)\b",
        r"\b(categor(y|ies|ization)|classification|hierarch(y|ical))\b",
        r"\b(workflow|pipeline|roadmap|building blocks|structural?|diagrams?)\b",
    ]

    _EN_CROSS_DOC_PATTERNS = [
        r"\b(compar(e|ison|ative)|contrast|difference between)\b",
        r"\b(across (the )?(documents|papers|files|reports))\b",
        r"\b(both (documents|papers|files)|all documents)\b",
        r"\b(similarit(y|ies)|commonalit(y|ies))\b",
    ]

    # Arabic Triggers
    _AR_GLOBAL_PATTERNS = [
        r"(ملخص|لخص|نبذة|نظرة عامة|فحوى|فكرة عامة|المحاور الرئيسية|الفهرس|مواضيع)",
        r"(عن ماذا يتحدث|عن ايش يتكلم|عن إيه بيتكلم|عن ايه بيتكلم|عن إيه|عن ايه|ما هو موضوع|ما هو محتوى|محتوى المستند)",
        r"(أهم النقاط|النتائج الرئيسية|الاستنتاجات العامة|خلاصة|إيجاز|ملخص تنفيذي)",
        r"(دا عن ايه|ده عن ايه|دي عن ايه|دا بيتكلم عن ايه|ده بيتكلم عن ايه)",
        r"^(دا|ده|دي|لخصه|لخصها|الملخص)$",
    ]

    _AR_TAXONOMY_ARCH_PATTERNS = [
        r"(تصنيف|تصنيفات|معمارية|هيكلية|تصميم النظام|مكونات|فئات)",
        r"(مخطط|سير العمل|أقسام|أبواب|إطار العمل|خارطة طريق|بنية النظام)",
    ]

    _AR_CROSS_DOC_PATTERNS = [
        r"(مقارنة|قارن|الفرق بين|أوجه التشابه|أوجه الاختلاف)",
        r"(بين المستندين|بين الملفين|بين الوثيقتين|جميع المستندات|عبر الوثائق)",
    ]

    def __init__(self):
        self.en_global_re = [re.compile(p, re.IGNORECASE) for p in self._EN_GLOBAL_PATTERNS]
        self.en_tax_re = [re.compile(p, re.IGNORECASE) for p in self._EN_TAXONOMY_ARCH_PATTERNS]
        self.en_cross_re = [re.compile(p, re.IGNORECASE) for p in self._EN_CROSS_DOC_PATTERNS]

        self.ar_global_re = [re.compile(p) for p in self._AR_GLOBAL_PATTERNS]
        self.ar_tax_re = [re.compile(p) for p in self._AR_TAXONOMY_ARCH_PATTERNS]
        self.ar_cross_re = [re.compile(p) for p in self._AR_CROSS_DOC_PATTERNS]

    def classify(self, query: str, doc_count: int = 1) -> Dict[str, Any]:
        """
        Classifies user query intent.
        Returns:
            {
                "intent": QueryIntent.*,
                "is_global_or_summary": bool,
                "matched_triggers": List[str],
                "confidence": float
            }
        """
        if not query:
            return {
                "intent": QueryIntent.SPECIFIC_FACT,
                "is_global_or_summary": False,
                "matched_triggers": [],
                "confidence": 0.0,
            }

        q_clean = query.strip()
        q_lower = q_clean.lower()
        matched_triggers: List[str] = []

        # 1. Very short generic queries (e.g. "دا", "لخص", "summary", "overview")
        if len(q_clean) <= 12 and (
            q_clean in {"دا", "ده", "دي", "لخص", "لخصه", "ملخص"}
            or q_lower in {"summary", "summarize", "overview", "tldr", "tl;dr", "about", "what is it"}
        ):
            matched_triggers.append(f"short_trigger:{q_clean}")
            return {
                "intent": QueryIntent.GLOBAL_SUMMARY,
                "is_global_or_summary": True,
                "matched_triggers": matched_triggers,
                "confidence": 0.95,
            }

        # 2. Cross-document / Comparison intent
        for r in self.en_cross_re:
            m = r.search(q_lower)
            if m:
                matched_triggers.append(f"en_cross:{m.group(0)}")
        for r in self.ar_cross_re:
            m = r.search(q_clean)
            if m:
                matched_triggers.append(f"ar_cross:{m.group(0)}")

        if matched_triggers and doc_count > 1:
            return {
                "intent": QueryIntent.CROSS_DOC_SUMMARY,
                "is_global_or_summary": True,
                "matched_triggers": matched_triggers,
                "confidence": 0.90,
            }

        # 3. Taxonomy / Architecture intent
        tax_triggers: List[str] = []
        for r in self.en_tax_re:
            m = r.search(q_lower)
            if m:
                tax_triggers.append(f"en_tax:{m.group(0)}")
        for r in self.ar_tax_re:
            m = r.search(q_clean)
            if m:
                tax_triggers.append(f"ar_tax:{m.group(0)}")

        if tax_triggers:
            matched_triggers.extend(tax_triggers)
            return {
                "intent": QueryIntent.TAXONOMY_ARCHITECTURE,
                "is_global_or_summary": True,
                "matched_triggers": matched_triggers,
                "confidence": 0.88,
            }

        # 4. Global / Overview / Summary intent
        glob_triggers: List[str] = []
        for r in self.en_global_re:
            m = r.search(q_lower)
            if m:
                glob_triggers.append(f"en_global:{m.group(0)}")
        for r in self.ar_global_re:
            m = r.search(q_clean)
            if m:
                glob_triggers.append(f"ar_global:{m.group(0)}")

        if glob_triggers:
            matched_triggers.extend(glob_triggers)
            return {
                "intent": QueryIntent.GLOBAL_SUMMARY,
                "is_global_or_summary": True,
                "matched_triggers": matched_triggers,
                "confidence": 0.92,
            }

        # 5. Default to Specific Fact lookup
        return {
            "intent": QueryIntent.SPECIFIC_FACT,
            "is_global_or_summary": False,
            "matched_triggers": [],
            "confidence": 0.50,
        }


# Global singleton instance for high-throughput zero-alloc usage
intent_classifier = QueryIntentClassifier()
