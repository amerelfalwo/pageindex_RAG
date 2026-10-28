# RAG evaluation report

Server `http://localhost:7860` · judge `openai/gpt-oss-20b` · 40 items

## Answerable questions (N=32)

| Category | N | Correctness | Key-fact recall | Faithfulness | Cite recall | Cite precision | False refusal |
|---|--:|--:|--:|--:|--:|--:|--:|
| Global | 8 | 0.0% | 3.1% | 17.9% | 0.0% | n/a | 75.0% |
| Local | 10 | 20.0% | 15.0% | 13.3% | 0.0% | 0.0% | 70.0% |
| Multi-hop | 10 | 30.0% | 29.2% | 29.1% | 10.0% | 75.0% | 60.0% |
| Numerical | 4 | 0.0% | 0.0% | n/a | 0.0% | n/a | 100.0% |
| **Overall** | 32 | 15.6% | 14.6% | 20.4% | 3.1% | 30.0% | 71.9% |

## Unanswerable questions (N=8)

- Correct refusal: **100.0%**
- Fabrication: **0.0%**
- Absence-probe hallucination (deterministic): **0.0%**
- Web fallback used: 0.0% · arXiv fallback: 0.0%

## Latency

- {'mean': 33.53, 'median': 52.73, 'p95': 56.68, 'mean_ttfb': 12.77}

Web/arXiv tools used on answerable questions: 3.1%

Request errors: none

## Caveats

- Retrieval recall/context precision are not exposed by /ask; citation-page recall/precision (pages the answer cites vs. ground-truth evidence pages) is used as the retrieval proxy.
- Faithfulness = share of answer claims supported by the benchmark evidence quotes/reference (judge-estimated); correct-but-uncited claims count as unsupported.
- LLM judge differs from the answering model but is still an LLM; N is small (40).