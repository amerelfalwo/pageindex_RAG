# RAG evaluation report

Server `http://localhost:7860` · judge `qwen/qwen3.8-27b` · 40 items

## Answerable questions (N=32)

| Category | N | Correctness | Key-fact recall | Faithfulness | Cite recall | Cite precision | False refusal |
|---|--:|--:|--:|--:|--:|--:|--:|
| Global | 8 | 25.0% | 32.5% | 24.6% | 6.2% | 50.0% | 12.5% |
| Local | 10 | 40.0% | 40.0% | 30.0% | 10.0% | 50.0% | 50.0% |
| Multi-hop | 10 | 30.0% | 33.3% | 27.8% | 0.0% | 0.0% | 50.0% |
| Numerical | 4 | 25.0% | 16.7% | 25.0% | 0.0% | n/a | 75.0% |
| **Overall** | 32 | 31.2% | 33.1% | 27.3% | 4.7% | 33.3% | 43.8% |

## Unanswerable questions (N=8)

- Correct refusal: **100.0%**
- Fabrication: **0.0%**
- Absence-probe hallucination (deterministic): **37.5%**
- Web fallback used: 0.0% · arXiv fallback: 12.5%

## Latency

- {'mean': 31.97, 'median': 33.94, 'p95': 53.69, 'mean_ttfb': 12.17}

Web/arXiv tools used on answerable questions: 21.9%

Request errors: none

## Caveats

- Retrieval recall/context precision are not exposed by /ask; citation-page recall/precision (pages the answer cites vs. ground-truth evidence pages) is used as the retrieval proxy.
- Faithfulness = share of answer claims supported by the benchmark evidence quotes/reference (judge-estimated); correct-but-uncited claims count as unsupported.
- LLM judge differs from the answering model but is still an LLM; N is small (40).