# RAG evaluation report

Server `http://localhost:7860` · judge `openai/gpt-oss-20b` · 40 items

## Answerable questions (N=32)

| Category | N | Correctness | Key-fact recall | Faithfulness | Cite recall | Cite precision | False refusal |
|---|--:|--:|--:|--:|--:|--:|--:|
| Global | 8 | 12.5% | 15.6% | 22.8% | 0.0% | 0.0% | 62.5% |
| Local | 10 | 30.0% | 21.7% | 23.3% | 10.0% | 50.0% | 30.0% |
| Multi-hop | 10 | 30.0% | 30.8% | 22.6% | 5.0% | 25.0% | 50.0% |
| Numerical | 4 | 0.0% | 0.0% | n/a | 0.0% | n/a | 100.0% |
| **Overall** | 32 | 21.9% | 20.3% | 22.9% | 4.7% | 20.0% | 53.1% |

## Unanswerable questions (N=8)

- Correct refusal: **100.0%**
- Fabrication: **0.0%**
- Absence-probe hallucination (deterministic): **0.0%**
- Web fallback used: 0.0% · arXiv fallback: 0.0%

## Latency

- {'mean': 54.24, 'median': 55.21, 'p95': 58.31, 'mean_ttfb': 18.15}

Web/arXiv tools used on answerable questions: 3.1%

Request errors: none

## Caveats

- Retrieval recall/context precision are not exposed by /ask; citation-page recall/precision (pages the answer cites vs. ground-truth evidence pages) is used as the retrieval proxy.
- Faithfulness = share of answer claims supported by the benchmark evidence quotes/reference (judge-estimated); correct-but-uncited claims count as unsupported.
- LLM judge differs from the answering model but is still an LLM; N is small (40).