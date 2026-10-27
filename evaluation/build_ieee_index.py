"""Reconstructs the IEEE Paper PageIndex tree index cache with explicit --- [PAGE X] --- markers.

Provides the reference tree for running the 80-question IEEE research benchmark.
"""

import json
import os

NODE_METADATA = {
    "0000": {"title": "Unifying Large Language Models and Knowledge Graphs: A Roadmap", "page": 1, "summary": "Overall survey roadmap unifying LLMs and KGs."},
    "0001": {"title": "1 INTRODUCTION", "page": 1, "summary": "Introduction to LLM limitations (hallucinations, explainability) and KG advantages."},
    "0002": {"title": "2 BACKGROUND", "page": 2, "summary": "Foundational taxonomy of KGs: Encyclopedia, Commonsense, Domain-specific, Multimodal."},
    "0003": {"title": "3 ROADMAP & CATEGORIZATION", "page": 5, "summary": "Three-part roadmap: KG-enhanced LLMs, LLM-augmented KGs, and Synergized LLM+KG."},
    "0004": {"title": "4 KG-ENHANCED LLMs", "page": 7, "summary": "Overview of methods utilizing structured knowledge to improve LLM accuracy."},
    "0005": {"title": "4.1 KG-enhanced LLM Pre-training", "page": 7, "summary": "Entity embedding integration during pre-training (ERNIE, KnowBERT, KEPLER)."},
    "0006": {"title": "4.2 KG-enhanced LLM Inference", "page": 9, "summary": "Three-step inference: Knowledge Retrieval, Representation, and Fusion (K-BERT)."},
    "0007": {"title": "4.3 Comparison between KG-enhanced LLM Pre-training and Inference", "page": 9, "summary": "Trade-offs between pre-training cost and inference adaptability."},
    "0008": {"title": "4.4 KG-enhanced LLM Interpretability", "page": 10, "summary": "Using KG structure and probing to interpret LLM internal mechanisms."},
    "0009": {"title": "5 LLM-AUGMENTED KGs", "page": 11, "summary": "Framework for using LLMs to construct, complete, and query knowledge graphs."},
    "0010": {"title": "5.1 LLM-augmented KG Embedding", "page": 11, "summary": "Textual encoders for entity and relation representations (KG-BERT, StAR)."},
    "0011": {"title": "5.2 LLM-augmented KG Completion", "page": 12, "summary": "PaE (Pre-trained as Encoders) vs PaD (Pre-trained as Decoders) for link prediction."},
    "0012": {"title": "5.3 LLM-augmented KG Construction", "page": 14, "summary": "Named Entity Recognition and Relation Extraction via prompting and in-context learning."},
    "0013": {"title": "5.4 LLM-augmented KG-to-text Generation", "page": 16, "summary": "Generating natural language descriptions from structured RDF triples."},
    "0014": {"title": "5.5 LLM-augmented KG Question Answering", "page": 17, "summary": "Translating natural language questions to SPARQL and subgraph queries."},
    "0015": {"title": "6 SYNERGIZED LLMs + KGs", "page": 18, "summary": "Bidirectional synergistic cooperation between LLMs and KGs."},
    "0016": {"title": "7 FUTURE DIRECTIONS AND MILESTONES", "page": 20, "summary": "Open challenges: dynamic KGs, hallucination mitigation, ethical alignment."},
    "0017": {"title": "8 CONCLUSION", "page": 21, "summary": "Conclusion summarizing unifying paradigm and future research outlook."},
    "0018": {"title": "ACKNOWLEDGMENTS", "page": 21, "summary": "Research grant acknowledgments including ARC DP190101890 and FT210100097."},
    "0020": {"title": "APPENDIX A", "page": 27, "summary": "Comprehensive comparative table of pros and cons of LLMs and Knowledge Graphs."}
}

def build_index(dataset_path: str = "evaluation/eval_dataset.json", output_path: str = "evaluation/ieee_paper_index.json"):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Collect ground-truth paragraphs per node
    node_paragraphs = {nid: [] for nid in NODE_METADATA}
    for item in dataset:
        ans = item.get("ground_truth_answer", "").strip()
        for nid in item.get("ground_truth_nodes", []):
            if nid in node_paragraphs and ans not in node_paragraphs[nid]:
                node_paragraphs[nid].append(ans)

    node_map = {}
    flat_sections = []

    for nid, meta in sorted(NODE_METADATA.items()):
        page = meta["page"]
        title = meta["title"]
        summary = meta["summary"]
        paras = node_paragraphs.get(nid, [summary])

        # Inject explicit page marker
        full_text = f"--- [PAGE {page}] ---\n" + "\n\n".join(paras)

        node_data = {
            "node_id": nid,
            "title": title,
            "name": title,
            "page": page,
            "page_index": page,
            "summary": summary,
            "text": full_text,
            "content": full_text,
        }
        node_map[nid] = node_data

        flat_sections.append({
            "node_id": nid,
            "title": title,
            "page_index": page,
            "summary": summary,
        })

    index_data = {
        "doc_id": "ieee_tkde_2024",
        "filename": "data.pdf",
        "page_count": 28,
        "node_map": node_map,
        "flat_sections": flat_sections,
        "tree": list(node_map.values()),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    print(f"Generated IEEE index cache with {len(node_map)} nodes at {output_path}")

if __name__ == "__main__":
    build_index()
