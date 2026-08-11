from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.embeddings import hash_embedding
from app.evaluation import cosine_similarity, evaluate_rankings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic RAG retrieval evaluation")
    parser.add_argument("--dataset", default="evals/rag_cases.json")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--minimum-recall", type=float, default=0.75)
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    documents = dataset["documents"]
    dimensions = get_settings().embedding_dimensions
    document_vectors = {
        document["id"]: hash_embedding(document["text"], dimensions) for document in documents
    }

    rankings: list[list[str]] = []
    expected: list[set[str]] = []
    for case in dataset["cases"]:
        query_vector = hash_embedding(case["query"], dimensions)
        ranked = sorted(
            document_vectors,
            key=lambda document_id: cosine_similarity(query_vector, document_vectors[document_id]),
            reverse=True,
        )
        rankings.append(ranked[: args.k])
        expected.append(set(case["expected_ids"]))

    metrics = evaluate_rankings(rankings, expected)
    print(
        json.dumps(
            {
                "cases": metrics.cases,
                "recall_at_k": round(metrics.recall_at_k, 4),
                "mean_reciprocal_rank": round(metrics.mean_reciprocal_rank, 4),
                "k": args.k,
            },
            indent=2,
        )
    )
    return 0 if metrics.recall_at_k >= args.minimum_recall else 1


if __name__ == "__main__":
    raise SystemExit(main())
