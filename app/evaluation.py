from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    recall_at_k: float
    mean_reciprocal_rank: float
    cases: int


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    denominator = left_norm * right_norm
    return numerator / denominator if denominator else 0.0


def evaluate_rankings(rankings: list[list[str]], expected_ids: list[set[str]]) -> RetrievalMetrics:
    if not rankings or len(rankings) != len(expected_ids):
        raise ValueError("Rankings and expected IDs must contain the same non-zero number of cases")

    hits = 0
    reciprocal_rank_total = 0.0
    for ranking, expected in zip(rankings, expected_ids, strict=True):
        if expected.intersection(ranking):
            hits += 1
        for index, candidate in enumerate(ranking, start=1):
            if candidate in expected:
                reciprocal_rank_total += 1 / index
                break

    case_count = len(rankings)
    return RetrievalMetrics(
        recall_at_k=hits / case_count,
        mean_reciprocal_rank=reciprocal_rank_total / case_count,
        cases=case_count,
    )
