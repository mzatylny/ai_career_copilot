import pytest

from app.evaluation import cosine_similarity, evaluate_rankings


def test_cosine_similarity_handles_aligned_and_empty_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_retrieval_metrics_measure_hits_and_rank():
    metrics = evaluate_rankings(
        [["a", "b"], ["x", "c"]],
        [{"a"}, {"c"}],
    )

    assert metrics.recall_at_k == 1.0
    assert metrics.mean_reciprocal_rank == 0.75
    assert metrics.cases == 2


def test_retrieval_metrics_reject_mismatched_cases():
    with pytest.raises(ValueError):
        evaluate_rankings([["a"]], [])
