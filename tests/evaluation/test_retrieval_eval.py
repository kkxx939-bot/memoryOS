"""检索效果评估基础指标的单元测试。"""

import pytest

from tests.evaluation.retrieval_eval import precision_at_k, recall_at_k


@pytest.mark.parametrize("k", [0, -1])
def test_metrics_return_zero_for_non_positive_k(k: int) -> None:
    """无有效截断范围时，不应把任何检索结果计入指标。"""

    assert precision_at_k(["memory-1"], ["memory-1"], k) == 0.0
    assert recall_at_k(["memory-1"], ["memory-1"], k) == 0.0


def test_precision_at_k_counts_relevant_results_in_top_k() -> None:
    """准确率只统计前 K 条结果，并以实际返回数量为分母。"""

    results = ["memory-1", "memory-2", "memory-3"]
    expected = ["memory-1", "memory-3"]

    assert precision_at_k(results, expected, 2) == 0.5


def test_recall_at_k_counts_expected_results_covered_by_top_k() -> None:
    """召回率应统计期望结果中被前 K 条结果覆盖的比例。"""

    results = ["memory-1", "memory-2", "memory-3"]
    expected = ["memory-1", "memory-3"]

    assert recall_at_k(results, expected, 2) == 0.5


def test_metrics_return_zero_without_results_or_expected_ids() -> None:
    """缺少实际结果或期望结果时，指标应返回零。"""

    assert precision_at_k([], ["memory-1"], 3) == 0.0
    assert recall_at_k(["memory-1"], [], 3) == 0.0
