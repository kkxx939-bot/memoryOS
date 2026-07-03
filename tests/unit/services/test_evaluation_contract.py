from __future__ import annotations

import unittest

from memoryos.evaluation.retrieval_eval import precision_at_k, recall_at_k


class EvaluationContractTest(unittest.TestCase):
    def test_precision_and_recall_at_k(self) -> None:
        results = ["a", "b", "c"]
        expected = ["b", "d"]

        self.assertEqual(precision_at_k(results, expected, 2), 0.5)
        self.assertEqual(recall_at_k(results, expected, 2), 0.5)


if __name__ == "__main__":
    unittest.main()
