import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dmlf.distributed.algorithms import parameter_server_average, ring_allreduce_average
from dmlf.distributed.mock_tensor import MockTensor


class RingAllReduceTests(unittest.TestCase):
    def test_ring_allreduce_matches_parameter_server_average(self) -> None:
        rank_grads = [
            MockTensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            MockTensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            MockTensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            MockTensor([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]),
        ]

        expected = parameter_server_average(rank_grads)
        reduced = ring_allreduce_average(rank_grads)

        self.assertEqual(len(reduced), len(rank_grads))
        for rank_tensor in reduced:
            self.assertTrue(rank_tensor.almost_equal(expected))

    def test_ring_allreduce_single_rank_is_identity(self) -> None:
        grads = [MockTensor([1.0, 2.0, 3.0, 4.0])]
        reduced = ring_allreduce_average(grads)
        self.assertEqual(reduced[0].values, grads[0].values)

    def test_ring_allreduce_requires_chunk_divisibility(self) -> None:
        grads = [
            MockTensor([1.0, 2.0, 3.0]),
            MockTensor([4.0, 5.0, 6.0]),
        ]

        with self.assertRaises(ValueError):
            ring_allreduce_average(grads)


if __name__ == "__main__":
    unittest.main()
