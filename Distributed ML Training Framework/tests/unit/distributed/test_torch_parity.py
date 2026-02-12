import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dmlf.distributed.algorithms import parameter_server_average, ring_allreduce_average
from dmlf.distributed.mock_tensor import MockTensor
from dmlf.env import DependencyError, require_torch


def _to_mock(values: list[float]) -> MockTensor:
    return MockTensor(values)


class TorchParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.torch = require_torch("running torch-backed parity tests")
        except DependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def test_parameter_server_average_matches_torch_mean(self) -> None:
        grads = [
            _to_mock([1.0, 2.0, 3.0, 4.0]),
            _to_mock([2.5, 1.5, 0.5, -0.5]),
            _to_mock([-1.0, 0.0, 2.0, 6.0]),
        ]

        mock_avg = parameter_server_average(grads)

        torch_grads = self.torch.tensor([g.values for g in grads], dtype=self.torch.float64)
        torch_avg = self.torch.mean(torch_grads, dim=0)

        self.assertEqual(len(mock_avg.values), torch_avg.numel())
        for i, value in enumerate(mock_avg.values):
            self.assertAlmostEqual(value, float(torch_avg[i].item()), places=12)

    def test_ring_allreduce_average_matches_torch_mean_per_rank(self) -> None:
        grads = [
            _to_mock([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]),
            _to_mock([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            _to_mock([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]),
            _to_mock([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
        ]

        reduced = ring_allreduce_average(grads)

        torch_grads = self.torch.tensor([g.values for g in grads], dtype=self.torch.float64)
        torch_avg = self.torch.mean(torch_grads, dim=0)

        self.assertEqual(len(reduced), len(grads))
        for rank_tensor in reduced:
            self.assertEqual(len(rank_tensor.values), torch_avg.numel())
            for i, value in enumerate(rank_tensor.values):
                self.assertAlmostEqual(value, float(torch_avg[i].item()), places=12)


if __name__ == "__main__":
    unittest.main()
