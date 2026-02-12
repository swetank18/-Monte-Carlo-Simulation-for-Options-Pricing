import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dmlf.distributed.algorithms import parameter_server_average
from dmlf.distributed.mock_tensor import MockTensor


class ParameterServerAggregationTests(unittest.TestCase):
    def test_mock_tensor_addition_and_scaling(self) -> None:
        left = MockTensor([1.0, 2.0, 3.0])
        right = MockTensor([0.5, -1.0, 4.0])

        summed = left + right
        scaled = summed.scale(0.5)

        self.assertEqual(summed.values, (1.5, 1.0, 7.0))
        self.assertEqual(scaled.values, (0.75, 0.5, 3.5))

    def test_parameter_server_average(self) -> None:
        grads = [
            MockTensor([1.0, 2.0, 3.0, 4.0]),
            MockTensor([2.0, 4.0, 6.0, 8.0]),
            MockTensor([4.0, 8.0, 12.0, 16.0]),
        ]

        averaged = parameter_server_average(grads)
        expected = MockTensor([7.0 / 3.0, 14.0 / 3.0, 7.0, 28.0 / 3.0])
        self.assertTrue(averaged.almost_equal(expected))

    def test_parameter_server_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            parameter_server_average([])

    def test_parameter_server_rejects_shape_mismatch(self) -> None:
        grads = [MockTensor([1.0, 2.0]), MockTensor([1.0])]

        with self.assertRaises(ValueError):
            parameter_server_average(grads)


if __name__ == "__main__":
    unittest.main()
