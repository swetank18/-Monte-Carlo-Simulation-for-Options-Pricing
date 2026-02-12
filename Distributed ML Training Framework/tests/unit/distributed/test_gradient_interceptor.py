import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dmlf.distributed.gradient_interceptor import GradientInterceptor
from dmlf.env import DependencyError, require_torch


class GradientInterceptorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.torch = require_torch("running gradient interception tests")
            cls.nn = cls.torch.nn
        except DependencyError as exc:
            raise unittest.SkipTest(str(exc))

    def test_intercepts_gradients_for_each_parameter(self) -> None:
        model = self.nn.Sequential(self.nn.Linear(4, 3), self.nn.ReLU(), self.nn.Linear(3, 2))
        interceptor = GradientInterceptor(clone=True, move_to_cpu=True)
        interceptor.attach(model)

        x = self.torch.randn(5, 4)
        y = self.torch.randn(5, 2)
        criterion = self.nn.MSELoss()

        interceptor.start_backward_capture()
        loss = criterion(model(x), y)
        loss.backward()
        snapshot = interceptor.finish_backward_capture()

        self.assertEqual(snapshot.backward_index, 1)
        self.assertEqual(len(snapshot.missing_parameters), 0)

        for name, parameter in model.named_parameters():
            self.assertIn(name, snapshot.gradients)
            captured = snapshot.gradients[name]
            self.assertIsNotNone(captured)
            self.assertEqual(captured.shape, tuple(parameter.grad.shape))
            self.assertEqual(captured.device, "cpu")
            self.assertTrue(self.torch.allclose(captured.gradient, parameter.grad.cpu()))

    def test_marks_unused_parameters(self) -> None:
        class PartialUseModel(self.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.used = self.nn.Linear(4, 2)
                self.unused = self.nn.Linear(4, 2)

            def forward(self, inputs):
                return self.used(inputs)

        model = PartialUseModel()
        interceptor = GradientInterceptor()
        interceptor.attach(model)

        x = self.torch.randn(3, 4)
        y = self.torch.randn(3, 2)
        criterion = self.nn.MSELoss()

        interceptor.start_backward_capture()
        loss = criterion(model(x), y)
        loss.backward()
        snapshot = interceptor.finish_backward_capture()

        self.assertIn("unused.weight", snapshot.missing_parameters)
        self.assertIn("unused.bias", snapshot.missing_parameters)
        self.assertIsNone(snapshot.gradients["unused.weight"])
        self.assertIsNone(snapshot.gradients["unused.bias"])
        self.assertIsNotNone(snapshot.gradients["used.weight"])
        self.assertIsNotNone(snapshot.gradients["used.bias"])


if __name__ == "__main__":
    unittest.main()
