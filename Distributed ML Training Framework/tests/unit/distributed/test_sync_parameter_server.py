import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dmlf.distributed.mock_tensor import MockTensor
from dmlf.distributed.parameter_server_sync import (
    GradientMessage,
    load_latest_checkpoint,
    run_synchronous_parameter_server_training,
    serial_synchronous_reference,
    _validate_gradient_batch,
)


class SynchronousParameterServerTests(unittest.TestCase):
    def test_sync_parameter_server_matches_serial_reference(self) -> None:
        world_size = 3
        num_steps = 5
        learning_rate = 0.05
        parameter_dim = 4
        shard_size = 6

        result = run_synchronous_parameter_server_training(
            world_size=world_size,
            num_steps=num_steps,
            learning_rate=learning_rate,
            parameter_dim=parameter_dim,
            shard_size=shard_size,
        )
        expected = serial_synchronous_reference(
            world_size=world_size,
            num_steps=num_steps,
            learning_rate=learning_rate,
            parameter_dim=parameter_dim,
            shard_size=shard_size,
        )

        self.assertTrue(result.final_parameters.almost_equal(expected, tol=1e-9))
        self.assertEqual(len(result.server_parameter_history), num_steps + 1)
        self.assertEqual(len(result.worker_reports), world_size)

        for worker in result.worker_reports:
            self.assertEqual(len(worker.gradient_history), num_steps)
            self.assertEqual(len(worker.parameter_history), num_steps)
            for step in range(num_steps):
                self.assertTrue(
                    worker.parameter_history[step].almost_equal(
                        result.server_parameter_history[step + 1], tol=1e-9
                    )
                )

    def test_recovers_after_worker_failure_from_checkpoint(self) -> None:
        world_size = 3
        num_steps = 6
        learning_rate = 0.05
        parameter_dim = 4
        shard_size = 6

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            result = run_synchronous_parameter_server_training(
                world_size=world_size,
                num_steps=num_steps,
                learning_rate=learning_rate,
                parameter_dim=parameter_dim,
                shard_size=shard_size,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval_steps=1,
                max_restarts=2,
                inject_failure_rank=1,
                inject_failure_step=3,
            )

            expected = serial_synchronous_reference(
                world_size=world_size,
                num_steps=num_steps,
                learning_rate=learning_rate,
                parameter_dim=parameter_dim,
                shard_size=shard_size,
            )

            self.assertTrue(result.final_parameters.almost_equal(expected, tol=1e-9))
            self.assertEqual(result.restart_count, 1)
            self.assertIsNotNone(result.last_checkpoint_path)

    def test_corrupted_checkpoint_is_rejected(self) -> None:
        world_size = 2
        num_steps = 3
        learning_rate = 0.1
        parameter_dim = 4
        shard_size = 3

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            run_synchronous_parameter_server_training(
                world_size=world_size,
                num_steps=num_steps,
                learning_rate=learning_rate,
                parameter_dim=parameter_dim,
                shard_size=shard_size,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval_steps=1,
            )

            latest = sorted(Path(checkpoint_dir).glob("checkpoint_step_*.json"))[-1]
            with latest.open("r", encoding="utf-8") as f:
                record = json.load(f)

            record["payload"]["parameters"][0] += 1.0
            with latest.open("w", encoding="utf-8") as f:
                json.dump(record, f)

            with self.assertRaises(RuntimeError):
                load_latest_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    world_size=world_size,
                    parameter_dim=parameter_dim,
                    shard_size=shard_size,
                    learning_rate=learning_rate,
                )

    def test_validate_gradient_batch_rejects_duplicate_rank(self) -> None:
        messages = [
            GradientMessage(worker_rank=0, step=2, gradient=MockTensor([1.0, 2.0])),
            GradientMessage(worker_rank=0, step=2, gradient=MockTensor([3.0, 4.0])),
        ]

        with self.assertRaises(RuntimeError):
            _validate_gradient_batch(messages, expected_step=2, world_size=2)

    def test_validate_gradient_batch_rejects_step_mismatch(self) -> None:
        messages = [
            GradientMessage(worker_rank=0, step=1, gradient=MockTensor([1.0])),
            GradientMessage(worker_rank=1, step=2, gradient=MockTensor([2.0])),
        ]

        with self.assertRaises(RuntimeError):
            _validate_gradient_batch(messages, expected_step=1, world_size=2)


if __name__ == "__main__":
    unittest.main()
