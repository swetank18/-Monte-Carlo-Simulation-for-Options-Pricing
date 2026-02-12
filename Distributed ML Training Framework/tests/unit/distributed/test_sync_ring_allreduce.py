import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dmlf.distributed.ring_allreduce_sync import (
    run_synchronous_ring_allreduce_training,
    serial_ring_reference,
)


class SynchronousRingAllReduceTests(unittest.TestCase):
    def test_ring_allreduce_matches_single_node_reference(self) -> None:
        world_size = 4
        num_steps = 6
        learning_rate = 0.04
        parameter_dim = 8
        shard_size = 5

        result = run_synchronous_ring_allreduce_training(
            world_size=world_size,
            num_steps=num_steps,
            learning_rate=learning_rate,
            parameter_dim=parameter_dim,
            shard_size=shard_size,
        )
        expected = serial_ring_reference(
            world_size=world_size,
            num_steps=num_steps,
            learning_rate=learning_rate,
            parameter_dim=parameter_dim,
            shard_size=shard_size,
        )

        self.assertTrue(result.final_parameters.almost_equal(expected, tol=1e-9))
        self.assertEqual(len(result.worker_reports), world_size)

        for worker in result.worker_reports:
            self.assertEqual(len(worker.gradient_history), num_steps)
            self.assertEqual(len(worker.parameter_history), num_steps)
            for step in range(num_steps):
                self.assertTrue(
                    worker.parameter_history[step].almost_equal(
                        result.worker_reports[0].parameter_history[step], tol=1e-9
                    )
                )

    def test_parameter_dim_must_be_divisible_by_world_size(self) -> None:
        with self.assertRaises(ValueError):
            run_synchronous_ring_allreduce_training(
                world_size=3,
                num_steps=2,
                learning_rate=0.1,
                parameter_dim=10,
                shard_size=4,
            )


if __name__ == "__main__":
    unittest.main()
