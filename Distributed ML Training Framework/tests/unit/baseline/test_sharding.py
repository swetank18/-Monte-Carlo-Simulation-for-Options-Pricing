import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dmlf.baseline.trainer import _rank_shard_indices


class ShardIndexTests(unittest.TestCase):
    def test_shards_are_disjoint_and_cover_dataset(self) -> None:
        dataset_size = 23
        world_size = 4

        shards = [set(_rank_shard_indices(dataset_size, rank, world_size)) for rank in range(world_size)]

        for i in range(world_size):
            for j in range(i + 1, world_size):
                self.assertEqual(len(shards[i].intersection(shards[j])), 0)

        union = set().union(*shards)
        self.assertEqual(union, set(range(dataset_size)))

    def test_invalid_rank_raises(self) -> None:
        with self.assertRaises(ValueError):
            _rank_shard_indices(dataset_size=10, rank=3, world_size=3)


if __name__ == "__main__":
    unittest.main()
