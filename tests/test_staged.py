import numpy as np

from graphspec.build.output import Stage
from graphspec.build.staged import IDS, Staged
from graphspec.facts.stored import STORED, Facts
from graphspec.infra.files import write_json
from graphspec.schema.schema import Schema
from graphspec.sources.synthetic import Counter, graph_rows, uniform
from tuning.build import build

SETTINGS = dict(entities=40, steps=12, degree=2.0, max_degree=6, alive=0.8, lifetime=4.0, period=3)
SHAPE = uniform(4, 2, "task")


class Based(Counter):
    def __init__(self, index):
        super().__init__()
        self.next = {}
        self.base = index << IDS

    def take(self, table, count):
        start = self.next.get(table, self.base)
        self.next[table] = start + count
        return np.arange(start, start + count, dtype=np.int64)


def staging(root, name, shards, based):
    rng = np.random.default_rng(7)
    directories = [root / name / ("shard_%02d" % shard) for shard in range(shards)]
    stages = [Stage(directory) for directory in directories]
    counter = Counter()
    for index in range(4):
        stages[index * shards // 4].write(
            graph_rows(rng, index, "graph_%03d" % index, SHAPE, SETTINGS, Based(index) if based else counter)
        )
    for stage in stages:
        stage.close()
    Stage.merge(root / name, directories)
    task = root / (name + ".task.json")
    write_json(
        task,
        dict(name=name, database=name + ".lbug", staging=name, schema=SHAPE["schema"]),
    )
    return task


def test_sharded_staging_with_block_ids_builds_the_same_bank(tmp_path):
    reference, sharded = tmp_path / "bank_dense", tmp_path / "bank_based"
    build(staging(tmp_path, "dense", 1, False), reference, 1, ("train",))
    build(staging(tmp_path, "based", 2, True), sharded, 1, ("train",))
    source = Staged(tmp_path / "based", Schema(SHAPE["schema"]))
    assert (
        source.based >= {"observation"}
        and "observation" not in source.dense
        and len(source.by_table["observation"][0].files) == 2
    )
    for index in range(4):
        left, right = (
            Facts.load(reference / "truth" / ("graph_%04d" % index)),
            Facts.load(sharded / "truth" / ("graph_%04d" % index)),
        )
        assert left.name == right.name and left.unit == right.unit
        for name in STORED:
            if name == "task_key":
                continue
            assert np.array_equal(left.arrays[name], right.arrays[name], equal_nan=True), name
