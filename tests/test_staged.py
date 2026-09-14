import numpy as np

from graph_tuning_infra.database import Stage
from graph_tuning_infra.files import write_json
from graphspec.staged import IDS, Staged
from graphspec.synthetic import Counter, TARGET, graph_rows
from graphspec.truth import ARRAYS, Graph
from tuning.build import build

SETTINGS = dict(entities=40, steps=12, features=8, signals=2, degree=2.0, max_degree=6, alive=0.8, lifetime=4.0, period=3)


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
    directories = [root / name / ('shard_%02d' % shard) for shard in range(shards)]
    stages = [Stage(directory) for directory in directories]
    counter = Counter()
    for index in range(4):
        stages[index * shards // 4].write(graph_rows(rng, index, 'graph_%03d' % index, SETTINGS, Based(index) if based else counter, 'query'))
    for stage in stages:
        stage.close()
    Stage.merge(root / name, directories)
    task = root / (name + '.task.json')
    write_json(task, dict(name=name, database=name + '.lbug', staging=name, target=TARGET))
    return task


def test_sharded_staging_with_block_ids_builds_the_same_bank(tmp_path):
    reference, sharded = tmp_path / 'bank_dense', tmp_path / 'bank_based'
    build(staging(tmp_path, 'dense', 1, False), reference, 1)
    build(staging(tmp_path, 'based', 2, True), sharded, 1)
    source = Staged(tmp_path / 'based')
    assert source.based >= {'observation'} and 'observation' not in source.dense and len(source.by_table['observation'][0].files) == 2
    for index in range(4):
        left, right = Graph.load(reference / 'truth' / ('graph_%04d' % index)), Graph.load(sharded / 'truth' / ('graph_%04d' % index))
        assert left.name == right.name and left.steps == right.steps
        for name in ARRAYS:
            if name in ('query_key', 'label_id'):
                continue
            assert np.array_equal(left.arrays[name], right.arrays[name], equal_nan=True), name
