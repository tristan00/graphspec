import pytest

from graphspec.synthetic import generate
from tuning.build import build


@pytest.fixture(scope='session')
def bank(tmp_path_factory):
    root = tmp_path_factory.mktemp('bank')
    task = generate(root, 'tiny', 4, 'query', 1, entities=40, steps=12, features=8, signals=2, degree=2.0, max_degree=6, alive=0.8, lifetime=4.0, period=3)
    build(task, root / 'bank', 1)
    return root / 'bank'
