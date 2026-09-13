import json

import pytest

from tuning.build import build

from .test_bank import database


@pytest.fixture(scope='session')
def book(tmp_path_factory):
    root = tmp_path_factory.mktemp('book')
    database(root / 'book.lbug')
    (root / 'book.task.json').write_text(json.dumps(dict(name='book', database='book.lbug', target='label')))
    build(root / 'book.task.json', root / 'bank', 1)
    return root / 'bank'
