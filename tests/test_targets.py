import copy
import time

import numpy as np
import pytest

from conftest import SPLITS
from graphspec.schema.schema import Schema
from graphspec.schema.targets import TARGETS, UTILITIES, valued
from graphspec.sources import synthetic
from tuning.fits.training import fit
from tuning.statement import state

pytestmark = pytest.mark.xdist_group("targets")


def restated(**target):
    held = copy.deepcopy(synthetic.varied("task")["schema"])
    held["tasks"]["kinds"]["buy"]["target"] = dict(held["tasks"]["kinds"]["buy"]["target"], **target)
    return held


def test_kinds_of_target_and_utilities_are_registered_and_a_target_names_them():
    assert TARGETS.names()[:2] == ("number", "path") and UTILITIES.names()[0] == "weighted"
    schema = Schema(restated())
    assert schema.tasks["kinds"]["buy"]["target"]["utility"] == dict(
        name="weighted", better="higher", weights=dict(outcome=1.0)
    )
    assert schema.targets() == ("number",) and schema.target_columns() == {}
    assert Schema(synthetic.varied("path")["schema"]).target_columns() == dict(tau="FLOAT", series="STRING")
    with pytest.raises(KeyError, match="no kind of target is registered under the name 'ranking'"):
        Schema(restated(kind="ranking"))
    with pytest.raises(ValueError, match="the target of the kind of task buy names its one series without colons"):
        Schema(restated(series=["outcome"]))
    with pytest.raises(KeyError, match="no utility is registered under the name 'profit'"):
        Schema(restated(utility=dict(name="profit", better="higher")))
    with pytest.raises(ValueError, match="states which value is better as one of higher, lower, found 'more'"):
        Schema(restated(utility=dict(name="weighted", better="more", weights=dict(outcome=1.0))))
    with pytest.raises(ValueError, match="weighs every series of its target, outcome, by a finite number"):
        Schema(restated(utility=dict(name="weighted", better="higher", weights=dict(other=1.0))))
    with pytest.raises(ValueError, match="holds fields it does not know: rate"):
        Schema(restated(utility=dict(name="weighted", better="lower", weights=dict(outcome=1.0), rate=2)))


def test_the_utility_of_a_target_is_one_number_per_task():
    path = Schema(synthetic.varied("path")["schema"]).tasks["kinds"]["buy"]["target"]
    stated = dict(path, utility=dict(path["utility"], weights=dict(level=2.0, trend=-1.0)))
    points = dict(task=[4, 4, 9, 4], series=["level", "trend", "level", "level"], y=[1.0, 3.0, 5.0, 0.5])
    tasks, values = valued(stated, points)
    assert tasks.tolist() == [4, 9] and values.tolist() == [2.0 - 3.0 + 1.0, 10.0]
    number = Schema(restated()).tasks["kinds"]["buy"]["target"]
    tasks, values = valued(number, dict(task=[1, 0], series=["outcome", "outcome"], y=[2.5, -1.0]))
    assert tasks.tolist() == [0, 1] and np.array_equal(values, [-1.0, 2.5])


def test_a_statement_naming_what_cannot_read_a_kind_of_target_of_the_bank_stops_naming_both(
    path_bank, tmp_path, statement
):
    config = dict(seed=statement["seed"], splits=SPLITS, pipeline=[])
    with pytest.raises(
        ValueError,
        match="names the measure r2, which reads targets of the kinds number and cannot read the kind of target path",
    ):
        fit(path_bank, tmp_path, config, [], 8, time.perf_counter() + 30, state(statement))
    with pytest.raises(KeyError, match="no loss is registered under the name 'absolute'"):
        fit(path_bank, tmp_path, config, [], 8, time.perf_counter() + 30, state(dict(statement, loss="absolute")))
    assert not list(tmp_path.iterdir())
