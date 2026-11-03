import copy

import numpy as np
import pytest

from graphspec.facts.stored import Facts
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.timing import timed
from graphspec.schema.schema import FACT_COLUMNS, VERSION
from graphspec.sources import synthetic
from graphspec.views.view import sequences, view
from graphspec.views.visible import unstated
from tuning.bank.bank import Deriver
from tuning.bank.build import build

pytestmark = [pytest.mark.xdist_group("partial_facts"), pytest.mark.extended]


@timed("partial_facts.fact")
def fact(since, until, values, **stamps):
    return dict(kind="node", form="state", parameters={},
                rows=dict(key=["n"], since=[since], until=[until], **values, **stamps))


@timed("partial_facts.case")
def case():
    value = dict(type="number", encoder="value")
    schema = dict(
        version=VERSION, time=dict(model="clocks", units=["tick"], ties=None), observers=True,
        columns=dict(a=value, b=value, c=value), graph=dict(columns=[]),
        nodes=dict(node=dict(columns=["a", "b", "c"], holds=None)), sets={}, edges={},
        tasks=dict(key="ticket", grouping=None, signals=0,
                   references=dict(at=dict(sort="node", kinds=["node"])),
                   kinds=dict(level=dict(references=["at"], columns=[], weight=False, absent="stop",
                                         event=None, matures=None,
                                         target=dict(kind="number", series="level", about=None,
                                                     utility=synthetic.valued(["level"]))))),
    )
    times = [time for time in range(1, 5) for _ in range(2)]
    graph = dict(
        name="partial", unit="tick", columns={}, nodes=dict(node=["n"]), edges=[],
        values=[fact(0, None, dict(a=[1])), fact(0, 3, dict(b=[2])),
                fact(3, None, dict(b=[None])), fact(0, None, dict(c=[9]), observers=[["bob"]])],
        tasks=dict(key=list(range(8)), kind=["level"] * 8, time=times, observer=["alice", "bob"] * 4,
                   split=["train" if time < 3 else "validation" for time in times], weight=[None] * 8,
                   references=dict(at=["n"] * 8), target=[float(at) for at in range(8)], known=[None] * 8),
    )
    return dict(schema=schema, graphs=[graph])


@timed("partial_facts.built")
def built(root, name, given):
    path = root / name
    build(synthetic.stage(root, name + "-source", given), path, 1, ("train",))
    return Deriver(path, "cpu", 7, RESIDENT)


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    return built(tmp_path_factory.mktemp("partial"), "bank", case())


@timed("partial_facts.shown")
def shown(bank, several):
    truth = Facts.load(bank.truth(0))
    result = {}
    for sequence in sequences(truth, bank.registry, "tasks", unstated(bank.splits)):
        graph = view(truth, bank.registry, sequence, several).dense()
        result[truth.observers[sequence["observer"]]] = np.asarray(graph.x_table)[np.asarray(graph.obs_ref)]
    return result


def test_disjoint_facts_combine_without_ties_or_a_selection_rule_and_null_stays_missing(bank):
    assert not bank.registry.extras["several"]
    values = shown(bank, None)
    for observer, rows in values.items():
        assert rows[:, 0].tolist() == [1] * 4
        np.testing.assert_equal(rows[:, 1], [2, 2, np.nan, np.nan])
        np.testing.assert_equal(rows[:, 2], [9] * 4 if observer == "bob" else [np.nan] * 4)
    truth = Facts.load(bank.truth(0))
    declared = truth.stated(bank.registry, bank.splits, truth.observers)["observations"]["node"]
    assert sorted(declared[FACT_COLUMNS]) == [["a"], ["b"], ["b"], ["c"]]


def test_selection_applies_per_column_without_discarding_independent_facts(tmp_path):
    given = case()
    given["graphs"][0]["values"][1] = fact(0, None, dict(b=[2]))
    held = built(tmp_path, "overlapping", given)
    assert held.registry.extras["several"]
    with pytest.raises(ValueError, match="time/several"):
        shown(held, None)
    first, last = shown(held, "first"), shown(held, "last")
    for observer in first:
        assert first[observer][:, 0].tolist() == last[observer][:, 0].tolist() == [1] * 4
        assert first[observer][:, 1].tolist() == [2] * 4
        np.testing.assert_equal(last[observer][:, 1], [2, 2, np.nan, np.nan])


def test_a_late_column_and_its_correction_do_not_change_earlier_values(tmp_path):
    given = case()
    given["graphs"][0]["values"][1:3] = [
        fact(0, None, dict(b=[2]), known=[2], replaced=[4]),
        fact(0, None, dict(b=[7]), known=[4], replaced=[None]),
    ]
    held = built(tmp_path, "corrections", given)
    for rows in shown(held, "last").values():
        np.testing.assert_equal(rows[:, 1], [np.nan, 2, 2, 7])
        assert rows[:, 0].tolist() == [1] * 4
    wrong = copy.deepcopy(given)
    wrong["graphs"][0]["values"][2]["rows"]["known"] = [3]
    with pytest.raises(ValueError, match="two facts.*column b"):
        built(tmp_path, "untied", wrong)
