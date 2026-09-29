import copy
import logging
import time

import numpy as np
import pytest

from graphspec import synthetic
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.infra.timing import timed
from graphspec.schema import VERSION, Schema
from graphspec.truth import Graph
from tuning.build import build
from tuning.data import Deriver
from tuning.model_spec import configuration

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("leakage")
MAKERS, SITES = tuple("m%d" % number for number in range(6)), tuple("s%d" % number for number in range(6))
SPLITS = dict(
    trained=["alpha"], scored=["omega"], sees=dict(alpha=["without_scored", "without_targets"], omega=[])
)
CHANGES = {"time/steps": "changes"}
ASKED = (
    (2, "near", "alpha", "w0", "m0", 1.0, 1.0, 3),
    (2, "near", "omega", "w1", "m0", 10.0, 777.0, 3),
    (4, "flow", "alpha", "w0", "m1", 2.0, 2.0, None),
    (4, "near", "omega", "w1", "m5", 50.0, 5.0, 5),
    (6, "near", "alpha", "w0", "m0", 3.0, 3.0, 7),
    (6, "flow", "alpha", "w1", "m2", 4.0, 4.0, None),
    (6, "flow", "omega", "w1", "m3", 60.0, 6.0, None),
)
STAMPED = ("since", "until", "known", "replaced", "observers")
HOLDS = dict(nodes=dict(part=dict(columns=["grain"])), edges={}, encoder="pooled")


@timed("leakage.stated")
def stated():
    standard = dict(type="number", encoder="standard")
    kinds = {
        kind: dict(
            references=["unit", "link"],
            columns=["ask"],
            target=dict(kind="number", series=kind, about=about),
            weight=False,
            absent="stop",
            event=None,
            matures=dict(reference="unit", column="mark") if kind == "near" else None,
        )
        for kind, about in (("near", "unit"), ("flow", "link"))
    }
    return dict(
        version=VERSION,
        time=dict(model="clock", units=["tick"], ties="key"),
        observers=True,
        columns=dict(
            size=standard,
            shade=dict(type="category", encoder="onehot"),
            mark=standard,
            load=standard,
            weight=standard,
            ask=standard,
            grain=standard,
            era=dict(type="category", encoder="onehot"),
        ),
        graph=dict(columns=["era"]),
        nodes=dict(
            maker=dict(columns=["size", "shade", "mark"], holds=HOLDS), site=dict(columns=["load"], holds=None)
        ),
        sets={},
        edges=dict(
            serves=synthetic.edge("maker", "site", ["weight"]),
            beside=synthetic.edge("site", "site", [], directed=False),
        ),
        tasks=dict(
            key="ticket",
            grouping=None,
            signals=0,
            references=dict(unit=dict(sort="node", kinds=["maker"]), link=dict(sort="edge", kinds=["maker"])),
            kinds=kinds,
        ),
    )


@timed("leakage.graph")
def graph(name, asked, hidden):
    serves = (MAKERS[:2] + MAKERS[1:], SITES[:2] + SITES[1:])
    values = [
        dict(
            form="state",
            parameters={},
            kind="maker",
            rows=dict(
                key=list(MAKERS),
                since=[0] * 6,
                until=[None] * 6,
                size=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0 * hidden],
                shade=["red", "blue", "red", "blue", "red", "rare" if hidden > 1 else "blue"],
                mark=[0.5 * number for number in range(6)],
            ),
        ),
        dict(
            form="state",
            parameters={},
            kind="site",
            rows=dict(
                key=list(SITES), since=[0] * 6, until=[None] * 6, load=[2.0 + number for number in range(6)]
            ),
        ),
    ]
    edges = [
        dict(
            form="state",
            parameters={},
            table="serves",
            ends=("maker", "site"),
            rows=dict(
                source=list(serves[0]),
                target=list(serves[1]),
                present=[True] * 7,
                since=[0, 0, 4, 0, 0, 0, 0],
                until=[None, 4, None, None, None, None, None],
                weight=[1.0, 2.0, 7.0 * hidden, 3.0, 4.0, 5.0, 6.0 * hidden],
            ),
        ),
        dict(
            form="state",
            parameters={},
            table="beside",
            ends=("site", "site"),
            rows=dict(
                source=["s0", "s2"], target=["s1", "s3"], present=[True] * 2, since=[0, 0], until=[None] * 2
            ),
        ),
    ]
    scored = [entry[2] == "omega" for entry in asked]
    tasks = dict(
        key=list(range(len(asked))),
        kind=[entry[1] for entry in asked],
        time=[entry[0] for entry in asked],
        split=[entry[2] for entry in asked],
        observer=[entry[3] for entry in asked],
        weight=[None] * len(asked),
        ask=[entry[5] * (hidden if far else 1) for entry, far in zip(asked, scored)],
        references=dict(
            unit=[entry[4] for entry in asked],
            link=[(entry[4], "serves", SITES[MAKERS.index(entry[4])]) for entry in asked],
        ),
        target=[entry[6] * (hidden if far else 1) for entry, far in zip(asked, scored)],
        known=[entry[7] for entry in asked],
    )
    parts = dict(owner=[], kind=[], key=[], grain=[])
    for number, maker in enumerate(MAKERS):
        for at in range(2):
            parts["owner"].append(maker)
            parts["kind"].append("part")
            parts["key"].append("p%d" % at)
            parts["grain"].append((1.0 + number + at) * (hidden if maker == MAKERS[-1] else 1))
    return dict(
        name=name,
        unit="tick",
        columns=dict(era="late" if asked == () and hidden > 1 else "early"),
        nodes=dict(maker=list(MAKERS), site=list(SITES)),
        values=values,
        edges=edges,
        held=dict(maker=dict(nodes=parts, edges=dict(owner=[], source=[], target=[], relation=[]))),
        tasks=tasks,
    )


@pytest.fixture(scope="module")
def banks(tmp_path_factory):
    started = time.perf_counter()
    root, result = tmp_path_factory.mktemp("leakage"), {}
    for name, hidden in (("seen", 1), ("hidden", 10)):
        case = dict(schema=stated(), graphs=[graph("g0", ASKED, hidden), graph("g1", (), hidden)])
        build(synthetic.stage(root, name, case), root / name, 1, ("alpha",))
        result[name] = root / name
    LOG.info("banks of leakage exit seconds=%.3f", time.perf_counter() - started)
    return result


@timed("leakage.opened")
def opened(bank, **values):
    held, given = Deriver(bank, "cpu", RESIDENT), dict(CHANGES, **values)
    return held, configuration(held.registry, dict(view="snapshot"), given, 5, copy.deepcopy(SPLITS))


@timed("leakage.views")
def views(held, cfg):
    return {
        (entry["source"], entry["observer"], entry["split"]): Graph.load(held.view(entry["index"], cfg))
        for entry in held.viewed(cfg).graphs
    }


@timed("leakage.column")
def column(held, view, key, name):
    dense = view.dense()
    rows = dense.obs_entity == view.keys.index(key)
    values = np.asarray(view.obs_values)[dense.obs_ref[rows], held.registry.encoding.layout.index(name)]
    return dict(zip(dense.obs_step[rows].tolist(), values.tolist()))


@timed("leakage.serving")
def serving(held, view, pair):
    dense = view.dense()
    rows = dense.fact_present & (dense.fact_code == held.registry.code("serves"))
    rows &= (dense.fact_source == view.keys.index(pair[0])) & (dense.fact_target == view.keys.index(pair[1]))
    return sorted(set(dense.fact_step[rows].tolist()))


def test_encoders_are_fitted_on_what_the_views_of_the_splits_trained_on_show_and_on_no_other(banks):
    found = {}
    for name, bank in banks.items():
        held, cfg = opened(bank)
        found[name] = (held, cfg, views(held, cfg), read_json(held.viewed(cfg).folder / "encoders.json"))
    seen, hidden = found["seen"], found["hidden"]
    assert seen[0].manifest["spec"]["encoding"]["states"] is None
    assert seen[0].registry.encoding.dictionaries["shade"] == ["blue", "red"]
    assert hidden[0].registry.encoding.dictionaries["shade"] == ["blue", "rare", "red"]
    assert seen[3] == hidden[3] and seen[3]["states"]["shade"] == dict(values=["blue", "red"])
    assert seen[3]["states"]["size"]["mean"] == pytest.approx(2.75)
    assert seen[3]["states"]["ask"]["mean"] == pytest.approx(2.5)
    alpha, stated = seen[0].splits.index("alpha"), ("obs_values", "task_values")
    trained = [key for key in seen[2] if key[2] == alpha]
    assert sorted(trained) == [(0, 0, alpha), (0, 1, alpha)] and sorted(hidden[2]) == sorted(seen[2])
    for key in trained:
        left, right = seen[2][key], hidden[2][key]
        for part in (name for name in left.arrays if name not in stated):
            assert np.array_equal(left.arrays[part], right.arrays[part], equal_nan=True), (key, part)
    far = [key for key in seen[2] if key[2] < 0 and key[0] == 0]
    assert far and any(not np.array_equal(seen[2][key].obs_x, hidden[2][key].obs_x) for key in far)
    assert seen[0].fitted(seen[1]).features == hidden[0].fitted(hidden[1]).features


def test_held_graphs_and_graph_attributes_are_fitted_on_what_the_trained_views_show_and_on_no_other(banks):
    found = {}
    for name, bank in banks.items():
        held, cfg = opened(bank)
        found[name] = (held, views(held, cfg), read_json(held.viewed(cfg).folder / "encoders.json")["states"])
    seen, hidden = found["seen"], found["hidden"]
    assert hidden[0].registry.encoding.dictionaries["era"] == ["early", "late"]
    assert seen[2]["era"] == hidden[2]["era"] == dict(values=["early"])
    assert seen[2]["grain"] == hidden[2]["grain"] and seen[2]["grain"]["mean"] == pytest.approx(3.25)
    alpha = seen[0].splits.index("alpha")
    for key in (key for key in seen[1] if key[2] == alpha):
        for part in ("obs_x", "entity_held", "graph_values"):
            assert np.array_equal(seen[1][key].arrays[part], hidden[1][key].arrays[part]), (key, part)
    far = seen[1][(0, 1, -1)].keys.index("m5")
    assert not np.array_equal(seen[1][(0, 1, -1)].entity_held[far], hidden[1][(0, 1, -1)].entity_held[far])
    assert seen[1][(1, -1, -1)].graph_values.tolist() != hidden[1][(1, -1, -1)].graph_values.tolist()


def test_a_matured_target_is_shown_to_its_observer_and_never_from_a_split_scored_on_to_one_trained_on(banks):
    held, cfg = opened(banks["seen"])
    shown = views(held, cfg)
    alpha, moments = held.splits.index("alpha"), [0, 2, 4, 6]
    assert column(held, shown[(0, 0, alpha)], "m0", "mark") == {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0}
    assert column(held, shown[(0, 1, alpha)], "m0", "mark") == {step: 0.0 for step in range(len(moments))}
    assert column(held, shown[(0, 1, -1)], "m0", "mark") == {0: 0.0, 1: 0.0, 2: 777.0, 3: 777.0}
    assert column(held, shown[(0, 1, -1)], "m5", "mark")[3] == 5.0
    assert (0, 0, -1) not in shown


def test_without_targets_keeps_the_edges_a_target_is_about_from_the_time_of_its_task(banks):
    held, cfg = opened(banks["seen"])
    shown = views(held, cfg)
    alpha = held.splits.index("alpha")
    assert serving(held, shown[(0, 0, alpha)], ("m0", "s0")) == [0, 1, 2, 3]
    assert serving(held, shown[(0, 0, alpha)], ("m1", "s1")) == [0, 1]
    assert serving(held, shown[(0, 0, alpha)], ("m2", "s2")) == [0, 1, 2]
    assert serving(held, shown[(0, 0, alpha)], ("m3", "s3")) == []
    assert serving(held, shown[(0, 1, -1)], ("m1", "s1")) == [0, 1, 2, 3]
    mapping = stated()
    mapping["tasks"]["kinds"]["flow"]["target"]["about"] = "place"
    with pytest.raises(ValueError, match="the target of the kind of task flow is about one of its"):
        Schema(mapping)
    del mapping["tasks"]["kinds"]["flow"]["target"]["about"]
    with pytest.raises(ValueError, match="the target of the kind of task flow lacks about"):
        Schema(mapping)


def test_a_graph_without_tasks_builds_and_a_choice_of_steps_that_cannot_show_it_stops_naming_both(banks):
    held, cfg = opened(banks["seen"])
    assert [graph["name"] for graph in held.manifest["graphs"]] == ["g0", "g1"]
    (context,) = [entry for entry in held.viewed(cfg).graphs if entry["source"] == 1]
    assert context["observer"] == -1 and context["labels"] == 0 and context["steps"] == 2
    _, tasks = opened(banks["seen"], **{"time/steps": "tasks"})
    with pytest.raises(ValueError, match="graph g1 holds no task, and the choice of steps tasks shows"):
        held.viewed(tasks)


def test_a_search_offers_a_choice_of_steps_only_where_it_shows_every_graph_of_the_bank(banks):
    from tuning.model_spec import configured
    from tuning.selector import check_bounds, graph_space, random_suggest
    from tuning.space import Space

    registry = Deriver(banks["seen"], "cpu", RESIDENT).registry
    found = {p.name: p for p in graph_space(registry, configured(registry, dict(view="snapshot")), {})}
    assert found["time/steps"].options == ("changes",)
    assert Space(registry, {}, 600, []).proposal_parameters["graph/time/steps"].options == ("changes",)
    assert random_suggest(registry, 3, 3, {}, 600)[0]["graph"]["time/steps"] == "changes"
    assert registry.extras["untasked"] == ["g1"]
    with pytest.raises(
        ValueError,
        match="the bank seen holds a graph that no choice of steps among tasks shows: tasks shows a graph at "
        "the times of its tasks, and these graphs of the bank hold no task: g1",
    ):
        check_bounds(registry, {"time/steps": "tasks"})
