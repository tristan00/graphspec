import copy
import logging
import time

import numpy as np
import pytest
import support

from conftest import inline
from graphspec import OPERATIONS, SEES, SORTS
from graphspec.budgets import GraphLimits
from graphspec.engine import segments, transform
from graphspec.facts.stored import Facts
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.infra.timing import timed
from graphspec.schema.references import Sort
from graphspec.schema.schema import VERSION, Schema
from graphspec.schema.time import ALWAYS, NEVER
from graphspec.sources import synthetic
from graphspec.views.graph import Graph
from graphspec.views.visible import Looking
from tuning.bank.bank import Deriver
from tuning.bank.build import build
from tuning.families.model_spec import VIEWS, views
from tuning.fits.training import fit
from tuning.statement import state

LOG = logging.getLogger(__name__)
configuration = support.configuration
pytestmark = pytest.mark.xdist_group("tasks")
MAKERS, SITES = ("m0", "m1", "m2", "m3"), ("s0", "s1", "s2", "s3")
SERVES = (("m0", "s0"), ("m1", "s1"), ("m2", "s2"), ("m3", "s3"), ("m0", "s1"))
BESIDE = (("s0", "s1"), ("s2", "s3"))
NAMES = ("north", "south", "east", "west")
SPLITS = dict(
    trained=["north", "south"],
    scored=["east", "west"],
    sees=dict(north=["without_targets"], south=["without_scored"], east=["without_beside"], west=[]),
)
STEPS = [0, 2, 3, 4, 6, 8, 9, 10]
LINK = ("m0", "serves", "s0")
ROUTE = [("m0", "serves", "s1"), ("s0", "beside", "s1")]
PART = dict(nodes=["m1"], edges=[("m1", "serves", "s1")])
WIDE = dict(unit="m1", link=LINK, group=["s1", "s0"], route=ROUTE, part=PART)
ASKED = (
    (2, "total", "north", {}),
    (2, "mark", "north", dict(unit="m0")),
    (2, "mark", "south", dict(unit="m1")),
    (2, "wide", "south", WIDE),
    (6, "total", "east", {}),
    (6, "mark", "south", dict(unit="m0")),
    (6, "mark", "east", dict(unit="m3")),
    (6, "wide", "north", WIDE),
    (6, "joint", None, dict(couple=["s2", "s3"])),
    (10, "total", "west", {}),
    (10, "mark", "west", dict(unit="m1")),
    (10, "wide", "west", dict(WIDE, unit="m2")),
    (10, "mark", "east", dict(unit="m0")),
)


def without_beside(facts, registry, held):
    started = time.perf_counter()
    result = (
        np.full(len(facts.obs_entity), NEVER, np.int64),
        np.where(np.asarray(facts.fact_code) == registry.code("beside"), ALWAYS, NEVER),
    )
    LOG.debug("without beside exit graph=%s seconds=%.4f", facts.name, time.perf_counter() - started)
    return result


@pytest.fixture(scope="module", autouse=True)
def registered():
    SORTS.register("pair", Sort(True, False, 2, 2, False))
    SEES.register("without_beside", without_beside)
    yield
    del SORTS.held["pair"], SEES.held["without_beside"]


@timed("tasks.stated")
def stated(absent):
    target = dict(total="total", mark="mark", wide="wide")
    references = dict(total=["whole"], mark=["unit"], wide=["unit", "link", "group", "route", "part"])
    about = dict(total=None, mark="unit", wide="link")
    kinds = {
        kind: dict(
            references=references[kind],
            columns=["ask"] if kind == "wide" else [],
            target=dict(kind="number", series=target[kind], about=about[kind]),
            weight=kind == "wide",
            absent=absent,
            event=None,
            matures=dict(reference="unit", column="mark") if kind == "mark" else None,
        )
        for kind in ("total", "mark", "wide")
    }
    kinds["joint"] = dict(
        references=["couple"], columns=[], target=None, weight=False, absent=absent, event=None, matures=None
    )
    return dict(
        version=VERSION,
        time=dict(model="clock", units=["hour"], ties="key"),
        observers=False,
        columns=dict(
            size=dict(type="number", encoder="value"),
            mark=dict(type="number", encoder="value"),
            load=dict(type="number", encoder="value"),
            weight=dict(type="number", encoder="logarithm"),
            ask=dict(type="number", encoder="value"),
        ),
        graph=dict(columns=[]),
        nodes=dict(maker=dict(columns=["size", "mark"], holds=None), site=dict(columns=["load"], holds=None)),
        sets={},
        edges=dict(
            serves=synthetic.edge("maker", "site", ["weight"]),
            beside=synthetic.edge("site", "site", [], directed=False),
        ),
        tasks=dict(
            key="ticket",
            grouping=None,
            signals=0,
            references=dict(
                whole=dict(sort="graph", kinds=[]),
                unit=dict(sort="node", kinds=["maker"]),
                link=dict(sort="edge", kinds=["maker"]),
                group=dict(sort="nodes", kinds=["site"]),
                route=dict(sort="path", kinds=["maker", "site"]),
                part=dict(sort="subgraph", kinds=["maker", "site"]),
                couple=dict(sort="pair", kinds=["site"]),
            ),
            kinds=kinds,
        ),
    )


@timed("tasks.graph")
def graph(place, name):
    values = []
    for number, key in enumerate(MAKERS):
        ends = [3, 6, 8 if key == "m2" else 9, None]
        rows = dict(
            key=[key] * 4,
            since=[0, 3, 6, 9],
            until=ends,
            size=[1.0 + number + place + 0.5 * at for at in range(4)],
            mark=[None] * 4,
        )
        if key == "m2":
            rows = {column: held[:3] for column, held in rows.items()}
        values.append(dict(form="state", parameters={}, rows=rows, kind="maker"))
    rows = dict(key=[], at=[], load=[])
    for number, key in enumerate(SITES):
        rows["key"] += [key] * 3
        rows["at"] += [0, 4, 8]
        rows["load"] += [2.0 + number + 0.25 * at + place for at in range(3)]
    values.append(dict(form="change", parameters=dict(keys=["key"]), rows=rows, kind="site"))
    edges = [
        dict(
            form="state",
            parameters={},
            rows=dict(
                source=[pair[0] for pair in SERVES],
                target=[pair[1] for pair in SERVES],
                present=[True] * len(SERVES),
                weight=[1.0 + at for at in range(len(SERVES))],
                since=[0] * len(SERVES),
                until=[8 if pair[0] == "m2" else None for pair in SERVES],
            ),
            table="serves",
            ends=("maker", "site"),
        ),
        dict(
            form="state",
            parameters={},
            rows=dict(
                source=[pair[0] for pair in BESIDE],
                target=[pair[1] for pair in BESIDE],
                present=[True] * len(BESIDE),
                since=[0] * len(BESIDE),
                until=[None] * len(BESIDE),
            ),
            table="beside",
            ends=("site", "site"),
        ),
    ]
    names = ("unit", "link", "group", "route", "part", "couple")
    tasks = dict(
        key=[100 * place + at for at in range(len(ASKED))],
        kind=[entry[1] for entry in ASKED],
        time=[entry[0] for entry in ASKED],
        split=[entry[2] for entry in ASKED],
        weight=[1.0 + at if entry[1] == "wide" else None for at, entry in enumerate(ASKED)],
        ask=[0.5 * at if entry[1] == "wide" else None for at, entry in enumerate(ASKED)],
        references={
            name: [entry[3][name] if name in entry[3] else None for entry in ASKED] for name in names
        },
        target=[
            None if entry[1] == "joint" else 0.5 * at + place * (1 + at % 3) + len(entry[1])
            for at, entry in enumerate(ASKED)
        ],
        known=[entry[0] + 3 if entry[1] == "mark" and entry[0] < 10 else None for entry in ASKED],
    )
    return dict(
        name=name,
        unit="hour",
        columns={},
        nodes=dict(maker=list(MAKERS), site=list(SITES)),
        values=values,
        edges=edges,
        tasks=tasks,
    )


@timed("tasks.case")
def case(absent):
    return dict(
        schema=stated(absent), graphs=[graph(place, name) for place, name in enumerate(("one", "two"))]
    )


@pytest.fixture(scope="module")
def bank(registered, tmp_path_factory):
    started = time.perf_counter()
    root = tmp_path_factory.mktemp("tasks")
    build(synthetic.stage(root, "tasks", case("keep")), root / "bank", 1, ("north", "south"))
    LOG.info("bank of tasks exit seconds=%.3f", time.perf_counter() - started)
    return root / "bank"


@timed("tasks.opened")
def opened(bank, **values):
    held = Deriver(bank, "cpu", RESIDENT)
    return held, configuration(held.registry, dict(view="snapshot"), values, 5, copy.deepcopy(SPLITS))


@timed("tasks.shown")
def shown(held, cfg, name, split):
    code = -1 if split is None else held.splits.index(split)
    for entry in held.viewed(cfg).graphs:
        if held.manifest["graphs"][entry["source"]]["name"] == name and entry["split"] == code:
            return Graph.load(held.view(entry["index"], cfg))
    raise KeyError("the bank shows no graph %s for the split %s" % (name, split))


@timed("tasks.serving")
def serving(held, view, step, relation):
    dense = view.dense()
    rows = (dense.fact_step == step) & dense.fact_present & (dense.fact_code == held.registry.code(relation))
    return sorted(
        (view.keys[source], view.keys[target])
        for source, target in zip(dense.fact_source[rows].tolist(), dense.fact_target[rows].tolist())
    )


@timed("tasks.observed")
def observed(held, view, step, key, column):
    dense = view.dense()
    rows = (dense.obs_step == step) & (dense.obs_entity == view.keys.index(key))
    values = np.asarray(view.obs_values)[dense.obs_ref[rows]]
    return values[:, held.registry.encoding.layout.index(column)].tolist()


def test_the_table_of_tasks_reads_back_from_the_bank_equal_to_what_was_stated(bank):
    held, _ = opened(bank)
    manifest, schema = held.manifest, held.registry.schema
    assert manifest["version"] == 8 and manifest["splits"] == ["east", "north", "south", "west"]
    assert manifest["trained"] == ["north", "south"]
    assert manifest["split_rows"] == dict(east=6, north=6, south=6, west=6)
    assert manifest["spec"]["extras"]["series"] == ["total", "mark", "wide"]
    written = support.written(bank.parent / "tasks.staging", schema)
    for entry in manifest["graphs"]:
        truth = Facts.load(held.truth(entry["index"]))
        read = truth.stated(held.registry, held.splits, manifest["observers"])
        wanted = written[truth.name]
        assert read["graph"] == wanted["graph"] and read["nodes"] == wanted["nodes"]
        for part in ("observations", "edges"):
            assert support.alike(read[part], wanted[part]), (truth.name, part)
        assert support.normal(read["tasks"], schema) == wanted["tasks"]
        assert read["targets"] == wanted["targets"]
        assert read["tasks"]["time"] == sorted(entry[0] for entry in ASKED)
        assert read["tasks"]["kind"].count("mark") == 6 and read["tasks"]["kind"].count("joint") == 1
        place = read["tasks"]["kind"].index("joint")
        assert read["tasks"]["split"][place] is None and read["tasks"]["weight"][place] is None
        assert read["tasks"]["references"]["couple"][place] == [("s2", None, None), ("s3", None, None)]
        assert read["tasks"]["references"]["unit"][place] is None
        wide = read["tasks"]["kind"].index("wide")
        found = {name: held[wide] for name, held in read["tasks"]["references"].items()}
        assert found["unit"] == [("m1", None, None)] and found["link"] == [LINK]
        assert found["group"] == [("s0", None, None), ("s1", None, None)] and found["route"] == ROUTE
        assert sorted(found["part"], key=repr) == sorted(
            [("m1", None, None), ("m1", "serves", "s1")], key=repr
        )
        assert found["whole"] is None and found["couple"] is None
        assert read["tasks"]["weight"][wide] == 4.0 and read["targets"]["known"].count(None) == 8
    rows = held.rows
    assert len(rows) == 24 and {row["kind"] for row in rows} == {"total", "mark", "wide"}
    assert {"weight" in row for row in rows if row["kind"] == "wide"} == {True}
    assert sorted(row["known"] for row in rows if "known" in row) == [5, 5, 5, 5, 9, 9, 9, 9]
    assert [entry["name"] for entry in held.split["series"]] == ["total", "mark", "wide"]
    for code, entry in enumerate(held.split["series"]):
        trained = [
            rows[place]["y"]
            for name in SPLITS["trained"]
            for place in held.split["indices"][name]
            if held.series[place] == code
        ]
        assert entry["mean"] == pytest.approx(np.mean(trained))
        assert entry["sd"] == pytest.approx(np.std(trained, ddof=1))


def test_a_view_holds_times_without_a_task_and_several_tasks_at_one_time(bank):
    held, cfg = opened(bank, **{"time/steps": "changes"})
    view = shown(held, cfg, "one", None)
    assert view.steps == len(STEPS) and view.task_step.tolist() == [1] * 4 + [4] * 3 + [7] * 3
    assert view.label_task.tolist() == [7, 8, 9]
    view = shown(held, cfg, "one", "south")
    assert view.task_step.tolist() == [1, 1, 1, 1, 4, 4, 4]
    assert sorted(set(range(len(STEPS))) - set(view.task_step.tolist())) == [0, 2, 3, 5, 6, 7]
    kinds = [held.registry.tasks[code] for code in view.task_kind.tolist()]
    assert kinds == ["total", "mark", "mark", "wide", "mark", "wide", "joint"]
    assert view.label_task.tolist() == [2, 3, 4]
    held, cfg = opened(bank)
    assert cfg["graph"]["time/steps"] == "tasks"
    view = shown(held, cfg, "two", "north")
    assert view.steps == 3 and view.task_step.tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert [held.rows[row]["ticket"] for row in view.label_row.tolist()] == [100, 101, 107]
    assert {held.rows[row]["graph"] for row in view.label_row.tolist()} == {"two"}


def test_kinds_state_one_and_five_references_of_every_sort(bank):
    held, cfg = opened(bank)
    schema = held.registry.schema
    assert [len(entry["references"]) for entry in schema.tasks["kinds"].values()] == [1, 1, 5, 1]
    sorts = {entry["sort"] for entry in schema.tasks["references"].values()}
    assert sorts == {"graph", "node", "edge", "nodes", "path", "subgraph", "pair"}
    assert set(SORTS.names()) >= sorts and SORTS.named("graph").columns() is None
    assert SORTS.named("path").columns() == dict(
        relation="STRING", to_kind="STRING", to_key="STRING", place="INT32"
    )
    with pytest.raises(ValueError, match="the sort of reference pair is registered twice"):
        SORTS.register("pair", Sort(True, False, 2, 2, False))
    with pytest.raises(KeyError, match="no sort of reference is registered under the name 'absent'"):
        Schema(
            dict(
                stated("keep"),
                tasks=dict(stated("keep")["tasks"], references=dict(whole=dict(sort="absent", kinds=[]))),
            )
        )
    truth = Facts.load(held.truth(0))
    names = list(schema.tasks["references"])
    wide = int(np.flatnonzero(truth.task_kind == held.registry.tasks.index("wide"))[0])
    members = np.flatnonzero(truth.member_task == wide)
    assert [names[code] for code in truth.member_name[members].tolist()] == [
        "unit",
        "link",
        "group",
        "group",
        "route",
        "route",
        "part",
        "part",
    ]
    assert truth.member_place[members].tolist() == [0, 0, 0, 1, 0, 1, 0, 1]
    assert (truth.member_code[members] >= 0).tolist() == [False, True, False, False, True, True, False, True]
    found = {name: 0 for name in names}
    for batch in batched(held, cfg):
        counted = {
            name: int((batch["member_name"] == code).sum()) + int((batch["whole_name"] == code).sum())
            for code, name in enumerate(names)
        }
        assert set(batch["whole_name"].tolist()) <= {names.index("whole")}
        assert counted["route"] == 2 * counted["link"] and 2 * counted["part"] == 3 * counted["link"]
        route = batch["member_name"] == names.index("route")
        rows = batch["member_row"][route]
        assert batch["member_place"][route].tolist() == [0, 1, 2, 3] * len(np.unique(rows))
        assert not batch["member_place"][~route].any()
        wide, pair = batch["row_kind"] == held.registry.tasks.index("wide"), batch["pair"]
        assert (pair[~wide] == -1).all() and (batch["member_name"][pair[wide]] == names.index("link")).all()
        assert (batch["partner"][pair[wide][:, 0]] == pair[wide][:, 1]).all()
        for name, count in counted.items():
            found[name] += count
    assert all(found[name] > 0 for name in names if name != "couple") and found["couple"] == 0


def test_a_sort_checks_the_members_a_task_names():
    none = np.full(2, -1)
    SORTS.named("node").check("the reference unit", [1, 1], np.arange(2), np.arange(2), none, none, none * 0)
    with pytest.raises(
        ValueError, match="unit holds exactly 1 member for every task that states it, and a task names 2"
    ):
        SORTS.named("node").check("the reference unit", [2], np.zeros(2), np.arange(2), none, none, none * 0)
    with pytest.raises(ValueError, match="group holds 1 or more members for every task that states it"):
        SORTS.named("nodes").check(
            "the reference group", [0], none[:0], none[:0], none[:0], none[:0], none[:0]
        )
    with pytest.raises(ValueError, match="couple holds exactly 2 members for every task that states it"):
        SORTS.named("pair").check(
            "the reference couple", [1], np.zeros(1), np.zeros(1), none[:1], none[:1], none[:1]
        )
    with pytest.raises(ValueError, match="the reference unit names nodes, and a task names an edge"):
        SORTS.named("node").check(
            "the reference unit", [1], np.zeros(1), np.zeros(1), np.ones(1), np.zeros(1), np.zeros(1)
        )
    with pytest.raises(ValueError, match="the reference link names edges, and a task names a node"):
        SORTS.named("edge").check(
            "the reference link", [1], np.zeros(1), np.zeros(1), none[:1], none[:1], np.zeros(1)
        )
    route = dict(
        task=np.zeros(2, np.int64), source=np.asarray([0, 1]), target=np.asarray([1, 2]), code=np.zeros(2)
    )
    SORTS.named("path").check("the reference route", [2], place=np.arange(2), **route)
    with pytest.raises(ValueError, match="route numbers the members of a task from zero in their order"):
        SORTS.named("path").check("the reference route", [2], place=np.ones(2, np.int64), **route)
    with pytest.raises(
        ValueError, match="route holds edges in order, each sharing a node with the one before"
    ):
        SORTS.named("path").check(
            "the reference route", [2], place=np.arange(2), **dict(route, source=np.asarray([0, 3]))
        )
    with pytest.raises(ValueError, match="holds members exactly when it states what they are"):
        Sort(True, False, 0, 0, False)
    with pytest.raises(ValueError, match="states the least and the most members it holds, found 2 and 1"):
        Sort(True, False, 2, 1, False)
    with pytest.raises(ValueError, match="the reference whole is of the sort graph and names kinds of node"):
        mapping = stated("keep")
        mapping["tasks"]["references"]["whole"]["kinds"] = ["maker"]
        Schema(mapping)


@timed("tasks.batched")
def batched(held, cfg):
    from tuning.batches.batching import pack_batch

    blocks = held.blocks(0, cfg["graph"]["time/history"], cfg)
    return [pack_batch(held, [block], cfg, "cpu") for block in blocks]


def test_a_task_reads_the_matured_target_of_an_earlier_one(bank):
    held, cfg = opened(bank)
    truth = Facts.load(held.truth(0))
    stated_marks = np.asarray(truth.obs_values)[:, held.registry.encoding.layout.index("mark")]
    assert np.isnan(stated_marks).all()
    rows = {row["ticket"]: row for row in held.rows if row["graph"] == "one"}
    view = shown(held, cfg, "one", None)
    assert view.steps == 3
    assert np.isnan(observed(held, view, 0, "m0", "mark")).all()
    assert observed(held, view, 1, "m0", "mark") == [pytest.approx(rows[1]["y"])]
    assert observed(held, view, 2, "m0", "mark") == [pytest.approx(rows[5]["y"])]
    assert observed(held, view, 1, "m1", "mark") == [pytest.approx(rows[2]["y"])]
    assert np.isnan(observed(held, view, 1, "m3", "mark")).all()
    assert np.isnan(observed(held, view, 2, "m3", "mark")).all()
    assert observed(held, shown(held, cfg, "one", "east"), 2, "m3", "mark") == [pytest.approx(rows[6]["y"])]
    held, cfg = opened(bank, **{"time/steps": "changes"})
    view = shown(held, cfg, "one", None)
    found = [observed(held, view, step, "m0", "mark")[0] for step in range(len(STEPS))]
    assert np.isnan(found[:4]).all() and STEPS[4] == 6
    assert found[4:6] == [pytest.approx(rows[1]["y"])] * 2
    assert found[6:] == [pytest.approx(rows[5]["y"])] * 2
    case_known = case("keep")
    case_known["graphs"][0]["tasks"]["known"][1] = 2
    with pytest.raises(ValueError, match="a target of graph one is known no later than the time of its task"):
        build(synthetic.stage(bank.parent, "soon", case_known), bank.parent / "soon", 1, ("north", "south"))


def test_the_statement_says_what_the_tasks_of_each_split_may_see(bank):
    held, cfg = opened(bank)
    views = {split: shown(held, cfg, "one", split) for split in ("north", "south", "east", None)}
    everything = sorted(SERVES[:2] + SERVES[3:] + SERVES[2:3])
    assert serving(held, views[None], 1, "serves") == everything
    assert serving(held, views["north"], 0, "serves") == everything
    assert serving(held, views["north"], 1, "serves") == sorted(set(SERVES) - {SERVES[0]})
    assert serving(held, views["north"], 1, "beside") == list(BESIDE)
    assert serving(held, views["south"], 1, "serves") == [("m0", "s0"), ("m0", "s1"), ("m1", "s1")]
    assert observed(held, views["south"], 1, "m3", "size") == [] and observed(
        held, views["north"], 1, "m3", "size"
    )
    assert serving(held, views["east"], 1, "beside") == [] and serving(
        held, views[None], 1, "beside"
    ) == list(BESIDE)
    assert SEES.names()[:2] == ("without_targets", "without_scored") and "without_beside" in SEES.names()
    with pytest.raises(
        ValueError, match="the rule of what a split may see without_beside is registered twice"
    ):
        SEES.register("without_beside", without_beside)
    splits = held.splits
    with pytest.raises(
        ValueError, match="the splits trained on name train, and the bank holds the splits east"
    ):
        Looking(splits, ["train"], ["east"], dict(train=[], east=[]))
    with pytest.raises(ValueError, match="the splits north are both trained on and scored on"):
        Looking(splits, ["north"], ["north"], dict(north=[]))
    with pytest.raises(
        ValueError, match="what a split may see is stated for every split trained on or scored on"
    ):
        Looking(splits, ["north"], ["east"], dict(north=[]))
    with pytest.raises(
        KeyError, match="no rule of what a split may see is registered under the name 'blind'"
    ):
        Looking(splits, ["north"], ["east"], dict(north=["blind"], east=[]))
    scoring = dict(SPLITS, sees=dict(SPLITS["sees"], east=["without_scored"]))
    other = configuration(held.registry, dict(view="snapshot"), {}, 5, scoring)
    with pytest.raises(ValueError, match="the rule is stated for a split scored on"):
        held.viewed(other)
    wrong = dict(trained=["north"], scored=["east"], sees=dict(north=[], east=[]))
    with pytest.raises(
        ValueError, match="the scaling of the targets of the bank was fitted on the splits north, south"
    ):
        held.blocks(0, 4, configuration(held.registry, dict(view="snapshot"), {}, 5, wrong))


def test_a_reference_to_a_node_not_observed_resolves_as_its_kind_states(bank):
    held, cfg = opened(bank)
    view = shown(held, cfg, "one", None)
    assert observed(held, view, 2, "m2", "size") == [] and observed(held, view, 1, "m2", "size") != []
    names = list(held.registry.schema.tasks["references"])
    late = np.flatnonzero((view.task_step == 2) & (view.task_kind == held.registry.tasks.index("wide")))
    units = np.flatnonzero(np.isin(view.member_task, late) & (view.member_name == names.index("unit")))
    assert [view.keys[entity] for entity in view.member_source[units].tolist()] == ["m2"]
    arrays = held.arrays(
        next(entry["index"] for entry in held.viewed(cfg).graphs if entry["split"] == -1), cfg
    )
    owner, entity = np.asarray(arrays["member_row"]), np.asarray(arrays["member_entity"])
    rows = np.flatnonzero(np.asarray(arrays["row_step"]) == 2)
    unit = np.isin(owner, rows) & (np.asarray(arrays["member_name"]) == names.index("unit"))
    wide = np.unique(owner[unit & (entity == view.keys.index("m2"))])
    assert len(wide) == 1
    members = np.flatnonzero(owner == wide[0])
    absent = np.asarray(arrays["member_observed"])[members] < 0
    assert absent.sum() == 1 and entity[members][absent].tolist() == [view.keys.index("m2")]
    assert all((batch["member_index"] >= 0).all() for batch in batched(held, cfg))
    with pytest.raises(
        ValueError,
        match="the reference unit of the task 11 of the kind wide of graph one names a node not observed "
        "at the time of the task, and its kind states that this stops the work",
    ):
        build(synthetic.stage(bank.parent, "stop", case("stop")), bank.parent / "stop", 1, ("north", "south"))
    assert not (bank.parent / "stop" / "manifest.json").exists()


def test_an_operation_states_that_it_may_remove_a_referenced_node_and_does(bank, monkeypatch):
    held, cfg = opened(bank)
    names = {parameter.name: parameter for parameter in support.parameters(held.registry, ())}
    assert names["nodes/referenced"].options == ("keep", "remove")
    assert names["nodes/referenced"].default == "keep"
    stages = {name: OPERATIONS.named(name) for name in support.SEVEN}
    assert [name for name, stage in stages.items() if transform.REMOVES in stage.changes] == ["graphspec.nodes"]
    dense = shown(held, cfg, "one", None).dense()
    sites = np.flatnonzero(dense.kinds == held.registry.kind("site"))
    found = {}
    for choice in ("keep", "remove"):
        steps = support.steps(held.registry, {"nodes/site/keep": False, "nodes/referenced": choice})[0]
        derived = segments.whole(dense, held.fitted(cfg), steps, transform.SEED, GraphLimits("cpu"), None)
        kept = derived.obs_entity[np.isin(derived.obs_entity, sites)]
        found[choice] = sorted({dense.keys[entity] for entity in kept.tolist()})
        assert not np.isin(derived.fact_source, sites).any() and not np.isin(derived.fact_target, sites).any()
        assert np.array_equal(derived.member_source, dense.member_source)
    assert found == dict(keep=["s0", "s1", "s2", "s3"], remove=[])
    monkeypatch.setattr(stages["graphspec.nodes"], "changes", ("observations", "facts"))
    with pytest.raises(
        ValueError,
        match="the operation graphspec.nodes removes the node of the reference "
        "(link|group|route|part|couple) of the task "
        r"\d+ of the kind (wide|joint) of graph one and does not state that it may remove a referenced node",
    ):
        segments.whole(dense, held.fitted(cfg), steps, transform.SEED, GraphLimits("cpu"), None)
    strict = support.registry("strict", ["maker"], [])
    assert strict.stops.all()
    names = {parameter.name: parameter for parameter in support.parameters(strict, ())}
    assert names["nodes/referenced"].options == ("keep",)


def test_a_search_offers_every_family_under_every_context_for_references_of_every_sort(bank):
    from tuning.families.model_spec import configured
    from tuning.searches.selector import check_bounds, graph_space, random_suggest
    from tuning.searches.space import Space

    held, _ = opened(bank)
    registry = held.registry
    assert views(registry) == VIEWS[:8]
    for family in VIEWS[:8]:
        found = {p.name: p for p in graph_space(registry, configured(registry, dict(view=family)), {})}
        assert (found["time/context"].low, found["time/context"].high) == (0, 3)
    drawn = [random_suggest(registry, seed, seed, {}, 600, support.template())[0] for seed in range(12)]
    assert len({config["model"]["view"] for config in drawn}) > 1
    space, rng = Space(registry, {}, 600, [], support.template()), np.random.default_rng(5)
    assert len({space.draw(rng, 0, {})["graph"]["time/context"] for _ in range(12)}) > 1
    check_bounds(registry, {"time/context": {"low": 1, "high": 3}}, support.template())


@pytest.fixture
def packed(bank, monkeypatch):
    return inline(bank, ("snapshot",), monkeypatch)


def test_a_bank_of_four_splits_with_targets_on_nodes_and_on_graphs_completes_one_fit(
    bank, packed, tmp_path, statement
):
    held = Deriver(bank, "cpu", RESIDENT)
    given = dict(statement, **copy.deepcopy(SPLITS))
    cfg = dict(
        configuration(
            held.registry,
            dict(view="snapshot"),
            {
                "time/history": 4,
                "time/steps": "changes",
                "nodes/site/filter": "degree",
                "nodes/site/min_degree": 2,
                "nodes/referenced": "remove",
            },
            5,
            copy.deepcopy(SPLITS),
        ),
        epochs=1,
    )
    blocks = held.blocks(0, 4, cfg)
    started = time.perf_counter()
    fit(bank, tmp_path, cfg, blocks, 16, started + statement["seconds"], state(given))
    LOG.info("fit of tasks exit seconds=%.3f", time.perf_counter() - started)
    saved = read_json(tmp_path / "result.json")
    assert saved["status"] == "complete" and 0 < saved["optimizer_updates"] <= 20
    assert saved["statement"] == given and saved["config"]["splits"] == SPLITS
    trained, scored = held.members(cfg, 0), held.members(cfg, 1)
    assert saved["train_unique_rows"] == len(trained) == 12 and saved["holdout_rows"] == len(scored) == 12
    assert {held.rows[row]["kind"] for row in trained} == {"total", "mark", "wide"}
    assert {held.rows[row]["kind"] for row in scored} == {"total", "mark", "wide"}
    splits = {name for name, rows in held.split["indices"].items() if set(rows) & set(trained.tolist())}
    assert splits == {"north", "south"} and np.isfinite(saved["raw_val_mse"])
    with pytest.raises(ValueError, match="the splits of the configuration are"):
        fit(
            bank,
            tmp_path,
            cfg,
            blocks,
            16,
            started + 30,
            state(dict(given, scored=["east"], sees=dict(north=[], south=[], east=[]))),
        )
