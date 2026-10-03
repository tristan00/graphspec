import copy
import logging
import time

import numpy as np
import pytest
import support
import torch

from conftest import inline, shared
from graphspec.facts.stored import Facts
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.infra.hashing import token
from graphspec.infra.timing import timed
from graphspec.schema.schema import VERSION, Schema
from graphspec.schema.targets import TARGETS, valued
from graphspec.sources import synthetic
from graphspec.views.graph import Graph
from tuning.bank.bank import Deriver
from tuning.bank.build import build
from tuning.batches.batching import batches, pack_batch
from tuning.families.model_spec import VIEWS, views
from tuning.families.models import build as network
from tuning.fits.losses import outcomes
from tuning.fits.training import fit
from tuning.statement import state

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("nodes")
USERS, ITEMS = ("u0", "u1", "u2", "u3"), ("i0", "i1", "i2", "i3", "i4")
TIMES, LINKED, CHANGED = (1, 2, 3, 4, 5, 6), (1, 2), ("g0", 4)
WITHHELD = dict(train=["without_targets"], validation=["without_targets"])
SEEING = dict(trained=["train"], scored=["validation"], sees=WITHHELD)
OPEN = dict(trained=["train"], scored=["validation"], sees=dict(train=[], validation=[]))
MASK = (1 << 62) - 1
KINDS = dict(
    rank=("ranking", "relevance", "items", {}),
    pick=("nodes", "chosen", "items", {}),
    next=("link", "bought", "who", dict(relation="buys")),
)
NEGATIVES = support.SEVEN + ("graphspec.negatives",)
DRAWN = {"negatives/of": "link:buys:who", "negatives/rule": "uniform", "negatives/count": 5}
NAMED = {token("item:" + key) & MASK: key for key in ITEMS}
OWNED = dict(u0=("i0", "i1", "i4"), u1=("i1", "i2", "i5"), u2=("i2", "i3", "i6"), u3=("i3", "i0", "i7"))
NEXT, BOUGHT = dict(u0="i2", u1="i3", u2="i0", u3="i1"), (1, 2, 3, 4)
SHIPPED = dict(uniform={}, degree={}, hops={"negatives/hops/within": 3})
OPENED, HELD, NETS = {}, {}, {}


@timed("nodes.aimed")
def aimed(kind, series, about, parameters):
    return dict(kind=kind, series=series, about=about, utility=synthetic.valued([series]), **parameters)


@timed("nodes.schema")
def schema(kinds):
    number = dict(type="number", encoder="value")
    asked = {
        name: dict(
            references=["who", "items"] if about == "items" else ["who"],
            columns=[],
            target=aimed(kind, series, about, parameters),
            weight=False,
            absent="stop",
            event=None,
            matures=None,
        )
        for name, (kind, series, about, parameters) in kinds.items()
    }
    asked["visit"] = dict(
        references=["who", "at"],
        columns=[],
        target=None,
        weight=False,
        absent="stop",
        event=["at", "who"],
        matures=None,
    )
    return dict(
        version=VERSION,
        time=dict(model="clock", units=["tick"], ties="key"),
        observers=False,
        columns=dict(load=number, price=number),
        graph=dict(columns=[]),
        nodes=dict(user=dict(columns=["load"], holds=None), item=dict(columns=["price"], holds=None)),
        sets={},
        edges=dict(
            buys=synthetic.edge("user", "item", []),
            near=synthetic.edge("item", "item", [], directed=False),
        ),
        tasks=dict(
            key="ticket",
            grouping=None,
            signals=1,
            references=dict(
                who=dict(sort="node", kinds=["user"]),
                at=dict(sort="node", kinds=["item"]),
                items=dict(sort="nodes", kinds=["item"]),
            ),
            kinds=asked,
        ),
    )


@timed("nodes.listed")
def listed(at, place):
    return [ITEMS[(at + place + step) % len(ITEMS)] for step in range(3)]


@timed("nodes.who")
def who(kind, at, place):
    return USERS[(at + place + (2 if kind == "next" else 0)) % len(USERS)]


@timed("nodes.stated")
def stated(kind, at, place, changed):
    items = listed(at, place)
    if kind == "rank":
        relevance = {item: float((at + position) % 3) for position, item in enumerate(items)}
        return dict(relevance, **({items[0]: 5.0} if changed else {}))
    if kind == "pick":
        chosen = {item: (at + position + place) % 2 == 0 for position, item in enumerate(items)}
        return dict(chosen, **({items[0]: not chosen[items[0]]} if changed else {}))
    return ITEMS[(2 * at + place + (3 if changed else 1)) % len(ITEMS)]


@timed("nodes.graph")
def graph(place, kinds, changed):
    names = list(kinds) + ["visit"]
    tasks = dict(key=[], kind=[], time=[], split=[], weight=[], target=[], known=[], signal=[], signal_observed=[])
    references = dict(who=[], at=[], items=[])
    for at, moment in enumerate(TIMES):
        for kind in names:
            tasks["key"].append(len(tasks["key"]))
            tasks["kind"].append(kind)
            tasks["time"].append(moment)
            tasks["split"].append(None if kind == "visit" else "train" if at < len(TIMES) // 2 else "validation")
            tasks["weight"].append(None)
            tasks["known"].append(None if kind == "visit" else moment + 1)
            tasks["signal"].append([0.1 * at + place])
            tasks["signal_observed"].append([True])
            touched = changed and ("g%d" % place, at) == CHANGED
            tasks["target"].append(None if kind == "visit" else stated(kind, at, place, touched))
            references["who"].append(who(kind, at, place))
            references["at"].append(ITEMS[at % len(ITEMS)] if kind == "visit" else None)
            references["items"].append(listed(at, place) if kind in ("rank", "pick") else None)
    users = dict(key=list(USERS), since=[0] * 4, until=[None] * 4, load=[1.0 + at + place for at in range(4)])
    items = dict(key=list(ITEMS), since=[0] * 5, until=[None] * 5, price=[2.0 * at - place for at in range(5)])
    buys = [(who("next", at, place), stated("next", at, place, False), TIMES[at]) for at in LINKED]
    return dict(
        name="g%d" % place,
        unit="tick",
        columns={},
        nodes=dict(user=list(USERS), item=list(ITEMS)),
        values=[
            dict(form="state", parameters={}, rows=users, kind="user"),
            dict(form="state", parameters={}, rows=items, kind="item"),
        ],
        edges=[
            dict(
                form="state",
                parameters={},
                table="buys",
                ends=("user", "item"),
                rows=dict(
                    source=[edge[0] for edge in buys],
                    target=[edge[1] for edge in buys],
                    present=[True] * len(buys),
                    since=[edge[2] for edge in buys],
                    until=[None] * len(buys),
                ),
            ),
            dict(
                form="state",
                parameters={},
                table="near",
                ends=("item", "item"),
                rows=dict(source=["i0", "i2"], target=["i1", "i3"], present=[True] * 2, since=[0, 0], until=[None] * 2),
            ),
        ],
        tasks=dict(tasks, references={name: values for name, values in references.items() if any(values)}),
    )


@timed("nodes.case")
def case(changed):
    return dict(schema=schema(KINDS), graphs=[graph(place, KINDS, changed) for place in range(2)])


@timed("nodes.bought")
def bought():
    items = sorted({item for held in OWNED.values() for item in held})
    tasks = dict(key=[], kind=[], time=[], split=[], weight=[], target=[], known=[], signal=[], signal_observed=[])
    for at, moment in enumerate(BOUGHT):
        for user in USERS:
            tasks["key"].append(len(tasks["key"]))
            tasks["kind"].append("next")
            tasks["time"].append(moment)
            tasks["split"].append("train" if at < 2 else "validation")
            tasks["weight"].append(None)
            tasks["known"].append(moment + 1)
            tasks["signal"].append([0.1 * at])
            tasks["signal_observed"].append([True])
            tasks["target"].append(NEXT[user])
    pairs = [(user, item) for user in USERS for item in OWNED[user]]
    rows = dict(
        source=[pair[0] for pair in pairs],
        target=[pair[1] for pair in pairs],
        present=[True] * len(pairs),
        since=[0] * len(pairs),
        until=[None] * len(pairs),
    )
    users = dict(key=list(USERS), since=[0] * 4, until=[None] * 4, load=[1.0 + at for at in range(4)])
    priced = dict(key=items, since=[0] * len(items), until=[None] * len(items), price=[1.0 * at for at in range(len(items))])
    graph = dict(
        name="g0",
        unit="tick",
        columns={},
        nodes=dict(user=list(USERS), item=items),
        values=[
            dict(form="state", parameters={}, rows=users, kind="user"),
            dict(form="state", parameters={}, rows=priced, kind="item"),
        ],
        edges=[
            dict(form="state", parameters={}, table="buys", ends=("user", "item"), rows=rows),
            dict(
                form="state",
                parameters={},
                table="near",
                ends=("item", "item"),
                rows=dict(source=["i0"], target=["i5"], present=[True], since=[0], until=[None]),
            ),
        ],
        tasks=dict(
            tasks,
            references=dict(
                who=[USERS[key % len(USERS)] for key in tasks["key"]],
                at=[None] * len(tasks["key"]),
                items=[None] * len(tasks["key"]),
            ),
        ),
    )
    return dict(schema=schema(dict(next=KINDS["next"])), graphs=[graph])


@timed("nodes.made")
def made(root):
    started = time.perf_counter()
    for name in ("stated", "changed"):
        build(synthetic.stage(root, name, case(name == "changed")), root / name, 1, ("train",))
    build(synthetic.stage(root, "bought", bought()), root / "bought", 1, ("train",))
    LOG.info("banks of nodes exit seconds=%.3f", time.perf_counter() - started)


@pytest.fixture(scope="module")
def banks():
    root = shared("nodes", made)
    yield {name: root / name for name in ("stated", "changed", "bought")}
    OPENED.clear()


@timed("nodes.opened")
def opened(banks, name):
    if name not in OPENED:
        OPENED[name] = Deriver(banks[name], "cpu", support.SEED, RESIDENT)
    return OPENED[name]


@timed("nodes.configured")
def configured(registry, family, splits, names=NEGATIVES, history=2):
    graph = dict(DRAWN, **{"time/history": history}) if names == NEGATIVES else {"time/history": history}
    return support.configuration(registry, dict(view=family), graph, 5, copy.deepcopy(splits), names)


@timed("nodes.read")
def read(held):
    found = {}
    kinds = held.registry.schema.tasks["kinds"]
    for row in held.rows:
        found.setdefault((row["graph"], row["kind"], row["ticket"]), []).append(row)
    return {key: TARGETS.named(kinds[key[1]]["target"]["kind"]).read(rows) for key, rows in found.items()}


@timed("nodes.built")
def built(root, name, given):
    return build(synthetic.stage(root, name, given), root / name, 1, ("train",))


def test_every_kind_naming_nodes_reads_back_as_stated_and_its_points_name_their_nodes(banks):
    held = opened(banks, "stated")
    schema, given = held.registry.schema, case(False)
    assert schema.target_columns() == dict(node="STRING") and Schema(schema.mapping()).mapping() == schema.mapping()
    assert schema.tasks["kinds"]["next"]["target"]["relation"] == "buys" and "relation buys" in schema.text(None)
    wanted = {
        (graph["name"], kind, key): target
        for graph in given["graphs"]
        for kind, key, target in zip(graph["tasks"]["kind"], graph["tasks"]["key"], graph["tasks"]["target"])
        if target is not None
    }
    assert read(held) == wanted
    keyed = np.asarray([token("item:" + row["node"]) & MASK for row in held.rows], np.int64)
    assert np.array_equal(held.nodes, keyed)
    for entry in held.manifest["graphs"]:
        truth = Facts.load(held.truth(entry["index"]))
        nodes = [truth.keys[node] for node in truth.target_node.tolist()]
        assert nodes == [held.rows[row]["node"] for row in truth.target_row.tolist()]


def test_a_ranking_states_a_relevance_for_every_item_its_task_lists():
    where, kind = "the target of the kind of task rank", TARGETS.named("ranking")
    for wrong in ({}, dict(i0=0.0), dict(i0=-1.0, i1=2.0), ["i0"]):
        with pytest.raises(ValueError, match="mapping from the key of every item its task lists to its relevance"):
            kind.staged(wrong, dict(series=("relevance",)), where)
    rows = dict(task=[0, 0, 1], graph=["g0"] * 3, y=[1.0, 0.0, 2.0], node=["i0", "i0", "i1"])
    with pytest.raises(ValueError, match="a task of graph g0 names the node i0 twice"):
        kind.points(rows, dict(series=("relevance",)), where)
    with pytest.raises(ValueError, match="one item at least of a relevance above 0"):
        kind.points(dict(rows, y=[0.0, 0.0, 2.0], node=["i0", "i1", "i1"]), dict(series=("relevance",)), where)
    assert kind.read([dict(node="i1", y=2.0), dict(node="i0", y=0.0)]) == dict(i1=2.0, i0=0.0)
    with pytest.raises(ValueError, match="is about the reference that lists the nodes its points name"):
        Schema(schema(dict(rank=("ranking", "relevance", None, {}))))
    with pytest.raises(ValueError, match="only a target of a kind that matures"):
        held = schema(KINDS)
        held["tasks"]["kinds"]["rank"]["matures"] = dict(reference="who", column="load")
        Schema(held)
    target = Schema(schema(KINDS)).tasks["kinds"]["rank"]["target"]
    tasks, values = valued(target, dict(task=[2, 2], series=["relevance"] * 2, y=[0.25, 0.75]))
    assert tasks.tolist() == [2] and values.tolist() == [1.0]


def test_a_ranking_names_every_listed_item_once_and_no_other_node(tmp_path):
    given = case(False)
    given["graphs"][0]["tasks"]["target"][0] = dict(i4=1.0, i1=0.0, i2=0.0)
    with pytest.raises(ValueError, match="names the node i4, which is not one member of the reference items"):
        built(tmp_path, "strange", given)
    given = case(False)
    given["graphs"][0]["tasks"]["target"][0] = {item: 1.0 for item in listed(0, 0)[:2]}
    with pytest.raises(ValueError, match="names 2 of the 3 members of the reference items of the task 0"):
        built(tmp_path, "short", given)
    given = case(False)
    given["graphs"][0]["tasks"]["target"][2] = "u1"
    with pytest.raises(ValueError, match="names the node u1 for the task 2; the graph holds 0 nodes of that key"):
        built(tmp_path, "unjoined", given)


def test_a_ranking_is_scored_listwise_and_measured_by_gain_and_reciprocal_rank():
    from tuning.fits.losses import LOSSES
    from tuning.measures.measures import measured, predicted, scored

    scores, row = torch.tensor([2.0, 0.0, 1.0, 1.0, 1.0, 3.0]), torch.tensor([0, 0, 0, 1, 1, 1])
    target = torch.tensor([0.0, 2.0, 1.0, 1.0, 0.0, 0.0])
    batch = dict(target=target, weight=torch.ones(6), point_row=row, tasks=2)
    share = torch.tensor([0.0, 2 / 3, 1 / 3, 1.0, 0.0, 0.0])
    logged = torch.cat([torch.log_softmax(scores[:3], 0), torch.log_softmax(scores[3:], 0)])
    assert float(LOSSES.named("listwise")(scores, batch, None)) == pytest.approx(float(-(share * logged).sum() / 2))
    probability = predicted(("ranking",), np.zeros(6, np.int64), row.numpy(), scores.numpy(), {})
    assert np.allclose(probability, logged.exp().numpy())
    held = scored(target.numpy(), probability, row.numpy(), np.zeros(6, np.int64), ["relevance"], np.ones(1), {})
    found = measured(["gain", "reciprocal", "hits1"], held, dict(relevance="ranking"))
    first = (0 + 1 / np.log2(3) + 2 / np.log2(4)) / (2 + 1 / np.log2(3))
    second = 1.0 * (1 / np.log2(3) + 1 / np.log2(4)) / 2
    assert found["gain"]["value"] == pytest.approx((first + second) / 2)
    assert found["reciprocal"]["value"] == pytest.approx((1 / 2 + 1 / 2.5) / 2)
    assert found["reciprocal"]["series"] == pytest.approx(dict(relevance=(1 / 2 + 1 / 2.5) / 2))
    assert found["hits1"]["value"] == 0.0


def test_a_set_of_nodes_is_one_decision_per_node_its_task_lists():
    from tuning.fits.losses import LOSSES
    from tuning.measures.measures import measured, predicted, scored

    where, kind = "the target of the kind of task pick", TARGETS.named("nodes")
    for wrong in ({}, dict(i0=1), dict(i0=True, i1="no"), ["i0"]):
        with pytest.raises(ValueError, match="mapping from the key of every node its task lists to whether it belongs"):
            kind.staged(wrong, dict(series=("chosen",)), where)
    rows = dict(task=[0, 0], graph=["g1"] * 2, y=[1.0, 0.5], node=["i0", "i1"])
    with pytest.raises(ValueError, match="state 1 for a node that belongs and 0 for one that does not"):
        kind.points(rows, dict(series=("chosen",)), where)
    assert kind.read([dict(node="i2", y=1.0), dict(node="i0", y=0.0)]) == dict(i2=True, i0=False)
    scores, row = torch.tensor([2.0, -1.0, 0.5, -3.0, -2.0]), torch.tensor([0, 0, 0, 1, 1])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    batch = dict(target=target, weight=torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0]), point_row=row, tasks=2)
    error = torch.nn.functional.binary_cross_entropy_with_logits(scores, target, reduction="none")
    expected = float((batch["weight"] * error).sum() / batch["weight"].sum())
    assert float(LOSSES.named("binary")(scores, batch, None)) == pytest.approx(expected)
    probability = predicted(("nodes",), np.zeros(5, np.int64), row.numpy(), scores.numpy(), {})
    assert np.allclose(probability, torch.sigmoid(scores).numpy())
    held = scored(target.numpy(), probability, row.numpy(), np.zeros(5, np.int64), ["chosen"], np.ones(1), {})
    found = measured(["f1", "jaccard", "binary"], held, dict(chosen="nodes"))
    assert found["f1"]["value"] == pytest.approx((2 * 1 / (2 + 2) + 1.0) / 2)
    assert found["jaccard"]["value"] == pytest.approx((1 / 3 + 1.0) / 2) == found["jaccard"]["series"]["chosen"]
    assert found["binary"]["value"] == pytest.approx(float(error.mean()))


def test_a_link_names_one_node_its_relation_joins_to_the_node_it_starts_from():
    from tuning.fits.losses import LOSSES
    from tuning.measures.measures import measured, predicted, scored

    where, kind = "the target of the kind of task next", TARGETS.named("link")
    for wrong in ("", ["i0"], None):
        with pytest.raises(ValueError, match="the key of the node its task links to"):
            kind.staged(wrong, dict(series=("bought",)), where)
    with pytest.raises(ValueError, match="state one node, the one its task links to, with y 1"):
        kind.points(dict(task=[0, 0], graph=["g0"] * 2, y=[1.0, 1.0], node=["i0", "i1"]), dict(series=("bought",)), where)
    assert kind.read([dict(node="i3", y=1.0)]) == "i3"
    refused = (
        (("link", "bought", "items", dict(relation="buys")), "the reference of the one node its link starts from"),
        (("link", "bought", "who", dict(relation="near")), "an edge table that joins the kinds of who to other nodes"),
        (("link", "bought", "who", {}), "lacks relation"),
    )
    for stated_kind, message in refused:
        with pytest.raises(ValueError, match=message):
            Schema(schema(dict(next=stated_kind)))
    scores, row = torch.tensor([0.5, 2.0, -1.0, 1.0, 1.0]), torch.tensor([0, 0, 0, 1, 1])
    target = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0])
    batch = dict(target=target, weight=torch.tensor([1.0, 1.0, 1.0, 3.0, 3.0]), point_row=row, tasks=2)
    logged = torch.cat([torch.log_softmax(scores[:3], 0), torch.log_softmax(scores[3:], 0)])
    expected = float(-(logged[0] * 1 + logged[4] * 3) / 4)
    assert float(LOSSES.named("entropy")(scores, batch, None)) == pytest.approx(expected)
    probability = predicted(("link",), np.zeros(5, np.int64), row.numpy(), scores.numpy(), {})
    held = scored(target.numpy(), probability, row.numpy(), np.zeros(5, np.int64), ["bought"], np.ones(1), {})
    found = measured(["reciprocal", "hits1", "hits10", "entropy"], held, dict(bought="link"))
    assert found["reciprocal"]["value"] == pytest.approx((1 / 2 + 1 / 1.5) / 2)
    assert found["hits1"]["value"] == 0.0 and found["hits10"]["value"] == 1.0
    assert found["entropy"]["value"] == pytest.approx(float(-(logged[0] + logged[4]) / 2))


def test_a_link_withholds_its_target_edge_from_its_split_and_is_scored_among_the_negatives_of_its_task(banks):
    held = opened(banks, "stated")
    shown = {}
    for name, splits in (("open", OPEN), ("seeing", SEEING)):
        cfg = configured(held.registry, "snapshot", splits)
        (entry,) = [
            entry
            for entry in held.viewed(cfg).graphs
            if held.manifest["graphs"][entry["source"]]["name"] == "g0" and entry["split"] in (-1, 0)
        ]
        view = Graph.load(held.view(entry["index"], cfg))
        dense, moment = view.dense(), TIMES[LINKED[0]]
        facts = zip(dense.fact_source, dense.fact_target, dense.fact_step, dense.fact_present)
        edges = {
            (view.keys[source], view.keys[target], int(np.asarray(view.step_time)[step]))
            for source, target, step, present in facts
            if present
        }
        shown[name] = (who("next", LINKED[0], 0), stated("next", LINKED[0], 0, False), moment) in edges
    assert shown == dict(open=True, seeing=False)
    cfg = configured(held.registry, "snapshot", SEEING)
    link, code = held.registry.schema.targets().index("link"), held.fitted(cfg).by_candidate["negatives"].code
    sizes = []
    for partition in (0, 1):
        for block in held.blocks(partition, 2, cfg):
            b = pack_batch(held, [block], cfg, "cpu")
            target = outcomes(held, b, "cpu", None)["target"]
            variant = held.arrays(block[0], cfg)
            tokens, rows, placed = variant["entity_token"], b["rows"], np.asarray(b["placed"])
            named = b["member_entity"][b["point_node"]].numpy()
            node = [NAMED[tokens[entity % len(tokens)]] for entity in named.tolist()]
            owner, drawn = np.r_[rows, b["drawn"]][placed], placed >= len(rows)
            for at in np.flatnonzero((b["aim"].numpy() == link) & ~drawn).tolist():
                row = held.rows[owner[at]]
                places = np.flatnonzero(drawn & (owner == owner[at]))
                negatives = [node[place] for place in places.tolist()]
                user = who("next", TIMES.index(row["time"]), int(row["graph"][1:]))
                anchor = np.flatnonzero(tokens == token("user:" + user) & MASK)
                chosen = (variant["candidate_code"] == code) & (variant["candidate_step"] == block[3])
                chosen &= np.isin(variant["candidate_source"], anchor)
                found = {NAMED[tokens[target]] for target in variant["candidate_target"][chosen].tolist()}
                assert node[at] == row["node"] and len(negatives) == len(set(negatives))
                assert set(negatives) == found - {node[at]}
                assert target[places].sum() == 0 and target[at] == 1 and "target" not in b
                keys = tokens[named[np.sort(np.r_[at, places])] % len(tokens)]
                assert np.all(keys[1:] > keys[:-1])
                sizes.append(len(negatives) + 1)
    assert len(sizes) == len(TIMES) * 2 and max(sizes) == len(ITEMS) > min(sizes)


def test_a_link_withholds_its_target_edge_only_until_the_target_is_known(banks):
    started, held = time.perf_counter(), opened(banks, "stated")
    shown = {}
    for name, splits in (("open", OPEN), ("seeing", SEEING)):
        cfg = configured(held.registry, "snapshot", splits)
        (entry,) = [
            entry
            for entry in held.viewed(cfg).graphs
            if held.manifest["graphs"][entry["source"]]["name"] == "g0" and entry["split"] in (-1, 0)
        ]
        view = Graph.load(held.view(entry["index"], cfg))
        dense, times = view.dense(), np.asarray(view.step_time).tolist()
        found = {}
        for source, target, step, present in zip(dense.fact_source, dense.fact_target, dense.fact_step, dense.fact_present):
            if present:
                found.setdefault((view.keys[source], view.keys[target]), []).append(times[step])
        shown[name] = found, times
    for at in LINKED:
        edge, moment = (who("next", at, 0), stated("next", at, 0, False)), TIMES[at]
        assert shown["open"][0][edge] == [time for time in shown["open"][1] if time >= moment]
        assert shown["seeing"][0].get(edge) == [time for time in shown["seeing"][1] if time >= moment + 1]
    LOG.info("links withheld until known exit seconds=%.3f", time.perf_counter() - started)


@timed("nodes.late")
def late(absent, kind, key):
    linking = dict(next=KINDS["next"])
    given = dict(schema=schema(linking), graphs=[graph(0, linking, False)])
    given["schema"]["tasks"]["kinds"]["next"]["absent"] = absent
    stated_graph = given["graphs"][0]
    stated_graph["tasks"]["references"]["items"] = [None] * len(stated_graph["tasks"]["key"])
    stated_graph["edges"][1]["rows"]["since"] = [2, 0]
    rows = stated_graph["values"][0 if kind == "user" else 1]["rows"]
    rows["since"][rows["key"].index(key)] = 2
    return given


def test_a_link_from_or_to_a_node_not_observed_at_its_task_is_refused_when_the_bank_is_built(tmp_path):
    started = time.perf_counter()
    refused = (
        ("item", "i1", "the target of the task 0 of the kind next of graph g0 names the node i1, which is not"),
        ("user", "u2", "the reference who of the task 0 of the kind next of graph g0 names the node u2, which is not"),
    )
    for absent in ("keep", "stop"):
        for kind, key, message in refused:
            with pytest.raises(ValueError, match=message + " observed at the time of the task; a link starts from"):
                built(tmp_path, "%s_%s" % (absent, key), late(absent, kind, key))
            assert not (tmp_path / ("%s_%s" % (absent, key)) / "manifest.json").exists()
    LOG.info("links refused exit seconds=%.3f", time.perf_counter() - started)


@timed("nodes.passed")
def passed(held, cfg):
    from graphspec.engine import segments
    from graphspec.operations.rows import select
    from graphspec.schema.references import MEMBERS

    steps, registry, result = held.stated(cfg)[1], held.fitted(cfg), {}
    for entry in held.viewed(cfg).graphs:
        dense = Graph.load(held.view(entry["index"], cfg)).dense()
        dense = select(dense, MEMBERS, np.asarray(dense.member_name) < len(registry.references))
        derived = segments.whole(dense, {}, registry, steps, held.seed, held.limits(), held.partitions(cfg))
        structural = ~np.asarray(derived.obs_anchor, bool)
        tokens = (np.asarray(derived.entity_token, np.uint64) & np.uint64(MASK)).astype(np.int64)
        result[entry["index"]] = set(
            zip(derived.obs_step[structural].tolist(), tokens[derived.obs_entity[structural]].tolist())
        )
    return result


def test_a_link_whose_nodes_the_pipeline_does_not_keep_is_left_out_and_its_kept_candidates_passed_alike(banks):
    from tuning.fits.training import shortened

    started, held = time.perf_counter(), opened(banks, "stated")
    link = held.registry.schema.targets().index("link")
    filters = (
        {"nodes/item/filter": "degree", "nodes/item/min_degree": 1},
        {"nodes/item/filter": "sample", "nodes/item/sample": 0.5},
    )
    for values in filters:
        graph_values = dict(DRAWN, **{"time/history": 2, "negatives/count": 1}, **values)
        cfg = support.configuration(
            held.registry, dict(view="snapshot"), graph_values, 5, copy.deepcopy(SEEING), NEGATIVES
        )
        kept, counts, every, held_out = passed(held, cfg), dict(kept=0, left=0), [], []
        for partition in (0, 1):
            for block in held.blocks(partition, 2, cfg):
                b = pack_batch(held, [block], cfg, "cpu")
                tokens, rows, placed = held.arrays(block[0], cfg)["entity_token"], b["rows"], np.asarray(b["placed"])
                owner, drawn = np.r_[rows, b["drawn"]][placed], placed >= len(rows)
                node, lefts = np.asarray(b["point_node"]), np.zeros(len(owner), bool)
                named = np.where(node >= 0, tokens[b["member_entity"].numpy()[np.maximum(node, 0)]], -1)
                every.append(owner)
                for at in np.flatnonzero((b["aim"].numpy() == link) & ~drawn).tolist():
                    row = held.rows[owner[at]]
                    step = int(b["row_steps"][b["point_row"][at]]) + block[1]
                    start = token("user:" + who("next", TIMES.index(row["time"]), int(row["graph"][1:]))) & MASK
                    negatives = named[drawn & (owner == owner[at])].tolist()
                    candidates = [(step, start), (step, int(held.nodes[owner[at]]))]
                    if named[at] < 0 and not negatives:
                        counts["left"] += 1
                        lefts |= owner == owner[at]
                        assert not all(candidate in kept[block[0]] for candidate in candidates)
                        continue
                    counts["kept"] += 1
                    assert named[at] == held.nodes[owner[at]]
                    assert all((step, node) in kept[block[0]] for node in negatives + [start])
                    assert candidates[1] in kept[block[0]]
                held_out.append((b, lefts, np.isin(rows, owner[lefts])))
        assert counts["kept"] > 0 and counts["left"] > 0, values
        for b, lefts, rows in held_out:
            assert np.array_equal(outcomes(held, b, "cpu", None)["counted"].numpy(), ~lefts)
            assert np.array_equal(np.asarray(b["left"]), rows)
        assert shortened(held, cfg["pipeline"], np.concatenate(every))[1]["next"] >= counts["left"]
    LOG.info("links left out exit seconds=%.3f", time.perf_counter() - started)


def test_the_draw_of_a_links_negatives_leaves_out_its_positive_under_every_shipped_rule(banks):
    started, held = time.perf_counter(), opened(banks, "bought")
    for rule, extra in SHIPPED.items():
        values = dict(DRAWN, **{"time/history": 2, "negatives/rule": rule, "negatives/count": 2}, **extra)
        cfg = support.configuration(held.registry, dict(view="snapshot"), values, 5, copy.deepcopy(OPEN), NEGATIVES)
        code, found = held.fitted(cfg).by_candidate["negatives"].code, {}
        for entry in held.viewed(cfg).graphs:
            variant = held.arrays(entry["index"], cfg)
            tokens, times = variant["entity_token"], np.asarray(variant["step_time"]).tolist()
            chosen = np.asarray(variant["candidate_code"]) == code
            columns = (np.asarray(variant["candidate_" + name])[chosen] for name in ("step", "source", "target"))
            for step, source, target in zip(*columns):
                found.setdefault((times[step], int(tokens[source])), []).append(int(tokens[target]))
        for index, row in enumerate(held.rows):
            user = USERS[row["ticket"] % len(USERS)]
            drawn = found[(row["time"], token("user:" + user) & MASK)]
            assert row["node"] == NEXT[user] and token("item:" + NEXT[user]) & MASK == held.nodes[index]
            assert int(held.nodes[index]) not in drawn and len(drawn) == len(set(drawn)) == 2, (rule, row)
    LOG.info("negatives without their positive exit seconds=%.3f", time.perf_counter() - started)


def test_a_pipeline_that_removes_a_kind_a_link_names_is_refused_when_stated_and_never_offered(banks):
    from graphspec.operations.pipeline import resolved
    from tuning.searches.selector import check_bounds, random_suggest

    started, registry = time.perf_counter(), opened(banks, "bought").registry
    contracted = {"contract/grouping": "item:price", "contract/aggregate": "mean", "contract/parallel": "keep"}
    removed = {"nodes/user/keep": True, "nodes/item/keep": False, "nodes/referenced": "keep", "nodes/user/filter": "none"}
    for operation, parameters in (("graphspec.contract", contracted), ("graphspec.nodes", removed)):
        with pytest.raises(
            ValueError,
            match="the entry %s of the pipeline removes the kind item, which the target of the kind of task next "
            "names" % operation,
        ):
            resolved(registry, [dict(operation=operation, parameters=parameters)])
    template = support.template(("graphspec.contract", "graphspec.nodes", "graphspec.negatives"))
    for seed in range(4):
        drawn = random_suggest(registry, seed, 5, {"pipeline/graphspec.contract": [True]}, 30, template)[0]
        stated = {entry["operation"]: entry["parameters"] for entry in drawn["pipeline"]}
        assert stated["graphspec.contract"]["contract/grouping"] == "user:load"
        assert "graphspec.nodes" not in stated or stated["graphspec.nodes"]["nodes/item/keep"]
    bounds = {"pipeline/graphspec.contract": [True], "pipeline/graphspec.contract/contract/grouping": ["item:price"]}
    with pytest.raises(
        ValueError,
        match="the bounds hold the entry graphspec.contract of the template present, and it removes the kind item, "
        "which the target of the kind of task next names",
    ):
        check_bounds(registry, bounds, template)
    LOG.info("kinds a link names kept exit seconds=%.3f", time.perf_counter() - started)


def test_a_fit_whose_pipeline_draws_no_negatives_for_a_link_stops_naming_its_kind(banks, tmp_path, statement):
    held = opened(banks, "stated")
    cfg = dict(configured(held.registry, "snapshot", SEEING, support.SEVEN, 4), epochs=1)
    losses = dict(ranking="listwise", nodes="binary", link="entropy")
    given = state(dict(statement, sees=WITHHELD, loss=losses, measures=["gain", "f1", "hits1"]))
    with pytest.raises(ValueError, match="the kind of task next states a link among sampled negatives, and the"):
        fit(banks["stated"], tmp_path, cfg, [], 16, time.perf_counter() + 30, given)


def test_a_search_holds_the_negatives_a_link_needs_and_a_template_without_them_stops_naming_its_kind(banks):
    from tuning.batches.inputs import drawing
    from tuning.searches.selector import check_bounds, pipeline_space, random_suggest

    registry, entry = opened(banks, "stated").registry, "pipeline/graphspec.negatives"
    template = support.template(NEGATIVES)
    found = {parameter.name: parameter.options for parameter in pipeline_space(registry, template, {})}
    assert found[entry] == (True,) and found[entry + "/negatives/of"] == ("link:buys:who",)
    for seed in range(3):
        config = random_suggest(registry, seed, 5, {}, 30, template)[0]
        assert list(drawing(registry.schema, config["pipeline"]).values()) == ["negatives"]
    needs = "the kind of task next states a target that needs an entry of graphspec.negatives with negatives/of "
    refused = (
        (support.template(), {}),
        (template, {entry: [False]}),
        (template, {entry + "/negatives/of": ["link:buys"]}),
    )
    for stated_template, bounds in refused:
        with pytest.raises(ValueError, match=needs + "link:buys:who, and the template holds none"):
            check_bounds(registry, bounds, stated_template)
    twice = template + [dict(operation="graphspec.negatives", label="more")]
    found = {parameter.name: parameter.options for parameter in pipeline_space(registry, twice, {})}
    assert found["pipeline/more"] == (False, True) and "link:buys:who" not in found["pipeline/more/negatives/of"]
    bounds = {entry + "/negatives/of": ["link:near"], "pipeline/more/negatives/of": ["link:buys:who"]}
    found = {parameter.name: parameter.options for parameter in pipeline_space(registry, twice, bounds)}
    assert found[entry] == (False, True) and found["pipeline/more"] == (True,)
    with pytest.raises(ValueError, match="the bounds hold the entry more of the template present, and every value"):
        check_bounds(registry, {"pipeline/more": [True], "pipeline/more/negatives/of": ["link:buys:who"]}, twice)


def test_a_bank_of_kinds_naming_nodes_completes_one_fit(banks, tmp_path, statement, monkeypatch):
    bank = banks["stated"]
    inline(bank, (), monkeypatch)
    held = opened(banks, "stated")
    filtered = {"nodes/item/filter": "degree", "nodes/item/min_degree": 1}
    graph_values = dict(DRAWN, **{"time/history": 4, "negatives/count": 1}, **filtered)
    cfg = dict(
        support.configuration(held.registry, dict(view="snapshot"), graph_values, 5, copy.deepcopy(SEEING), NEGATIVES),
        epochs=1,
    )
    losses = dict(ranking="listwise", nodes="binary", link="entropy")
    measures = ["gain", "reciprocal", "f1", "jaccard", "hits1", "entropy"]
    given = dict(statement, sees=WITHHELD, loss=losses, measures=measures)
    started = time.perf_counter()
    fit(bank, tmp_path, cfg, held.blocks(0, 4, cfg), 16, started + statement["seconds"], state(given))
    LOG.info("fit of kinds naming nodes exit seconds=%.3f", time.perf_counter() - started)
    saved = read_json(tmp_path / "result.json")
    assert saved["status"] == "complete" and 0 < saved["optimizer_updates"] <= 20
    found = saved["measures"]
    assert all(0 < found[name]["value"] <= 1 for name in ("gain", "reciprocal"))
    assert all(0 <= found[name]["value"] <= 1 for name in ("f1", "jaccard", "hits1"))
    assert list(found["reciprocal"]["series"]) == ["relevance", "bought"] and np.isfinite(found["entropy"]["value"])
    with np.load(tmp_path / "predictions.npz") as stored:
        linking = stored["aim"] == held.registry.schema.targets().index("link")
        rows, positive = stored["row_index"][linking], stored["prediction"][linking]
        alone = ~np.isin(rows, stored["drawn_row"])
        assert saved["unmeasured"] == dict(next=int(alone.sum())) and 0 < alone.sum() < len(rows)
        negative = np.asarray([stored["drawn_prediction"][stored["drawn_row"] == row][0] for row in rows[~alone]])
        ranks = np.where(positive[~alone] > negative, 1.0, np.where(positive[~alone] < negative, 2.0, 1.5))
        assert found["reciprocal"]["series"]["bought"] == pytest.approx(float(np.mean(1 / ranks)))
    with np.load(tmp_path / "predictions.npz") as stored:
        task, prediction, aim = held.task_of[stored["row_index"]], stored["prediction"], stored["aim"]
        drawn, owner = stored["drawn_prediction"], held.task_of[stored["drawn_row"]]
        targets = held.registry.schema.targets()
        for kind in ("ranking", "link"):
            chosen = aim == targets.index(kind)
            linking = kind == "link"
            total = np.bincount(
                np.r_[task[chosen], owner if linking else []].astype(np.int64),
                np.r_[prediction[chosen], drawn if linking else []],
            )
            assert np.allclose(total[np.unique(task[chosen])], 1.0)
        assert len(drawn) <= int((aim == targets.index("link")).sum()) == len(set(task[aim == targets.index("link")]))
        assert ((prediction[aim == targets.index("nodes")] > 0) & (prediction[aim == targets.index("nodes")] < 1)).all()


def test_the_batch_a_model_reads_holds_no_outcome_of_a_kind_naming_nodes(banks):
    started = time.perf_counter()
    held = opened(banks, "stated")
    assert held.registry.schema.targets() == ("ranking", "nodes", "link")
    assert support.outcomeless(held, configured(held.registry, "snapshot", SEEING)) == ([], True)
    LOG.info("outcomes of nodes exit seconds=%.3f", time.perf_counter() - started)


@timed("nodes.predicted")
def predicted(banks, family, name):
    if (family, name) not in HELD:
        held = opened(banks, name)
        cfg = configured(held.registry, family, SEEING)
        if family not in NETS:
            torch.manual_seed(5)
            NETS[family] = network(cfg, held.fitted(cfg)).eval()
        blocks = [block for partition in (0, 1) for block in held.blocks(partition, 2, cfg)]
        found = {}
        with torch.inference_mode():
            for graph in sorted({block[0] for block in blocks}):
                chosen = [block for block in blocks if block[0] == graph]
                tokens = held.arrays(graph, cfg)["entity_token"]
                for group in batches(chosen, len(chosen), held.sizes(cfg), None, frame_budget=1024):
                    b = pack_batch(held, group, cfg, "cpu")
                    owners = np.r_[b["rows"], b["drawn"]][b["placed"]]
                    named = b["member_entity"][b["point_node"]].tolist()
                    nodes = [NAMED[tokens[entity % len(tokens)]] for entity in named]
                    keys = [
                        "%s/%d/%s" % (held.rows[row]["graph"], held.rows[row]["ticket"], node)
                        for row, node in zip(owners.tolist(), nodes)
                    ]
                    found.update(zip(keys, NETS[family](b).tolist()))
        HELD[family, name] = found
    return HELD[family, name]


@pytest.mark.parametrize("family", VIEWS)
def test_changing_only_a_target_of_a_kind_naming_nodes_leaves_what_every_family_reads_unchanged(banks, family):
    assert family in views(opened(banks, "stated").registry)
    held, changed = predicted(banks, family, "stated"), predicted(banks, family, "changed")
    links = sum(row["kind"] == "next" for row in opened(banks, "stated").rows)
    assert len(held) > len(opened(banks, "stated").rows) + links
    assert sorted(held) == sorted(changed) and held == pytest.approx(changed, rel=1e-6, abs=1e-6)
    order = {}
    for name, found in (("stated", held), ("changed", changed)):
        for key in found:
            order.setdefault(name, {}).setdefault(key.rsplit("/", 1)[0], []).append(key.rsplit("/", 1)[1])
    assert order["stated"] == order["changed"]
    tasks = {}
    for key, value in held.items():
        tasks.setdefault(key.rsplit("/", 1)[0], set()).add(round(value, 6))
    assert any(len(values) > 1 for values in tasks.values())
    stated_targets, changed_targets = opened(banks, "stated").rows, opened(banks, "changed").rows
    differing = {row["kind"] for row, other in zip(stated_targets, changed_targets) if row != other}
    assert differing == {"rank", "pick", "next"}
