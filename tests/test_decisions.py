import copy
import logging
import re
import time

import numpy as np
import pyarrow as pa
import pytest
import support

from conftest import SPLITS, inline, shared
from graphspec import GENERATORS, RULES
from graphspec.build.decisions import Decided
from graphspec.facts.stored import Facts
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json, read_pack, write_json
from graphspec.infra.timing import timed
from graphspec.schema.decisions import RECORDED, drawn
from graphspec.schema.schema import Schema
from graphspec.schema.targets import TARGETS
from graphspec.sources import synthetic
from tuning.bank.bank import Deriver
from tuning.bank.build import build
from tuning.batches.batching import batches, pack_batch
from tuning.families.heads import NONE, REALISATION, REALISATIONS
from tuning.families.model_spec import VIEWS, views
from tuning.families.models import build as network
from tuning.families.periods import blocks as windows
from tuning.fits.check import check
from tuning.fits.losses import LOSSES, Summed, outcomes
from tuning.fits.training import fit
from tuning.statement import state

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("decisions")
GENERATOR = "nearby"
UNREALISED = {REALISATION: NONE}
DECIDING = ("turn", "show")
OPENED, READ = {}, []


class Nearby:
    version = 1


class Listed:
    version = 1

    @timed("decisions.listed.checked")
    def checked(self, stated, schema, where):
        if stated != dict(within=2):
            raise ValueError("%s reads within, found %r" % (where, stated))
        return stated


class Recorded:
    reads = ("decision",)

    @timed("decisions.recorded.call")
    def __call__(self, predictions, batch, bank):
        READ.append({key: value.numpy().copy() for key, value in batch.items() if key != "tasks"})
        return LOSSES.named("squared")(predictions, batch, bank)


@pytest.fixture(scope="module", autouse=True)
def registered():
    entries = ((GENERATORS, GENERATOR, Nearby()), (RULES, "listed", Listed()), (LOSSES, "recorded", Recorded()))
    for names, name, entry in entries:
        names.register(name, entry)
    yield
    for names, name, _ in entries:
        del names.held[name]
    OPENED.clear()


@timed("decisions.flipped")
def flipped(case):
    result = copy.deepcopy(case)
    for graph in result["graphs"]:
        for target in filter(None, graph["tasks"]["target"]):
            target["discount"] = 0.5 if target["discount"] is None else None
            for option in target["options"]:
                option["available"] = not option["available"]
                for name in ("shown", "logged"):
                    option[name] = 0.5 if option[name] is None else None
                if option["origin"] != RECORDED:
                    option["origin"] = dict(
                        option["origin"],
                        seed=option["origin"]["seed"] + 1,
                        drawn=0.5 if option["origin"]["drawn"] is None else None,
                    )
    return result


@timed("decisions.picked")
def picked(case):
    result = copy.deepcopy(case)
    kinds = result["schema"]["tasks"]["kinds"]
    for column in {name for kind in DECIDING for entry in kinds[kind]["target"]["actions"].values() for name in entry["parameters"]}:
        del result["schema"]["columns"][column]
    for kind in DECIDING:
        kinds[kind]["target"].update(actions=dict(go=dict(roles=dict(to=dict(sort="node", kinds=["spot"])), parameters=[])), nothing=None)
    result["graphs"] = result["graphs"][1:]
    for target in filter(None, result["graphs"][0]["tasks"]["target"]):
        target["options"] = [option for option in target["options"] if [action["kind"] for action in option["actions"]] == ["go"]]
        for option in target["options"]:
            option["actions"][0]["parameters"] = {}
        target["options"][0]["chosen"] |= not any(option["chosen"] for option in target["options"])
    return result


@timed("decisions.moved")
def moved(case):
    result = copy.deepcopy(case)
    for graph in result["graphs"]:
        for target in filter(None, graph["tasks"]["target"]):
            options = target["options"]
            chosen = [option for option in options if option["chosen"]]
            for option in chosen if len(chosen) > 1 else ():
                option["order"] = len(chosen) - 1 - option["order"]
            others = [
                option
                for option in options
                if not option["chosen"] and option["origin"] == RECORDED and (option["outcome"] is None) == (chosen[0]["outcome"] is None)
            ]
            if len(chosen) == 1 and others:
                chosen[0]["chosen"], others[-1]["chosen"] = False, True
            for option in options:
                option["origin"] = RECORDED
            unchosen = [option for option in options if not option["chosen"] and option["actions"]]
            unchosen[0]["origin"] = dict(generator=GENERATOR, seed=3, drawn=0.5)
            options.reverse()
    return result


@timed("decisions.made")
def made(root):
    started = time.perf_counter()
    case = synthetic.decisions(GENERATOR)
    for name, given in (
        ("stated", case),
        ("flipped", flipped(case)),
        ("picked", picked(case)),
        ("moved", moved(case)),
        ("moved_picked", moved(picked(case))),
    ):
        build(synthetic.stage(root, name, given), root / name, 1, ("train",))
    LOG.info("banks of decisions exit seconds=%.3f", time.perf_counter() - started)


@pytest.fixture(scope="module")
def banks(registered):
    root = shared("decisions", made)
    return {name: root / name for name in ("stated", "flipped", "picked", "moved", "moved_picked")}


@timed("decisions.opened")
def opened(banks, name):
    if name not in OPENED:
        OPENED[name] = Deriver(banks[name], "cpu", support.SEED, RESIDENT)
    return OPENED[name]


@timed("decisions.refused")
def refused(change, message, path):
    case = synthetic.decisions(GENERATOR)
    change(case)
    with pytest.raises(ValueError, match=message):
        build(synthetic.stage(path, "refused", case), path / "refused", 1, ("train",))


@timed("decisions.covered")
def covered(case):
    schema, found = Schema(case["schema"]), set()
    for graph in case["graphs"]:
        for kind, target, actors in zip(graph["tasks"]["kind"], graph["tasks"]["target"], graph["tasks"]["references"]["agents"]):
            if target is None:
                continue
            sorts ={name: {role: entry["sort"] for role, entry in held["roles"].items()} for name, held in schema.tasks["kinds"][kind]["target"]["actions"].items()}
            found.add(("actors", len(actors)))
            found.add(("concurrent", bool(target["concurrent"])))
            found.add(("outcomes", sum(option["outcome"] is not None for option in target["options"]) > 1))
            found.add(("chosen outcome", any(option["chosen"] and option["outcome"] for option in target["options"])))
            for option in target["options"]:
                found.add(("actions", min(len(option["actions"]), 2), option["together"]))
                found.add(("origin", option["origin"] == RECORDED))
                found.add(("outcome", type(option["outcome"]).__name__))
                found |= {("target", sorts[action["kind"]][role]) for action in option["actions"] for role in action["targets"]}
    return found


def test_a_decision_states_its_actors_kinds_of_action_and_the_option_that_does_nothing():
    stated = synthetic.decision_schema()
    schema = Schema(stated)
    assert schema.decided() == ("turn", "show") and schema.parameters() == ("speed", "mode", "lit")
    assert Schema(schema.mapping()).mapping() == schema.mapping() and schema.targets() == ("decision",)
    assert schema.target_columns() == dict(series="STRING", option="INT32", part="INT32", chosen="BOOL")
    assert schema.tasks["kinds"]["turn"]["target"]["nothing"] == "wait" and "actions" in schema.text(None)
    kind = TARGETS.named("decision")
    assert (kind.inputs, kind.outcomes, kind.scaled, kind.names) == ((), ("option", "part", "chosen"), True, None)
    wrong = (
        (dict(actors="nobody"), "names as its actors one of its references, of a sort that holds nodes only"),
        (dict(actions={}), "states its kinds of action as a mapping from name to roles and parameters, at least one"),
        (dict(nothing="a:b"), "the option that does nothing of the target of the kind of task turn needs a name"),
        (dict(rule=dict(name="unknown", parameters={})), "no rule of options is registered under the name 'unknown'"),
        (dict(rule=dict(name="listed", parameters=dict(within=3))), "reads within, found"),
    )
    for change, message in wrong:
        held = copy.deepcopy(stated)
        held["tasks"]["kinds"]["turn"]["target"].update(change)
        with pytest.raises((ValueError, KeyError), match=message):
            Schema(held)
    actions = (
        (("go", "roles", "to", "sort"), "wire", "is of one of the sorts node, edge, new, found 'wire'"),
        (("put", "roles", "made", "kinds"), ["mark", "spot"], "names the one kind of node of the element it creates"),
        (("go", "parameters"), ["weight"], "states its parameters as a list of distinct columns of the schema"),
        (("cut", "parameters"), ["mode", "mode"], "states its parameters as a list of distinct columns of the schema"),
    )
    for path, value, message in actions:
        held = copy.deepcopy(stated)
        entry = held["tasks"]["kinds"]["turn"]["target"]["actions"]
        for name in path[:-1]:
            entry = entry[name]
        entry[path[-1]] = value
        with pytest.raises(ValueError, match=message):
            Schema(held)
    first, aim = synthetic.decisions(GENERATOR)["graphs"][0]["tasks"]["target"][0], schema.tasks["kinds"]["turn"]["target"]
    for key, outcome in (("put", [None, None]), ("wait", [])):
        given = copy.deepcopy(first)
        (place,) = [at for at, option in enumerate(given["options"]) if option["key"] == key]
        given["options"][place]["outcome"] = outcome
        message = "the option %d of the graph g0, the task 10, of the key %s, states as its outcome per action %r, " % (place, key, outcome)
        with pytest.raises(ValueError, match="^" + re.escape(message + "the outcome of none of its actions; an option with no outcome states none") + "$"):
            kind.staged(given, aim, "the graph g0, the task 10")
    held = copy.deepcopy(stated)
    held["tasks"]["kinds"]["turn"]["target"]["rule"] = dict(name="listed", parameters=dict(within=2))
    ruled = Schema(held)
    assert ruled.tasks["kinds"]["turn"]["target"]["rule"] == dict(name="listed", parameters=dict(within=2))
    with pytest.raises(ValueError, match="the kind of task turn states its options by the rule listed; the options"):
        drawn(ruled)
    drawn(schema)


def test_the_synthetic_decisions_cover_every_case_of_the_check():
    found = covered(synthetic.decisions(GENERATOR))
    wanted = {
        ("actors", 1),
        ("actors", 2),
        ("concurrent", True),
        ("outcomes", True),
        ("outcomes", False),
        ("chosen outcome", True),
        ("chosen outcome", False),
        ("actions", 0, "set"),
        ("actions", 1, "set"),
        ("actions", 2, "set"),
        ("actions", 2, "sequence"),
        ("origin", True),
        ("origin", False),
        ("outcome", "dict"),
        ("outcome", "list"),
        ("outcome", "NoneType"),
        ("target", "node"),
        ("target", "edge"),
        ("target", "new"),
    }
    assert wanted <= found


def test_every_decision_reads_back_as_stated_with_its_fields_in_the_type_of_their_column(banks):
    held = opened(banks, "stated")
    schema, case = held.registry.schema, synthetic.decisions(GENERATOR)
    back, meant = support.read_back(held), support.meant(case)
    assert list(back) == list(meant) == ["g0", "g1"]
    for name, found in back.items():
        assert support.differing(found, meant[name], schema) == [], name
    assert meant["g0"]["nodes"]["mark"][0] == "m0" and "m0_3_a0" not in meant["g0"]["nodes"]["mark"]
    assert len(meant["g0"]["nodes"]["mark"]) > 2 and "m1_3_a0" in meant["g1"]["nodes"]["mark"]
    truth = Facts.load(held.truth(0))
    dtypes = dict(
        option_shown=np.float64,
        option_logged=np.float64,
        option_drawn=np.float64,
        option_seed=np.int64,
        option_order=np.int32,
        option_chosen=bool,
        option_available=bool,
        task_discount=np.float64,
        task_ended=np.int8,
        action_values=np.float32,
        target_y=np.float64,
    )
    assert {name: truth.arrays[name].dtype for name in dtypes} == {name: np.dtype(kind) for name, kind in dtypes.items()}
    assert truth.generators == [GENERATOR] and len(truth.options) == len(truth.option_task)
    rows = [row for row in held.rows if row["graph"] == "g0"]
    assert {type(row[name]) for row in rows for name in ("option", "part")} == {int}
    assert {type(row["chosen"]) for row in rows} == {bool}
    assert held.fields["chosen"].dtype == bool and held.fields["option"].dtype == np.int32
    assert np.array_equal(held.counted, held.fields["chosen"])
    stated = [option for target in filter(None, case["graphs"][0]["tasks"]["target"]) for option in target["options"]]
    outcomes = sum(
        len(option["outcome"]) if isinstance(option["outcome"], list) else option["outcome"] is not None
        for option in stated
    )
    assert len(rows) == 2 * outcomes and any(row["part"] == 1 for row in rows)
    first = case["graphs"][0]["tasks"]["target"][0]["options"]
    wanted = {
        (place, part): values
        for place, option in enumerate(first)
        for part, values in (
            enumerate(option["outcome"]) if isinstance(option["outcome"], list) else [(None, option["outcome"])]
        )
        if values is not None
    }
    assert TARGETS.named("decision").read([row for row in rows if row["ticket"] == 10]) == wanted
    chosen = opened(banks, "picked")
    back, meant = support.read_back(chosen), support.meant(picked(case))
    assert list(back) == list(meant) == ["g1"] and support.differing(back["g1"], meant["g1"], chosen.registry.schema) == []
    truth = Facts.load(chosen.truth(0))
    assert truth.arrays["action_values"].shape == (len(truth.arrays["action_option"]), 0) and len(truth.option_task) > 20


def test_a_target_or_a_generator_that_does_not_resolve_stops_the_build_naming_them(tmp_path):
    started = time.perf_counter()

    @timed("decisions.unresolved")
    def unresolved(case):
        option = case["graphs"][0]["tasks"]["target"][0]["options"][1]
        option["actions"][0]["targets"]["to"] = "s9"

    @timed("decisions.ungenerated")
    def ungenerated(case):
        option = case["graphs"][1]["tasks"]["target"][3]["options"][-1]
        option["origin"] = dict(option["origin"], generator="elsewhere")

    refused(
        unresolved,
        "the target to of the action 0, of the kind go, of the option go of the decision 10 of the kind turn of graph "
        "g0 names the node s9; the graph holds 0 nodes of that key of its kinds",
        tmp_path / "unresolved",
    )
    refused(
        ungenerated,
        "the option drawn of the decision 20 of the kind turn of graph g1 is regenerated by the generator elsewhere, "
        "which is not registered in graphspec.GENERATORS",
        tmp_path / "ungenerated",
    )
    LOG.info("unresolved decisions exit seconds=%.3f", time.perf_counter() - started)


def test_a_node_or_an_edge_an_action_targets_unobserved_at_the_step_of_its_decision_stops_the_build(tmp_path):
    started = time.perf_counter()

    @timed("decisions.later")
    def later(case):
        graph = case["graphs"][0]
        graph["nodes"]["spot"].append("s5")
        rows = graph["values"][1]["rows"]
        for name, value in (("key", "s5"), ("since", 4), ("until", None), ("load", 1.0)):
            rows[name].append(value)
        graph["tasks"]["target"][0]["options"][1]["actions"][0]["targets"]["to"] = "s5"

    @timed("decisions.unheld")
    def unheld(case):
        case["graphs"][0]["tasks"]["target"][0]["options"][2]["actions"][0]["targets"]["along"] = ["s0", "near", "s4"]

    @timed("decisions.joined")
    def joined(case):
        unheld(case)
        rows = case["graphs"][0]["edges"][0]["rows"]
        for name, value in (("source", "s0"), ("target", "s4"), ("present", True), ("since", 4), ("until", None)):
            rows[name].append(value)

    where = "the target %s of the action 0, of the kind %s, of the option %s of the decision 10 of the kind turn of graph g0 "
    edge = where % ("along", "cut", "cut") + "names the edge of near from s0 to s4, which is not observed at the step of its "
    for name, change, message in (
        ("later", later, where % ("to", "go", "go") + "names the node s5, which is not observed at the step of its "),
        ("unheld", unheld, edge),
        ("joined", joined, edge),
    ):
        seen = "never observed" if name == "unheld" else "first observed at the time 4"
        refused(change, message + "decision, at the time 1; it is %s$" % seen, tmp_path / name)
    LOG.info("unobserved targets exit seconds=%.3f", time.perf_counter() - started)


@timed("decisions.followed")
def followed(truth, registry, rows, changed):
    tasks = {name: np.asarray(truth.arrays[name]).copy() for name in ("task_key", "task_kind", "task_time", "task_ended")}
    keyed = {key: at for at, key in enumerate(tasks["task_key"].tolist())}
    for name, key, value in changed:
        tasks[name][keyed[key]] = value
    members = {name: truth.arrays[name] for name in ("member_task", "member_name", "member_source")}
    seen = dict(entity=truth.obs_entity, since=truth.obs_since)
    held = Decided(registry, tasks, members, truth.keys, truth.kinds, seen, truth.name)
    table = pa.table(
        dict(
            task=pa.array([keyed[row[0]] for row in rows], pa.int64()),
            other=pa.array([keyed[row[2]] for row in rows], pa.int64()),
            actor=pa.array([row[1] for row in rows], pa.string()),
        )
    )
    return held.followed(table, dict(option_task=truth.option_task), np.arange(len(keyed)))


def test_next_names_the_first_later_decision_of_each_actor_in_its_episode_and_any_other_stops_the_build(banks):
    started = time.perf_counter()
    held = opened(banks, "stated")
    truth = Facts.load(held.truth(0))
    keys, places = np.asarray(truth.task_key).tolist(), ("task", "actor", "other")
    stated = [(keys[first], truth.keys[actor], keys[other]) for first, actor, other in zip(*(truth.arrays["next_" + name].tolist() for name in places))]
    found = followed(truth, held.registry, stated, ())
    assert all(np.array_equal(found["next_" + name], truth.arrays["next_" + name]) for name in places)
    assert (10, "a0", 20) in stated and (20, "a0", 30) in stated and (51, "a1", 60) in stated
    skipped = [row for row in stated if row != (10, "a0", 20)]
    named = "the decision %d of the kind turn of graph g0 names %s as the next decision of the actor a0, which is %s"
    first = "the decision 20 of the kind turn, the first decision of that actor at a later time that states whether its episode ends"
    later = named % (10, "the decision 30 of the kind turn", first)
    for rows, changed, message in (
        (skipped + [(10, "a0", 30)], (), later),
        ([row for row in skipped if row[0] != 20] + [(10, "a0", 30)], (("task_ended", 20, 1),), later),
        (stated, (("task_ended", 10, -1),), named % (10, "the decision 20 of the kind turn", "none: it states no end of an episode, so no decision follows it")),
        (skipped, (), named % (10, "no decision", first)),
        (stated, (("task_ended", 20, 1),), named % (20, "the decision 30 of the kind turn", "none: its episode ends there")),
        (
            stated,
            (("task_time", 30, 2),),
            "the actor a0 takes the decision 20 of the kind turn and the decision 30 of the kind turn of graph g0 at the "
            "time 2, both stating whether their episode ends; an actor takes such decisions at different times, so that "
            "each has one next decision",
        ),
    ):
        with pytest.raises(ValueError, match="^" + re.escape(message) + "$"):
            followed(truth, held.registry, rows, changed)
    LOG.info("next decisions exit seconds=%.3f", time.perf_counter() - started)


@timed("decisions.equal")
def equal(left, right):
    return left.dtype == right.dtype and np.array_equal(left, right, equal_nan=left.dtype.kind == "f")


@timed("decisions.packed")
def packed(held, cfg):
    blocks = [block for partition in (0, 1) for block in held.blocks(partition, 2, cfg)]
    groups = [
        group
        for graph in sorted({block[0] for block in blocks})
        for group in batches(
            [block for block in blocks if block[0] == graph], len(blocks), held.sizes(cfg), None, frame_budget=1024
        )
    ]
    return [pack_batch(held, group, cfg, "cpu") for group in groups]


def test_no_option_without_an_outcome_and_no_option_not_chosen_reaches_a_loss_as_a_target(banks):
    held = opened(banks, "stated")
    cfg = support.configuration(held.registry, dict(view="snapshot", **UNREALISED), {"time/history": 2}, 5, SPLITS)
    loss, code, read = Summed(dict(decision="recorded"), held.registry.schema.targets(), None), 0, []
    READ.clear()
    for b in packed(held, cfg):
        assert "target" not in b and not any(key.startswith("field/") for key in b)
        known = outcomes(held, b, "cpu", None)
        rows = np.r_[b["rows"], b["drawn"]][b["placed"]]
        counted = ((b["aim"] == code) & known["counted"]).numpy()
        read += rows[counted].tolist()
        loss(b["aim"].double(), b, held)
    stated = [held.rows[row] for row in read]
    assert stated and all(row["chosen"] for row in stated)
    assert sorted(read) == sorted(row for row, found in enumerate(held.rows) if found["chosen"])
    assert all(found["field/chosen"].all() for found in READ) and sum(len(found["target"]) for found in READ) == len(read)
    case = synthetic.decisions(GENERATOR)
    places = {
        (graph["name"], key, place): option
        for graph in case["graphs"]
        for key, target in zip(graph["tasks"]["key"], graph["tasks"]["target"])
        if target is not None
        for place, option in enumerate(target["options"])
    }
    empty = {(graph, key, option["key"]) for (graph, key, _), option in places.items() if option["outcome"] is None}
    named = {(row["graph"], row["ticket"], row["option"]) for row in held.rows}
    assert empty and not {(graph, key, places[graph, key, place]["key"]) for graph, key, place in named} & empty


def test_availability_and_what_happened_never_reach_a_models_input(banks):
    started = time.perf_counter()
    stated, changed = opened(banks, "stated"), opened(banks, "flipped")
    first, second = (Facts.load(held.truth(0)) for held in (stated, changed))
    assert np.array_equal(first.option_available, ~second.option_available)
    assert read_json(banks["stated"] / "supervision.json") == read_json(banks["flipped"] / "supervision.json")
    cfg = support.configuration(stated.registry, dict(view="snapshot", **UNREALISED), {"time/history": 2}, 5, SPLITS)
    for index in range(len(stated.viewed(cfg).graphs)):
        views = [read_pack(held.view(index, cfg) / "arrays.pack", False) for held in (stated, changed)]
        assert sorted(views[0]) == sorted(views[1])
        assert all(equal(views[0][key], views[1][key]) for key in views[0])
    assert support.same(packed(stated, cfg), packed(changed, cfg))
    assert support.outcomeless(stated, cfg) == ([], True)
    LOG.info("availability unread exit seconds=%.3f", time.perf_counter() - started)


def test_a_bank_of_decisions_completes_one_fit_reading_the_task_and_the_outcome_of_the_chosen_option(
    banks, tmp_path, statement, monkeypatch
):
    bank = banks["stated"]
    inline(bank, (), monkeypatch)
    held = opened(banks, "stated")
    stated = dict(view="snapshot", **UNREALISED)
    cfg = support.configuration(held.registry, stated, {"time/history": 4, **support.several(held.registry)}, 5, SPLITS)
    cfg = dict(cfg, epochs=1)
    given = dict(statement, loss=dict(decision="recorded"), measures=["mse", "r2"])
    READ.clear()
    started = time.perf_counter()
    fit(bank, tmp_path, cfg, held.blocks(0, 4, cfg), 16, started + statement["seconds"], state(given))
    LOG.info("fit of decisions exit seconds=%.3f", time.perf_counter() - started)
    saved = read_json(tmp_path / "result.json")
    assert saved["status"] == "complete" and 0 < saved["optimizer_updates"] <= 20 and saved["unscored"] == 0
    trained = [held.rows[row] for row in held.split["indices"]["train"]]
    wanted = sorted(float(np.float32(row["y"])) for row in trained if row["chosen"])
    assert sorted(np.concatenate([found["target"] for found in READ]).tolist()) == wanted
    assert all(found["field/chosen"].all() for found in READ)
    with np.load(tmp_path / "predictions.npz") as found:
        chosen = found["field/chosen"]
        assert (~chosen).any() and set(found["series"].tolist()) == {0, 1}
        error = (found["prediction"] - found["target"])[chosen] ** 2
        assert saved["measures"]["mse"]["value"] == pytest.approx(float(error.mean()))
        assert list(saved["measures"]["mse"]["series"]) == ["gain", "cost"]


def test_the_check_skips_and_counts_a_window_whose_decisions_hold_no_chosen_outcome_as_training_does(
    banks, tmp_path, monkeypatch
):
    from test_realisations import REALISED

    started = time.perf_counter()
    held = opened(banks, "stated")
    REALISATIONS.register("whole", REALISED["whole"])
    try:
        models = [
            support.configuration(held.registry, dict(view="snapshot", **{REALISATION: name}), {"time/history": 2}, 5, SPLITS)
            for name in (NONE, "whole")
        ]
        cfg = models[0]
        monkeypatch.setattr("tuning.fits.check.samples", lambda registry: [model["model"] for model in models])
        for name in ("pipeline", "graph"):
            write_json(tmp_path / (name + ".json"), cfg[name])
        given = (tmp_path / "check.json", "cpu", 5, support.SEED, SPLITS, dict(decision="squared"))
        check(banks["stated"], *given, tmp_path / "pipeline.json", tmp_path / "graph.json", 1)
    finally:
        del REALISATIONS.held["whole"]
    found = read_json(tmp_path / "check.json")["results"]
    candidates = sorted(block for partition in (0, 1) for block in held.blocks(partition, 2, cfg) if block[0] == 1)
    last = pack_batch(held, windows(held, [candidates[-1]], cfg), cfg, "cpu")
    assert len(last["rows"]) and not held.counted[last["rows"]].any()
    assert [(result["skipped"], result["block"], result["ok"]) for result in found] == [
        (1, list(candidates[-2]), True),
        (0, list(candidates[-1]), True),
    ]
    assert [result["unused_parameters"] for result in found] == [[], []]
    LOG.info("check of decisions exit seconds=%.3f", time.perf_counter() - started)


@pytest.mark.parametrize("family", VIEWS)
def test_every_family_trains_on_the_outcome_of_the_chosen_option(banks, family):
    import torch

    started = time.perf_counter()
    held = opened(banks, "stated")
    assert family in views(held.registry)
    cfg = support.configuration(held.registry, dict(view=family, **UNREALISED), {"time/history": 2}, 5, SPLITS)
    torch.manual_seed(5)
    net = network(cfg, held.fitted(cfg))
    loss = Summed(dict(decision="squared"), held.registry.schema.targets(), None)
    blocks = held.blocks(0, 2, cfg)
    chosen = [block for block in blocks if block[0] == blocks[0][0]]
    b = pack_batch(held, batches(chosen, len(chosen), held.sizes(cfg), None, frame_budget=1024)[0], cfg, "cpu")
    value = loss(net(b), b, held)
    value.backward()
    assert torch.isfinite(value)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in net.parameters())
    LOG.info("family %s on decisions exit seconds=%.3f", family, time.perf_counter() - started)


def test_options_given_by_a_rule_stop_a_fit_until_a_generator_draws_them(tmp_path, statement):
    started = time.perf_counter()
    case = synthetic.decisions(GENERATOR)
    case["schema"]["tasks"]["kinds"]["turn"]["target"]["rule"] = dict(name="listed", parameters=dict(within=2))
    bank = tmp_path / "bank"
    build(synthetic.stage(tmp_path, "ruled", case), bank, 1, ("train",))
    config = dict(seed=statement["seed"], splits=SPLITS, pipeline=[])
    with pytest.raises(ValueError, match="the kind of task turn states its options by the rule listed; the options"):
        fit(bank, tmp_path / "fit", config, [], 8, time.perf_counter() + 30, state(dict(statement, loss=dict(decision="squared"))))
    assert not (tmp_path / "fit").exists()
    LOG.info("ruled decisions exit seconds=%.3f", time.perf_counter() - started)
