import copy
import gc
import logging
import re
import time
import weakref

import numpy as np
import pytest
import support
import torch
from test_decisions import banks, opened, registered
from test_realisations import PLAIN, SAMPLED, canonical, modelled, varied

from conftest import SPLITS, inline
from graphspec.engine.segments import Derivation
from graphspec.infra.files import read_json
from graphspec.infra.hashing import token
from graphspec.infra.timing import timed
from graphspec.schema.decisions import ROLES
from tuning import REALISATIONS, Edited
from tuning.bank import forks
from tuning.bank.options import READ as OFFERED
from tuning.bank.options import TOKEN
from tuning.batches.batching import pack_batch, pack_parts, parted, recipe_batch
from tuning.batches.tensors import keys, materialise
from tuning.families import model_spec
from tuning.families.heads import REALISATION, coded, realisation, refused
from tuning.families.model_spec import EDITS, VIEWS, realisable, space
from tuning.families.models import build as network
from tuning.families.periods import blocks as windows
from tuning.fits.guards import BACK, Forked, routed
from tuning.fits.losses import Summed
from tuning.fits.training import fit
from tuning.statement import state

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("decisions")
SHARED = (banks, registered)
RELATIONS = dict(
    went=dict(pairs=(("agent", "spot"),), directed=True, columns=("speed",)),
    cuts=dict(pairs=(("agent", "spot"),), directed=True, columns=("mode",)),
    made=dict(pairs=(("agent", "mark"),), directed=True, columns=("lit",)),
)
ACTED = dict(
    go=dict(to=dict(relation="went", actor="source")),
    cut=dict(along=dict(relation="cuts", actor="source")),
    put=dict(at=dict(relation="base", actor="source"), made=dict(relation="made", actor="source")),
)
WENT = dict(go=dict(to=dict(relation="went", actor="source")))
EDITED = dict(
    edited=Edited(dict(turn=ACTED, show=ACTED), RELATIONS, 1),
    moved=Edited(dict(turn=WENT, show=WENT), dict(went=dict(RELATIONS["went"], columns=())), 1),
)
STATED = {"snapshot/spatial": "relational", "memory/update": "mailbox_gru"}
READING = [family for family in VIEWS if model_spec.edits(dict(STATED, view=family))]
ALONE = 1e-5
JOINED = {"graphs/join": "single", "graphs/join_size": 2}
UNSEEN = ["parameter lit of put of show", "parameter mode of cut of show", "target along of cut of show"]
UNSEEN += ["target at of put of show", "target made of put of show"]


class Bare:
    version = 1
    reads = ("targets",)
    together = False
    back = True
    structure = False
    actor = True

    @timed("edited.Bare.refusal")
    def refusal(self, registry):
        return None


@pytest.fixture(scope="module", autouse=True)
def edits():
    for name, entry in EDITED.items():
        REALISATIONS.register(name, entry)
    yield
    for name in EDITED:
        del REALISATIONS.held[name]


@timed("edited.configured")
def configured(held, name, family, **stated):
    values = dict(modelled(family), **{REALISATION: name}, **stated)
    return support.configuration(held.registry, values, PLAIN, 5, SPLITS)


def last(held, cfg, count):
    started = time.perf_counter()
    blocks = sorted(held.blocks(1, 2, cfg))
    result = windows(held, [block for block in blocks if block[0] == blocks[-1][0]][-count:], cfg)
    LOG.debug("edited last exit blocks=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def test_an_edit_maps_every_kind_of_action_to_facts_and_is_refused_naming_what_it_leaves_unmapped(banks):
    started = time.perf_counter()
    stated, chosen = opened(banks, "stated"), opened(banks, "picked")
    registry = stated.registry
    assert refused("edited", registry) is None and refused("moved", chosen.registry) is None
    assert refused("moved", registry).endswith(
        "the parameter speed of the kind of action go of the kind of task turn becomes the value of no fact: no "
        "relation its roles become holds the column speed"
    )
    assert Edited(dict(turn=WENT, show=ACTED), RELATIONS, 1).refusal(registry) == (
        "the mapping of the edit states nothing for the kind of action cut of the kind of task turn, so an option "
        "holding it cannot be made an edit"
    )
    assert refused("edited", chosen.registry).endswith(
        "the edit adds the relation went between kinds of node or with columns the schema does not state"
    )
    assert EDITED["moved"].relations["went"]["columns"] == () and Edited(
        dict(turn=ACTED, show=WENT), EDITED["moved"].relations, 1
    ).refusal(chosen.registry) == (
        "the mapping of the edit states the kind of action cut of the kind of task turn, which it does not state"
    )
    went = dict(relation="went", actor="source")
    stay = varied(registry, "stay", dict(roles=dict(at=dict(sort="node", kinds=["spot"], count=[1, 1])), parameters=["speed"]), {})
    where = "the kind of action %s of the kind of task turn"
    for acted, relations, wrong, message in (
        (dict(ACTED, put=dict(at=went)), RELATIONS, registry, "the mapping of the edit states for %s the roles at, and it states the roles at, made" % (where % "put")),
        (dict(ACTED, go=dict(to=dict(went, relation="wings"))), RELATIONS, registry, "the role to of %s becomes wings, neither a relation the edit adds nor one the schema declares" % (where % "go")),
        (dict(ACTED, go=dict(to=dict(went, relation="made"))), RELATIONS, registry, "the role to of %s becomes facts of made from agent to spot, which made does not join" % (where % "go")),
        (dict(ACTED, go=dict(to=dict(went, actor="target"))), RELATIONS, registry, "the role to of %s becomes facts of went from spot to agent, which went does not join" % (where % "go")),
        (ACTED, dict(RELATIONS, went=dict(RELATIONS["went"], columns=())), registry, "the parameter speed of %s becomes the value of no fact: no relation its roles become holds the column speed" % (where % "go")),
        (ACTED, dict(RELATIONS, near=RELATIONS["went"]), registry, "the edit adds the relation near, a name without a colon the schema does not hold"),
        (dict(ACTED, stay=dict(at=went)), RELATIONS, stay, "the kinds of action go and stay of the kind of task turn become facts of the same relations, went, carrying the same parameters, so the field kind has no route"),
    ):
        assert Edited(dict(turn=acted, show=ACTED), relations, 1).refusal(wrong).startswith(message), message
    optional = varied(registry, "stay", dict(roles=dict(at=dict(sort="node", kinds=["spot"], count=[0, 1])), parameters=[]), {})
    assert Edited(dict(turn=dict(ACTED, stay=dict(at=went)), show=ACTED), RELATIONS, 1).refusal(optional) == (
        "the kind of action stay of the kind of task turn states no role holding a target in every action, and an edit "
        "makes the facts of an action from its targets, so an action holding none would make no fact"
    )
    assert Edited(dict(turn=ACTED, show=ACTED, visit=ACTED), RELATIONS, 1).refusal(registry).startswith(
        "the mapping of the edit states the kind of task visit, which decides nothing in the schema"
    )
    for mapping, version in ((dict(turn=dict(go=dict(to=dict(went, actor="side")))), 1), (dict(turn=ACTED), 0)):
        with pytest.raises(ValueError, match="an edit of the graph states its"):
            Edited(mapping, RELATIONS, version)
    REALISATIONS.register("bare", Bare())
    try:
        with pytest.raises(ValueError, match="the realisation of options bare states that messages pass from options back to the state, which they do through an edit"):
            realisation("bare")
    finally:
        del REALISATIONS.held["bare"]
    LOG.info("edit refusals exit seconds=%.3f", time.perf_counter() - started)


def test_each_family_states_what_of_an_edit_it_reads_and_a_field_it_cannot_read_has_no_route(banks):
    started = time.perf_counter()
    stated, chosen = opened(banks, "stated"), opened(banks, "picked")
    found = {family: model_spec.edits(dict(STATED, view=family)) for family in VIEWS}
    assert found["snapshot"] == EDITS and {family for family, held in found.items() if not held} == {"mixed", "affinity", "signal"}
    assert {family for family, held in found.items() if held == ("facts", "relations")} == set(READING) - {"snapshot"}
    assert model_spec.edits(dict(view="snapshot", **{"snapshot/spatial": "mean"})) == ("facts",)
    assert model_spec.edits(dict(STATED, view="memory", **{"memory/update": "mutual_rnn"})) == ()
    where = "the realisation of options %s puts options into the graph of the family of its model, and the family %s does not read them there: %s"
    for family, spatial, field, made in (
        ("event_graph", "relational", "parameter speed of go", "a value of a fact"),
        ("snapshot", "mean", "kind", "the relation of a fact"),
    ):
        spec = dict(view=family, **{"snapshot/spatial": spatial})
        reason = "an edit makes the field %s of the kind of task turn %s, and %s" % (field, made, model_spec.unedited(spec))
        with pytest.raises(ValueError, match="^" + re.escape(where % ("edited", family, reason)) + "$"):
            routed(stated, configured(stated, "edited", family, **({"snapshot/spatial": spatial} if family == "snapshot" else {})))
    mixed = "it reads no edge, so neither the facts an edit adds nor their relations nor their values"
    with pytest.raises(ValueError, match="^" + re.escape(where % ("edited", "mixed", mixed)) + "$"):
        routed(stated, configured(stated, "edited", "mixed"))
    with pytest.raises(ValueError, match="^" + re.escape(where % ("moved", "mixed", mixed)) + "$"):
        routed(chosen, configured(chosen, "moved", "mixed"))
    routed(stated, configured(stated, "edited", "snapshot"))
    for family in READING:
        routed(chosen, configured(chosen, "moved", family))
    for held, name, other, parts in ((stated, "edited", "moved", EDITS), (chosen, "moved", "edited", ("facts",))):
        parameters = realisable(held.registry, space(held.registry))
        (choosing,) = [parameter for parameter in parameters if parameter.name == REALISATION]
        assert name in choosing.options and other not in choosing.options
        for family in VIEWS:
            for spatial, update in (("relational", "mailbox_gru"), ("mean", "mutual_rnn"), ("none", "intensity")):
                spec = {"view": family, "snapshot/spatial": spatial, "memory/update": update}
                kept = model_spec.fitting(choosing, spec, held.registry.schema, parameters).options
                assert (name in kept) == (set(parts) <= set(model_spec.edits(spec))), (name, family, spatial, update)
    LOG.info("families reading edits exit seconds=%.3f", time.perf_counter() - started)


def test_an_edit_is_offered_only_under_settings_its_facts_reach_the_actor_and_a_statement_cutting_them_is_refused(banks):
    from tuning.searches.selector import check_bounds, graph_space, pipeline_space

    started = time.perf_counter()
    held = opened(banks, "stated")
    edit, vector = (configured(held, name, "snapshot")["model"] for name in ("edited", "vector"))
    directions = {name: [parameter for parameter in graph_space(held.registry, model, {}) if parameter.name == "relations/direction"][0].options for name, model in (("edited", edit), ("vector", vector))}
    assert directions == dict(edited=("inverse", "symmetric"), vector=("inverse", "symmetric", "directed"))
    where = "the field target to of go of the kind of task turn has no route to the prediction under the realisation of options edited: "
    directed = support.configuration(held.registry, dict(modelled("snapshot"), **{REALISATION: "edited"}), dict(PLAIN, **{"relations/direction": "directed"}), 5, SPLITS)
    with pytest.raises(ValueError, match="^" + re.escape(where + "an edit makes it facts of the relation went, which has a direction, from the actor, and the setting relations/direction directed reads a relation from its source to its target alone, so the actor reads none of them") + "$"):
        routed(held, directed)
    agentless = support.configuration(held.registry, dict(modelled("snapshot"), **{REALISATION: "edited"}), dict(PLAIN, **{"nodes/agent/keep": False}), 5, SPLITS)
    with pytest.raises(ValueError, match="^" + re.escape(where + "the pipeline removes the kind of node agent, which it may name, the entry graphspec.nodes removing it") + "$"):
        routed(held, agentless)
    unbased = support.configuration(held.registry, dict(modelled("snapshot"), **{REALISATION: "edited"}), dict(PLAIN, **{"relations/base/keep": False}), 5, SPLITS)
    with pytest.raises(ValueError, match="target at of put of the kind of task turn has no route .* the pipeline removes the relation base, which it may name, by relations/base/keep False of the entry graphspec.relations$"):
        routed(held, unbased)
    bounded = "the bounds leave relations/direction the values directed alone, and the family snapshot of a model the bounds leave the realisation of options edited reads the facts its edit adds under none of them"
    with pytest.raises(ValueError, match="^" + re.escape(bounded) + "$"):
        check_bounds(held.registry, {"relations/direction": ["directed"]}, support.template())
    check_bounds(held.registry, {"relations/direction": ["directed"], REALISATION: ["none", "vector"]}, support.template())
    found = {parameter.name: parameter for parameter in pipeline_space(held.registry, support.template(), {})}
    assert found["pipeline/graphspec.relations/relations/base/keep"].options == (True,)
    assert found["pipeline/graphspec.edges/edges/base/persistence"].high == 1
    chosen = opened(banks, "picked")
    for family, setting in (("event_graph", "events/kinds"), ("sequence", "neighbors/kinds"), ("walks", "neighbors/kinds")):
        model = configured(chosen, "moved", family)["model"]
        (kinds,) = [parameter.options for parameter in graph_space(chosen.registry, model, {}) if parameter.name == setting]
        assert kinds and all("relation" in option.split("+") for option in kinds) and "task" not in kinds, (family, kinds)
        hidden = support.configuration(chosen.registry, dict(modelled(family), **{REALISATION: "moved"}), dict(PLAIN, **{setting: "task"}), 5, SPLITS)
        shows = "an edit makes it facts of the relation went, and the setting %s task shows the family %s no relation" % (setting, family)
        with pytest.raises(ValueError, match="^" + re.escape("the field target to of go of the kind of task turn has no route to the prediction under the realisation of options moved: " + shows) + "$"):
            routed(chosen, hidden)
    bounded = "the bounds leave neighbors/kinds the values task alone, and the family sequence of a model the bounds leave the realisation of options moved reads the facts its edit adds under none of them"
    with pytest.raises(ValueError, match="^" + re.escape(bounded) + "$"):
        check_bounds(chosen.registry, {"neighbors/kinds": ["task"], "view": ["snapshot", "sequence", "projection"]}, support.template())
    check_bounds(chosen.registry, {"neighbors/kinds": ["task"], "view": ["snapshot", "projection"]}, support.template())
    LOG.info("edits offered where they reach exit seconds=%.3f", time.perf_counter() - started)


def test_the_options_of_a_decision_are_forks_of_one_history_each_adding_its_facts_at_its_step_alone(banks, monkeypatch):
    started = time.perf_counter()
    held = opened(banks, "stated")
    cfg = configured(held, "edited", "snapshot")
    blocks = sorted(block for block in held.blocks(1, 2, cfg) if block[0] == 0)[-2:]
    calls, continued = [], Derivation.continued

    @timed("edited.counted")
    def counted(derivation, high, edit):
        calls.append((derivation.time, high, edit is None))
        return continued(derivation, high, edit)

    monkeypatch.setattr(Derivation, "continued", counted)
    with forks.Forks(held):
        parts = [parted(held, [block], cfg) for block in blocks]
        for block in blocks:
            pack_batch(held, [block], cfg, "cpu")
        steps = [found[0][0].step for found in parts]
        options = [len(found) for found in parts]
        assert all(part[0].option >= 0 and part[0].move == () for found in parts for part in found) and options[1] > 2
        assert calls == [(0, steps[0], True)] + [(steps[0], steps[0] + 1, False)] * options[0] + [
            (steps[0], steps[0] + 1, True),
            (steps[0], steps[1], True),
        ] + [(steps[1], steps[1] + 1, False)] * options[1] + [(steps[1], steps[1] + 1, True)]
        monkeypatch.undo()
        codes = {held.fitted(cfg).code(name) for name in RELATIONS}
        whole = held.arrays(0, cfg)
        assert not np.isin(np.asarray(whole["spatial"])[:, 2], list(codes)).any()
        trunk = forks.trunk(held, cfg, 0, steps[1]).continued(steps[1] + 1, None).arrays()
        for part in parts[1]:
            fork = held.resident(part[0], cfg, "cpu")
            spatial, step = fork.numpy("spatial"), fork.numpy("spatial_step")
            entity, kept = fork.numpy("entity"), np.asarray(trunk["entity"])
            earlier = step < steps[1]
            assert np.array_equal(entity[spatial[earlier][:, :2]], kept[np.asarray(trunk["spatial"])[np.asarray(trunk["spatial_step"]) < steps[1]][:, :2]])
            assert not np.isin(spatial[earlier][:, 2], list(codes)).any()
            assert (step[np.isin(spatial[:, 2], list(codes))] == steps[1]).all()
        actions = forks.offered(held).actions[1][[part[0].option for part in parts[1]]]
        assert any(
            np.isin(held.resident(part[0], cfg, "cpu").numpy("spatial")[:, 2], list(codes)).any()
            for part, count in zip(parts[1], actions)
            if count
        )
        found = forks.offered(held)
        turned = copy.copy(found)
        turned.role = {name: values[np.lexsort((-np.arange(len(values)), found.role["code"], found.role["action"]))] for name, values in found.role.items()}
        edges, created, sizes, unmade = ROLES.index("edge"), ROLES.index("new"), set(), 0
        putting = coded(held.registry.schema)[0].index(("turn", "put"))
        for part in [part[0] for held_parts in parts for part in held_parts]:
            history, order = forks.trunk(held, cfg, part.graph, part.step), forks.ordered(held, part.option)
            edit = forks.made(held, cfg, history, part, order)
            low = int(found.roles[0][order[0]]) if order else 0
            high = int(found.roles[0][order[-1]] + found.roles[1][order[-1]]) if order else 0
            sorts = found.sorts[found.role["code"][low:high]]
            assert len(edit.added["facts"]["code"]) == len(sorts) + int((sorts == edges).sum()), part
            assert len(edit.added["observations"]["entity"]) == int((sorts == created).sum()), part
            sizes |= set(np.unique(np.stack((found.role["action"], found.role["code"]), 1)[low:high], axis=0, return_counts=True)[1].tolist())
            unmade += int(putting in found.action["kind"][order].tolist() and not (sorts == created).any())
            monkeypatch.setattr(forks, "offered", lambda bank: turned)
            again = forks.made(held, cfg, history, part, order)
            monkeypatch.setattr(forks, "offered", lambda bank: found)
            for side in ("observations", "facts"):
                for name, value in edit.added[side].items():
                    value = np.asarray(value)
                    assert np.array_equal(value, np.asarray(again.added[side][name]), equal_nan=value.dtype.kind == "f"), (side, name)
        assert 2 in sizes and unmade > 0
    LOG.info("forks of one history exit seconds=%.3f", time.perf_counter() - started)


def test_an_element_an_option_creates_reads_in_its_fork_as_no_entity_the_model_sees_elsewhere_chosen_or_not(banks):
    started = time.perf_counter()
    held = opened(banks, "stated")
    cfg = configured(held, "edited", "snapshot")
    mark, found = held.registry.kind("mark"), forks.offered(held)
    assert cfg["model"]["keys/embedding"] and held.shared_kinds()[mark]
    created = found.sorts[found.role["code"]] == ROLES.index("new")
    elements = dict(zip(found.action["option"][found.role["action"][created]].tolist(), found.role["source"][created].tolist()))
    chosen, seen, made = token("mark:m0") & TOKEN, set(), {}
    with forks.Forks(held):
        for index in held.graphs(cfg):
            whole = held.resident(index, cfg, "cpu")
            seen |= set(held.tokens(whole)[np.unique(whole.numpy("entity"))].tolist())
        blocks = sorted(block for block in held.blocks(0, 2, cfg) if block[0] == 0)
        parts = {part for part, *_ in parted(held, windows(held, blocks, cfg), cfg) if part.option in elements}
        for part in sorted(parts, key=lambda part: (part.step, part.option)):
            fork = held.resident(part, cfg, "cpu")
            entity, step = fork.numpy("entity"), fork.numpy("step")
            (element,) = np.unique(entity[(step == part.step) & (fork.numpy("entity_kind")[entity] == mark)]).tolist()
            made[elements[part.option]] = int(held.tokens(fork)[element])
    assert chosen in seen and chosen in made and len(made) > 2
    assert -1 not in made.values() and not set(made.values()) & seen and len(set(made.values())) == len(made)
    LOG.info("elements of forks exit forks=%d seconds=%.3f", len(made), time.perf_counter() - started)


def test_the_chains_and_forks_a_batch_reads_are_released_once_it_is_packed(banks, monkeypatch):
    started = time.perf_counter()
    held = opened(banks, "stated")
    cfg = configured(held, "edited", "snapshot")
    alive, made, derived = [], Derivation.__init__, forks.derived

    @timed("edited.tracked")
    def tracked(derivation, *arguments):
        made(derivation, *arguments)
        alive.append(("chain", weakref.ref(derivation)))

    @timed("edited.kept")
    def kept(*arguments):
        result = derived(*arguments)
        alive.append(("fork", weakref.ref(result)))
        return result

    monkeypatch.setattr(Derivation, "__init__", tracked)
    monkeypatch.setattr(forks, "derived", kept)
    b = pack_batch(held, last(held, cfg, 2), cfg, "cpu")
    gc.collect()
    counts = [sort for sort, _ in alive]
    assert len(b["options"]["row"]) > 2 and counts.count("fork") >= len(b["options"]["row"]) and "chain" in counts
    assert not [sort for sort, held in alive if held() is not None]
    LOG.info("forks released exit made=%d seconds=%.3f", len(counts), time.perf_counter() - started)


def test_a_batch_of_forks_carries_what_its_reader_needs_so_only_its_packer_derives_them(banks, monkeypatch):
    started = time.perf_counter()
    held = opened(banks, "stated")
    cfg = configured(held, "edited", "snapshot")
    chosen = last(held, cfg, 2)
    packed, tree = pack_batch(held, chosen, cfg, "cpu"), recipe_batch(held, chosen, cfg)
    calls, derived = [], forks.derived

    @timed("edited.read")
    def read(*arguments):
        calls.append(arguments[2])
        return derived(*arguments)

    monkeypatch.setattr(forks, "derived", read)
    assert len(packed["options"]["row"]) > 2 and keys(tree, set()) == set()
    assert support.same(materialise(held, cfg, tree, "cpu"), packed) and calls == []
    LOG.info("forks carried exit seconds=%.3f", time.perf_counter() - started)


@pytest.mark.parametrize("family", READING)
def test_each_option_scores_as_its_own_graph_encoded_alone_and_one_step_trains_every_family_that_reads_an_edit(banks, family):
    started = time.perf_counter()
    cases = [("picked", "moved", 1)] + ([("stated", "edited", 1)] if family == "snapshot" else [])
    largest = {}
    for bank, name, count in cases:
        held = opened(banks, bank)
        cfg = configured(held, name, family)
        torch.manual_seed(5)
        net = network(cfg, held.fitted(cfg))
        chosen = last(held, cfg, count)
        parts = parted(held, chosen, cfg)
        b = pack_batch(held, chosen, cfg, "cpu")
        values = net(b)
        with forks.Forks(held), torch.no_grad():
            alone = torch.cat([net(pack_parts(held, [part], cfg, "cpu")) for part in parts])
        assert len(parts) > 2 and len(b["options"]["row"]) == len(parts) and len(alone) == len(values) > 0
        largest[bank] = float((alone - values.detach()).abs().max())
        assert torch.allclose(alone, values.detach(), rtol=ALONE, atol=ALONE), (bank, family)
        loss = Summed(dict(decision="squared"), held.registry.schema.targets(), name)(values, b, held)
        loss.backward()
        torch.optim.SGD(net.parameters(), lr=0.01).step()
        assert torch.isfinite(loss) and net.head.options is None
        assert all(parameter.grad is not None for parameter in net.head.kinds.parameters())
    LOG.info("edits alone family=%s largest=%s exit seconds=%.3f", family, largest, time.perf_counter() - started)


def test_the_options_of_a_decision_in_a_joined_group_are_forks_of_its_history_each_scored_as_it_is_alone(banks):
    started = time.perf_counter()
    held = opened(banks, "stated")
    values = dict(modelled("snapshot"), **{REALISATION: "edited"})
    cfg = support.configuration(held.registry, values, dict(PLAIN, **JOINED), 5, SPLITS)
    chosen, members = last(held, cfg, 1), held.groups(cfg)
    first = held.viewed(cfg).graphs[members[chosen[0][0]][0]]["steps"]
    codes = {held.fitted(cfg).code(name) for name in RELATIONS}
    torch.manual_seed(5)
    net = network(cfg, held.fitted(cfg)).eval()
    with forks.Forks(held), torch.no_grad():
        parts = parted(held, chosen, cfg)
        values = net(pack_batch(held, chosen, cfg, "cpu"))
        alone = torch.cat([net(pack_parts(held, [part], cfg, "cpu")) for part in parts])
        made = [held.resident(part[0], cfg, "cpu") for part in parts]
        steps = [fork.numpy("spatial_step")[np.isin(fork.numpy("spatial")[:, 2], list(codes))] for fork in made]
    assert held.joining(cfg) and len(members[chosen[0][0]]) == 2 and len(parts) > 2
    assert all(part[0].step >= first for part in parts) and any(len(found) for found in steps)
    assert all((found == part[0].step).all() for found, part in zip(steps, parts))
    assert torch.allclose(alone, values, rtol=ALONE, atol=ALONE)
    LOG.info("joined forks exit options=%d seconds=%.3f", len(parts), time.perf_counter() - started)


def test_every_guard_holds_for_an_edit_and_one_refuses_a_field_whose_facts_the_family_does_not_read(banks):
    started = time.perf_counter()
    held = opened(banks, "stated")
    found, moved = {}, {}
    for spatial in ("relational", "mean", "relational_attention"):
        cfg = configured(held, "edited", "snapshot", **{"snapshot/spatial": spatial})
        torch.manual_seed(5)
        net = network(cfg, held.fitted(cfg)).eval()
        guard = Forked(held, cfg, net, 5, "cpu")
        with forks.Forks(held):
            guard.checked(pack_batch(held, last(held, cfg, 1), cfg, "cpu"))
        moved[spatial] = {name: value for (task, name), value in guard.moved.items() if task == "turn"}
        try:
            found[spatial] = guard.concluded(0.0)
        except ValueError as error:
            found[spatial] = str(error)
    offered, fields = forks.offered(held), {field["name"]: field for field in guard.fields["turn"]}
    created = offered.role["code"] == [code for code, entry in enumerate(guard.roles) if entry[:3] == ("turn", "put", "made")][0]
    putting = np.unique(offered.action["option"][offered.action["kind"] == guard.actions.index(("turn", "put"))])
    making = set(offered.action["option"][offered.role["action"][created]].tolist())
    held_made = {option: guard.holding(option, "turn", fields["target made of put"]) for option in putting.tolist()}
    assert set(held_made.values()) == {True, False} and {option for option, holds in held_made.items() if holds} == making
    assert all(guard.holding(option, "turn", fields["target at of put"]) for option in putting.tolist())
    assert found["relational"]["state"] == BACK and found["relational"]["unseen"] == UNSEEN
    assert found["relational"]["sets"] == 0 and found["relational"]["pairs"] > 0 and found["relational"]["share"] == 0.0
    stops = "with the weights drawn by the seed 5, the prediction does not move with the field %s of the kind of task turn for any option of the splits scored on, under the realisation of options edited"
    assert found["mean"] == stops % "kind" and found["relational_attention"] == stops % "parameter speed of go"
    assert (moved["mean"]["kind"], moved["mean"]["target to of go"]) == (False, True)
    assert (moved["relational_attention"]["kind"], moved["relational_attention"]["parameter speed of go"]) == (True, False)
    LOG.info("guards of an edit exit seconds=%.3f", time.perf_counter() - started)


def test_an_option_whose_edit_the_variant_shows_none_of_at_its_step_is_left_out_and_counted(banks):
    started = time.perf_counter()
    held = opened(banks, "stated")
    cfg = support.configuration(held.registry, dict(modelled("snapshot"), **{REALISATION: "edited"}), SAMPLED, 5, SPLITS)
    chosen = last(held, cfg, 2)
    b, found = pack_batch(held, chosen, cfg, "cpu"), forks.offered(held)
    with forks.Forks(held):
        parts = [part for part, *_ in parted(held, chosen, cfg) if part.option >= 0]
        empty = [part.option for part in parts if found.actions[1][part.option] and not forks.shown(held, cfg, part)]
    kept, rows = list(b["options"]["option"]), np.asarray(b["rows"])
    assert len(empty) > 0 and sorted(kept + empty) == sorted(part.option for part in parts) and not set(kept) & set(empty)
    assert list(b["options"]["left"]) == list(found.option["kind"][empty])
    left = np.asarray(b["left"], bool)
    assert left.any() and set(found.row_option[rows[left]].tolist()) <= set(empty)
    assert not set(found.row_option[rows[~left]].tolist()) & set(empty)
    LOG.info("edits shown none of exit left=%d seconds=%.3f", len(empty), time.perf_counter() - started)


def test_an_edit_completes_one_fit_whose_guards_record_that_messages_pass_back(banks, tmp_path, statement, monkeypatch):
    started = time.perf_counter()
    held = opened(banks, "picked")
    inline(banks["picked"], (), monkeypatch)
    cfg = dict(configured(held, "moved", "snapshot"), epochs=1)
    given = state(dict(statement, loss=dict(decision="squared"), measures=["mse"], alike=0.0))
    fit(banks["picked"], tmp_path, cfg, held.blocks(0, 2, cfg), 16, time.perf_counter() + statement["seconds"], given)
    saved = read_json(tmp_path / "result.json")
    found = saved["options"]
    assert saved["status"] == "complete" and saved["optimizer_updates"] > 0 and found["realisation"] == "moved"
    assert found["state"] == BACK and found["options"] > 0 and found["pairs"] > 0 and found["share"] == 0.0
    assert found["unseen"] == [] and saved["unmeasured"]["turn"] == 0
    LOG.info("fit of an edit exit seconds=%.3f", time.perf_counter() - started)


def test_the_facts_an_edit_adds_come_from_the_content_of_its_option_alone(banks):
    started = time.perf_counter()
    stated, flipped, moved = (opened(banks, name) for name in ("stated", "flipped", "moved"))
    assert not set(OFFERED) & {"option_" + name for name in ("available", "chosen", "order", "shown", "logged", "origin", "outcome")}
    made, packed, orders = [], [], []
    for held in (stated, flipped, moved):
        cfg = configured(held, "edited", "snapshot")
        made.append([])
        with forks.Forks(held):
            for part in parted(held, last(held, cfg, 1), cfg):
                history = forks.trunk(held, cfg, part[0].graph, part[0].step)
                order = forks.ordered(held, part[0].option)
                edit, turned = (forks.made(held, cfg, history, part[0], given) for given in (order, order[::-1]))
                made[-1].append([np.asarray(values) for side in edit.added.values() for values in side.values()])
                again = [np.asarray(values) for side in turned.added.values() for values in side.values()]
                orders.append((len(order), all(np.array_equal(left, right, equal_nan=left.dtype.kind == "f") for left, right in zip(made[-1][-1], again, strict=True))))
        packed.append(canonical(held, [pack_batch(held, last(held, cfg, 2), cfg, "cpu")]))
    assert any(count > 1 for count, _ in orders) and all(same for _, same in orders)
    assert all([len(edit) for edit in found] == [len(edit) for edit in made[0]] for found in made) and any(len(edit[-4]) for edit in made[0])
    assert all(
        np.array_equal(left, right, equal_nan=left.dtype.kind == "f")
        for other in made[1:]
        for one, two in zip(made[0], other, strict=True)
        for left, right in zip(one, two, strict=True)
    )
    assert support.same(packed[0], packed[1]) and support.same(packed[0], packed[2])
    LOG.info("edits from content alone exit seconds=%.3f", time.perf_counter() - started)
