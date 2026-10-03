import copy
import logging
import re
import time

import numpy as np
import pytest
import support
import torch
from test_decisions import GENERATOR, READ, banks, opened, registered
from torch import nn

from conftest import SPLITS, inline
from graphspec.infra.files import read_json
from graphspec.infra.hashing import token
from graphspec.infra.timing import timed
from graphspec.schema.schema import Schema
from graphspec.sources import synthetic
from tuning import REALISATIONS
from tuning.bank import options as offered
from tuning.bank.bank import Bank
from tuning.batches import inputs
from tuning.batches.batching import batches, pack_batch
from tuning.families import model_spec
from tuning.families import options as channel
from tuning.families.heads import FIELDS, NONE, REALISATION, coded, realisation, refused
from tuning.families.model_spec import VIEWS, realisable, space, views
from tuning.families.models import build as network
from tuning.families.options import Node
from tuning.families.periods import blocks as windows
from tuning.fits.guards import STATE, Guard, captured, routed
from tuning.fits.losses import Summed, counting, outcomes
from tuning.fits.training import fit
from tuning.statement import state

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("decisions")
KEYED = {"keys/embedding": True, "hidden": 8}
PLAIN = {"time/history": 2}
SAMPLED = dict(PLAIN, **{"nodes/spot/filter": "sample", "nodes/spot/sample": 0.5})
UNSEEN = ["parameter lit of put of show", "parameter mode of cut of show", "target along of cut of show"]
UNSEEN += ["target at of put of show", "target made of put of show"]
UNREAD = ("available", "chosen", "order", "shown", "logged", "generator", "seed", "drawn")
PACKED, BUILT, SHOWN = {}, {}, {}
SHARED = (banks, registered)
STRUCTURAL = ("action_nodes", "actor_edge", "set_edge")
READERS = ["snapshot", "mixed", "event_graph", "tree"]
SHIPPED = dict(node=("picked", []), vector=("stated", UNSEEN), **{name: ("stated", UNSEEN) for name in STRUCTURAL})
LAST = dict(node=1, vector=2, **{name: 2 for name in STRUCTURAL})
ALONE = 1e-5
REACHED = dict(
    node={"fields.role.weight", "fields.node.0.weight", "fields.node.0.bias"},
    vector={"fields.kind.weight", "fields.role.weight", "fields.relation.weight", "fields.node.0.weight"}
    | {"fields.node.0.bias", "realised.bound.0.weight", "realised.bound.0.bias", "realised.place.weight"}
    | {"realised.action.0.weight", "realised.action.0.bias", "realised.action.2.weight", "realised.action.2.bias"}
    | {"realised.empty"},
)


class Pooled(nn.Module):
    @timed("realisations.Pooled.__init__")
    def __init__(self, hidden, width, muted=False, placed=False):
        super().__init__()
        self.values, self.place = nn.Linear(width, hidden), nn.Linear(1, hidden, bias=False)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())
        self.muted, self.placed = muted, placed

    def acted(self, fields, held):
        targets = fields["targets"].new_zeros(fields["kind"].shape).index_add(0, held["role_action"], fields["targets"])
        values = self.values(fields["parameters"] * (0.0 if self.muted else 1.0))
        action = self.out(fields["kind"] + values + targets)
        sequence = held["sequence"][held["action_option"]] | self.placed
        return action + self.place(torch.log1p(held["action_place"].float())[:, None]) * sequence[:, None]

    def forward(self, fields, held):
        action, owner = self.acted(fields, held), held["action_option"]
        total = action.new_zeros((len(held["sequence"]), action.shape[1])).index_add(0, owner, action)
        count = action.new_zeros(len(held["sequence"])).index_add(0, owner, action.new_ones(len(owner)))
        return total / count.clamp_min(1)[:, None], action


class Mixing(Pooled):
    def forward(self, fields, held):
        options, actions = super().forward(fields, held)
        decision = held["decision"]
        total = options.new_zeros((int(decision.max()) + 1, options.shape[1])).index_add(0, decision, options)
        count = options.new_zeros(int(decision.max()) + 1).index_add(0, decision, options.new_ones(len(decision)))
        return options - (total / count[:, None])[decision], actions


class Flat(Pooled):
    def forward(self, fields, held):
        options, actions = super().forward(fields, held)
        return options - options.detach(), actions - actions.detach()


class Pooling(Pooled):
    def forward(self, fields, held):
        options, actions = super().forward(fields, held)
        targets = fields["targets"] - fields["targets"].mean(-1, keepdim=True)
        decision = held["decision"][held["action_option"][held["role_action"]]]
        pooled = targets.new_zeros((len(held["decision"]), targets.shape[1])).index_add(0, decision, targets)
        return options + pooled[held["decision"]], actions


class Realised:
    version = 1
    reads = FIELDS
    together = False
    back = False
    structure = False
    actor = False

    def __init__(self, built, **settings):
        self.built, self.settings = built, settings

    @timed("realisations.Realised.refusal")
    def refusal(self, schema):
        return None

    @timed("realisations.Realised.module")
    def module(self, hidden, width, spec, registry):
        return self.built(hidden, width, **self.settings)


class Deaf(Realised):
    version = 1
    reads = ("kind", "targets")


class Together(Realised):
    version = 1
    together = True


class Backward(Realised):
    version = 1
    back = True


REALISED = dict(
    whole=Realised(Pooled),
    deaf=Deaf(Pooled),
    muted=Realised(Pooled, muted=True),
    mixing=Realised(Mixing),
    together=Together(Mixing),
    ordered=Realised(Pooled, placed=True),
    flat=Realised(Flat),
    backward=Backward(Pooled),
    pooling=Realised(Pooling),
)


@pytest.fixture(scope="module", autouse=True)
def realisations():
    for name, entry in REALISED.items():
        REALISATIONS.register(name, entry)
    yield
    for name in REALISED:
        del REALISATIONS.held[name]
    PACKED.clear()
    BUILT.clear()
    SHOWN.clear()


@timed("realisations.configured")
def configured(held, realisation, values=KEYED, graph=SAMPLED):
    return support.configuration(held.registry, dict(values, **{REALISATION: realisation}), graph, 5, SPLITS)


@timed("realisations.grouped")
def grouped(held, cfg, partition):
    blocks = windows(held, held.blocks(partition, 2, cfg), cfg)
    return batches(blocks, len(blocks), held.sizes(cfg), None, frame_budget=1024)


@timed("realisations.modelled")
def modelled(family):
    return dict(KEYED if family in model_spec.EMBEDDING else {"hidden": 8}, view=family)


def packed(held, name, partition, family="snapshot", graph=SAMPLED):
    started = time.perf_counter()
    key = (id(held), name, partition, family, tuple(sorted(graph.items())))
    if key not in PACKED:
        cfg = configured(held, name, modelled(family), graph)
        PACKED[key] = [pack_batch(held, group, cfg, "cpu") for group in grouped(held, cfg, partition)]
    LOG.debug("realisations packed exit name=%s partition=%d seconds=%.4f", name, partition, time.perf_counter() - started)
    return PACKED[key]


@timed("realisations.built")
def built(held, name, family="snapshot"):
    if (name, family) not in BUILT:
        cfg = configured(held, name, modelled(family), PLAIN)
        torch.manual_seed(5)
        BUILT[name, family] = cfg, network(cfg, held.fitted(cfg)).eval()
    return BUILT[name, family]


@timed("realisations.guarded")
def guarded(held, name, alike=1.0):
    cfg, net = built(held, name)
    guard = Guard(held, cfg, net, 5)
    for b, plain in zip(packed(held, "whole", 1, graph=PLAIN), packed(held, NONE, 1, graph=PLAIN)):
        guard.checked(b, plain)
    return guard.concluded(alike)


@timed("realisations.decided")
def decided(b, held):
    rows = np.r_[np.asarray(b["rows"]), np.asarray(b["drawn"])][np.asarray(b["placed"])]
    found = {}
    for row, point in zip(rows.tolist(), b["point_row"].tolist()):
        found[point] = (held.rows[row]["graph"], held.rows[row]["ticket"])
    return found


def stated(registry, option):
    started = time.perf_counter()
    columns, dictionaries = registry.schema.parameters(), registry.encoding.dictionaries
    actions, values = [], []
    for action in option["actions"]:
        roles = {}
        for role, aim in sorted(action["targets"].items()):
            if isinstance(aim, list):
                roles[role] = (registry.code(aim[1]), frozenset(token("spot:" + key) & offered.TOKEN for key in aim[::2]))
            else:
                roles[role] = (token("spot:" + aim) & offered.TOKEN,) if role != "made" else ("new",)
        actions.append((action["kind"], roles))
        stated = action["parameters"]
        values.append(
            [
                np.nan
                if stated.get(name) is None
                else float(dictionaries[name].index(stated[name]) if name in dictionaries else stated[name])
                for name in columns
            ]
        )
    found = registry.encoding.vectors(columns, np.asarray(values, np.float32).reshape(-1, len(columns)))
    LOG.debug("realisations stated exit actions=%d seconds=%.4f", len(actions), time.perf_counter() - started)
    return option["together"] == "sequence", actions, found


def read(held, b, option):
    started = time.perf_counter()
    found, schema = b["options"], held.registry.schema
    acts, roles = coded(schema)
    chosen = np.flatnonzero(found["action_option"].numpy() == option)
    actions = []
    for action in chosen.tolist():
        held_roles = {}
        for role in np.flatnonzero(found["role_action"].numpy() == action).tolist():
            name = roles[int(found["role_code"][role])][2]
            ends = [int(b["member_token"][member]) for member in found["role_member"][role].tolist() if member >= 0]
            relation = int(found["role_relation"][role])
            held_roles[name] = (relation, frozenset(ends)) if relation >= 0 else tuple(ends) or ("new",)
        actions.append((acts[int(found["action_kind"][action])][1], dict(sorted(held_roles.items()))))
    result = bool(found["sequence"][option]), actions, found["action_values"][chosen].numpy()
    LOG.debug("realisations read exit actions=%d seconds=%.4f", len(actions), time.perf_counter() - started)
    return result


@timed("realisations.variant")
def variant(held, graph):
    if graph not in SHOWN:
        cfg = configured(held, "whole")
        index = [entry["index"] for entry in held.viewed(cfg).graphs if entry["name"] == graph][0]
        arrays = {name: np.asarray(value) for name, value in held.arrays(index, cfg).items()}
        tokens, entity, spatial = arrays["entity_token"], arrays["entity"], arrays["spatial"]
        SHOWN[graph] = (
            dict(zip(arrays["key"].tolist(), arrays["row_step"].tolist())),
            set(zip(arrays["step"].tolist(), tokens[entity].tolist())),
            set(zip(arrays["spatial_step"].tolist(), tokens[entity[spatial[:, 0]]].tolist(), tokens[entity[spatial[:, 1]]].tolist())),
        )
    return SHOWN[graph]


def unheld(held, graph, ticket, option):
    started = time.perf_counter()
    steps, shown, edges = variant(held, graph)
    step = steps[ticket]
    result = False
    for action in option["actions"]:
        for role, aim in action["targets"].items():
            if role == "made":
                continue
            ends = [aim[0], aim[2]] if isinstance(aim, list) else [aim]
            marks = [token("spot:" + key) & offered.TOKEN for key in ends]
            result |= not all((step, mark) in shown for mark in marks)
            result |= isinstance(aim, list) and (step, *marks) not in edges and (step, *marks[::-1]) not in edges
    LOG.debug("realisations unheld exit graph=%s seconds=%.4f", graph, time.perf_counter() - started)
    return result


def test_the_options_of_every_decision_reach_its_batch_beside_the_variant_and_read_back_as_stated(banks):
    started = time.perf_counter()
    held = opened(banks, "stated")
    case = {graph["name"]: graph for graph in synthetic.decisions(GENERATOR)["graphs"]}
    registry = held.fitted(configured(held, "whole"))
    seen, left, kept, counted = set(), 0, 0, 0
    for partition in (0, 1):
        for b in packed(held, "whole", partition):
            found, places = b["options"], decided(b, held)
            assert set(np.asarray(found["row"]).tolist()) <= set(places)
            for decision in sorted(set(found["decision"].tolist())):
                chosen = np.flatnonzero(found["decision"].numpy() == decision)
                graph, ticket = places[int(found["row"][chosen[0]])]
                tasks = case[graph]["tasks"]
                options = tasks["target"][tasks["key"].index(ticket)]["options"]
                kept_options = [option for option in options if not unheld(held, graph, ticket, option)]
                assert len(chosen) == len(kept_options) and (graph, ticket) not in seen
                seen.add((graph, ticket))
                left += len(options) - len(kept_options)
                for place, option in zip(chosen.tolist(), kept_options):
                    together, actions, values = stated(registry, option)
                    found_together, found_actions, found_values = read(held, b, place)
                    assert (found_together, found_actions) == (together, actions)
                    assert np.allclose(found_values, values)
                kept += len(chosen)
            counted += len(found["left"])
    wanted = {(row["graph"], row["ticket"]) for row in held.rows if row["kind"] in ("turn", "show")}
    assert seen == wanted and counted == left and kept > 50 and left > 5
    LOG.info("options read back exit kept=%d left=%d seconds=%.3f", kept, left, time.perf_counter() - started)


def test_nothing_a_realisation_must_not_read_reaches_its_batch(banks):
    started = time.perf_counter()
    stated, flipped = opened(banks, "stated"), opened(banks, "flipped")
    assert not set(offered.READ) & {"option_" + name for name in UNREAD}
    for partition in (0, 1):
        assert support.same(packed(stated, "whole", partition), packed(flipped, "whole", partition))
    cfg, targets, fields = configured(stated, "whole"), stated.targets, stated.fields
    stated.targets = targets[::-1] + 1
    stated.fields = dict(fields, chosen=~fields["chosen"])
    try:
        again = [pack_batch(stated, group, cfg, "cpu") for group in grouped(stated, cfg, 1)]
    finally:
        stated.targets, stated.fields = targets, fields
    assert support.same(packed(stated, "whole", 1), again)
    LOG.info("unread fields of options exit seconds=%.3f", time.perf_counter() - started)


def test_each_guard_refuses_its_broken_realisation_before_a_fit_naming_the_field_the_kind_of_task_and_it(banks):
    started = time.perf_counter()
    held = opened(banks, "stated")
    where = "the field %s of the kind of task turn has no route to the prediction under the realisation of options %s: "
    with pytest.raises(ValueError, match=where % ("parameter speed of go", "deaf") + "the realisation reads no parameters$"):
        routed(held, configured(held, "deaf"))
    removed = configured(held, "whole", graph={"time/history": 2, "nodes/spot/keep": False})
    removing = "the pipeline removes every kind of node it may name, spot, the entry graphspec.nodes removing spot$"
    with pytest.raises(ValueError, match=where % ("target to of go", "whole") + removing):
        routed(held, removed)
    with pytest.raises(ValueError, match="the realisation of options backward states that messages pass from options back"):
        routed(held, configured(held, "backward"))
    routed(held, configured(held, "whole"))
    found = guarded(held, "whole")
    assert found["sets"] > 0 and found["pairs"] > 0 and found["share"] == 0.0 and found["unseen"] == UNSEEN
    moved = "with the weights drawn by the seed 5, the prediction does not move with the field parameter speed of go"
    with pytest.raises(ValueError, match=moved + " of the kind of task turn for any option of the splits scored on, under the realisation of options muted$"):
        guarded(held, "muted")
    with pytest.raises(ValueError, match="option of the kind of task turn changes when the kind of another option of its decision change, and the realisation of options mixing does not state"):
        guarded(held, "mixing")
    assert guarded(held, "together")["pairs"] == found["pairs"]
    pooling = "changes when the members named by the targets of another option of its decision change, and the realisation of options pooling does not state"
    with pytest.raises(ValueError, match=pooling):
        guarded(held, "pooling")
    with pytest.raises(ValueError, match="whose actions, go, go, are a set changes with the order of its actions under the realisation of options ordered"):
        guarded(held, "ordered")
    assert guarded(held, "flat")["share"] == 1.0
    with pytest.raises(ValueError, match="have equal predictions under the realisation of options flat, a share of 1.0000 above the statement's alike 0.5$"):
        guarded(held, "flat", 0.5)
    LOG.info("guards of realisations exit seconds=%.3f", time.perf_counter() - started)


def test_a_search_offers_a_realisation_where_the_bank_decides_and_only_to_a_family_that_reads_options(banks, bank, monkeypatch):
    started = time.perf_counter()
    held = opened(banks, "stated")
    (offering,) = [parameter for parameter in space(held.registry) if parameter.name == REALISATION]
    assert offering.options == (NONE,) + REALISATIONS.names() and offering.default is None
    assert all(offering.active(dict(view=family)) for family in VIEWS) and model_spec.OPTIONED == VIEWS
    with pytest.raises(ValueError, match="states no options/realisation, which has no default"):
        support.configuration(held.registry, dict(view="snapshot"), {"time/history": 2}, 5, SPLITS)
    cfg = configured(held, "whole", graph=PLAIN)
    lacking = dict(cfg, model={name: value for name, value in cfg["model"].items() if name != REALISATION})
    stops = "the model of the family snapshot states no options/realisation, which a model of a family that reads options"
    with pytest.raises(ValueError, match=stops):
        routed(held, lacking)
    with pytest.raises(ValueError, match=stops):
        pack_batch(held, grouped(held, lacking, 1)[0], lacking, "cpu")
    plain = Bank(bank).registry
    assert REALISATION not in [parameter.name for parameter in space(plain)]
    assert model_spec.realised(lacking["model"], plain.schema) is None
    monkeypatch.setattr(model_spec, "OPTIONED", tuple(family for family in VIEWS if family != "signal"))
    (narrowed,) = [parameter for parameter in space(held.registry) if parameter.name == REALISATION]
    assert narrowed.active(dict(view="snapshot")) and not narrowed.active(dict(view="signal"))
    assert model_spec.realised(dict(view="signal", hidden=8), held.registry.schema) is None
    with pytest.raises(ValueError, match="spec sets inactive or unknown parameters: options/realisation"):
        configured(held, "whole", values={"view": "signal", "hidden": 8})
    LOG.info("realisation stated or absent exit seconds=%.3f", time.perf_counter() - started)


def test_the_loss_reads_the_points_of_the_realisation_its_model_states_whatever_the_batch_holds(banks):
    started = time.perf_counter()
    held = opened(banks, "stated")
    for name in (NONE, "whole"):
        for b in packed(held, name, 1, graph=PLAIN):
            placed = np.asarray(b["placed"])
            rows = np.r_[np.asarray(b["rows"]), np.asarray(b["drawn"])][placed]
            kept = ~np.r_[np.asarray(b["left"], bool), np.zeros(len(b["drawn"]), bool)][placed]
            assert np.array_equal(outcomes(held, b, "cpu", None)["counted"].numpy(), held.counted[rows] & kept)
            assert np.array_equal(outcomes(held, b, "cpu", "whole")["counted"].numpy(), kept)
            assert counting(held, b, "whole") == bool(kept.any())
    LOG.info("loss reads the stated realisation exit seconds=%.3f", time.perf_counter() - started)


@pytest.mark.parametrize("family", VIEWS)
def test_every_family_reads_the_options_through_a_realisation_and_its_state_reads_none_of_them(banks, family):
    started = time.perf_counter()
    held = opened(banks, "stated")
    assert family in views(held.registry) and family in model_spec.OPTIONED
    cfg, net = built(held, "whole", family)
    b = packed(held, "whole", 1, family, PLAIN)[0]
    guard = Guard(held, cfg, net, 5)
    guard.checked(b, packed(held, NONE, 1, family, PLAIN)[0])
    found = guard.concluded(1.0)
    assert found["options"] == len(b["options"]["sequence"]) > 0 and found["sets"] > 0 and found["pairs"] > 0
    loss = Summed(dict(decision="squared"), held.registry.schema.targets(), "whole")
    value = loss(net(b), b, held)
    value.backward()
    assert torch.isfinite(value) and all(
        parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in net.head.options.parameters()
    )
    LOG.info("family %s reads options exit seconds=%.3f", family, time.perf_counter() - started)


def test_a_realisation_registered_from_outside_completes_one_fit_whose_loss_reads_every_option_with_an_outcome(
    banks, tmp_path, statement, monkeypatch
):
    started = time.perf_counter()
    held = opened(banks, "stated")
    inline(banks["stated"], (), monkeypatch)
    cfg = dict(configured(held, "whole"), epochs=1)
    given = dict(statement, loss=dict(decision="recorded"), measures=["mse", "r2"], alike=0.25)
    READ.clear()
    fit(banks["stated"], tmp_path, cfg, held.blocks(0, 2, cfg), 16, time.perf_counter() + statement["seconds"], state(given))
    saved = read_json(tmp_path / "result.json")
    found = saved["options"]
    assert saved["status"] == "complete" and found["realisation"] == "whole" and found["unseen"] == UNSEEN
    assert found["pairs"] > 0 and found["share"] <= 0.25 and found["seconds"] > 0
    kinds = list(held.registry.schema.tasks["kinds"])
    left = [kinds[code] for b in packed(held, "whole", 1) for code in b["options"]["left"].tolist()]
    assert {kind: saved["unmeasured"][kind] for kind in ("turn", "show")} == {kind: left.count(kind) for kind in ("turn", "show")}
    assert saved["unmeasured"]["turn"] > 0
    rows = [row for b in packed(held, "whole", 0) for row, out in zip(b["rows"].tolist(), b["left"].tolist()) if not out]
    wanted = sorted(float(np.float32(held.rows[row]["y"])) for row in rows if held.rows[row]["kind"] in ("turn", "show"))
    read = np.concatenate([batch["target"] for batch in READ]).tolist()
    assert sorted(read) == wanted and not all(np.concatenate([batch["field/chosen"] for batch in READ]))
    LOG.info("fit of a realisation exit seconds=%.3f", time.perf_counter() - started)


def test_a_fit_stops_before_it_trains_where_a_guard_refuses_or_equal_pairs_pass_the_share_its_statement_states(
    banks, tmp_path, statement, monkeypatch
):
    started = time.perf_counter()
    held = opened(banks, "stated")
    inline(banks["stated"], (), monkeypatch)
    given = state(dict(statement, loss=dict(decision="squared"), measures=["mse"], alike=0.5))
    for name, message in (
        ("deaf", "the field parameter speed of go of the kind of task turn has no route to the prediction"),
        ("flat", "have equal predictions under the realisation of options flat, a share of 1.0000 above the statement's"),
    ):
        cfg = dict(configured(held, name, modelled("snapshot"), PLAIN), epochs=1)
        (tmp_path / name).mkdir()
        with pytest.raises(ValueError, match=message):
            fit(banks["stated"], tmp_path / name, cfg, held.blocks(0, 2, cfg), 16, time.perf_counter() + 30, given)
        assert not (tmp_path / name / "result.json").exists() and not (tmp_path / name / "checkpoint.pt").exists()
    LOG.info("fits refused by guards exit seconds=%.3f", time.perf_counter() - started)


class Turns:
    @timed("realisations.Turns.__init__")
    def __init__(self, views):
        self.views = iter(views)

    @timed("realisations.Turns.__call__")
    def __call__(self, bank):
        return next(self.views)


def single(found, slot):
    started = time.perf_counter()
    view = copy.copy(found)
    held = found.count > slot
    view.first, view.count = np.where(held, found.first + slot, -1), held.astype(np.int64)
    decided = (found.decision >= 0) & held[np.maximum(found.decision, 0)]
    view.decision = np.where(decided, found.decision, -1)
    own = np.where(decided, view.first[np.maximum(found.decision, 0)], -1)
    view.row_option = np.where(decided & (found.row_option == own), found.row_option, -1)
    view.row_action = np.where(view.row_option >= 0, found.row_action, -1)
    LOG.debug("realisations single exit slot=%d decisions=%d seconds=%.4f", slot, int(held.sum()), time.perf_counter() - started)
    return view


def alone(held, cfg, chosen, monkeypatch):
    started = time.perf_counter()
    found = offered.offered(held)
    views = [single(found, slot) for slot in range(int(found.count.max()))]
    with monkeypatch.context() as patched:
        patched.setattr(inputs, "offered", Turns([view for view in views for _ in chosen]))
        result = pack_batch(held, [window for _ in views for window in chosen], cfg, "cpu")
    LOG.debug("realisations alone exit slots=%d windows=%d seconds=%.4f", len(views), len(chosen), time.perf_counter() - started)
    return result, len(views)


def slotted(held, b):
    started = time.perf_counter()
    found, placed = offered.offered(held), np.asarray(b["placed"])
    rows = np.r_[np.asarray(b["rows"]), np.asarray(b["drawn"])][placed]
    left = np.r_[np.asarray(b["left"], bool), np.zeros(len(b["drawn"]), bool)][placed]
    option = np.where(left, -1, found.row_option[rows])
    result = np.where(option >= 0, option - found.first[np.maximum(found.decision[rows], 0)], -1)
    LOG.debug("realisations slotted exit points=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@pytest.mark.parametrize("family", VIEWS)
def test_each_option_scores_as_it_does_packed_alone_and_one_step_trains_every_family_under_each_shipped_realisation(
    banks, family, monkeypatch
):
    started, largest, plain = time.perf_counter(), {}, {}
    for name, (bank, _) in sorted(SHIPPED.items()):
        held = opened(banks, bank)
        cfg = configured(held, name, modelled(family), PLAIN)
        if name in STRUCTURAL and model_spec.unstructured(cfg["model"]) is not None:
            continue
        torch.manual_seed(5)
        net = network(cfg, held.fitted(cfg))
        blocks = sorted(held.blocks(1, 2, cfg))
        chosen = windows(held, [block for block in blocks if block[0] == blocks[-1][0]][-LAST[name] :], cfg)
        b = pack_batch(held, chosen, cfg, "cpu")
        if name in STRUCTURAL:
            if not plain:
                unrealised = dict(cfg, model=dict(cfg["model"], **{REALISATION: NONE}))
                plain["state"] = captured(net, pack_batch(held, chosen, unrealised, "cpu"))[0]
            assert torch.allclose(captured(net, b)[0], plain["state"], rtol=STATE, atol=STATE), name
        a, slots = alone(held, cfg, chosen, monkeypatch)
        assert len(a["options"]["sequence"]) == len(b["options"]["sequence"]) > len(chosen)
        assert np.bincount(np.asarray(a["options"]["decision"])).max() == 1
        values = net(b)
        with torch.no_grad():
            each = net(a).numpy().reshape(slots, -1)
        slot = slotted(held, b)
        scored = np.flatnonzero(slot >= 0)
        batched, lone = values.detach().numpy()[scored], each[slot[scored], scored]
        largest[name] = float(np.abs(lone - batched).max())
        assert len(set(slot[scored].tolist())) > 1 and np.allclose(lone, batched, rtol=ALONE, atol=ALONE), name
        loss = Summed(dict(decision="squared"), held.registry.schema.targets(), name)(values, b, held)
        loss.backward()
        torch.optim.SGD(net.parameters(), lr=0.01).step()
        reached = {key for key, value in net.head.options.named_parameters() if value.grad is not None and value.grad.abs().sum() > 0}
        wanted = REACHED[name] if name in REACHED else {key for key, _ in net.head.options.named_parameters()}
        assert torch.isfinite(loss) and reached == wanted, (name, sorted(reached ^ wanted))
    reading = model_spec.STRUCTURE[family] is None
    assert sorted(largest) == sorted(name for name in SHIPPED if reading or name not in STRUCTURAL)
    LOG.info("options alone family=%s largest=%s exit seconds=%.3f", family, largest, time.perf_counter() - started)


@pytest.mark.parametrize("name", sorted(SHIPPED))
def test_each_shipped_realisation_passes_every_guard_on_its_shared_bank_and_completes_one_fit(
    banks, name, tmp_path, statement, monkeypatch
):
    started = time.perf_counter()
    bank, unseen = SHIPPED[name]
    held = opened(banks, bank)
    inline(banks[bank], (), monkeypatch)
    cfg = dict(configured(held, name, modelled("snapshot"), PLAIN), epochs=1)
    given = state(dict(statement, loss=dict(decision="squared"), measures=["mse"], alike=0.0))
    fit(banks[bank], tmp_path, cfg, held.blocks(0, 2, cfg), 16, time.perf_counter() + statement["seconds"], given)
    saved = read_json(tmp_path / "result.json")
    found = saved["options"]
    assert saved["status"] == "complete" and saved["optimizer_updates"] > 0 and found["realisation"] == name
    assert found["options"] > 0 and found["pairs"] > 0 and found["share"] == 0.0 and found["unseen"] == unseen
    LOG.info("fit of shipped realisation %s exit seconds=%.3f", name, time.perf_counter() - started)


@timed("realisations.varied")
def varied(base, action, entry, columns):
    stated = copy.deepcopy(base)
    stated["tasks"]["kinds"]["turn"]["target"]["actions"][action] = entry
    stated["columns"].update(columns)
    return Schema(stated)


@timed("realisations.offering")
def offering(held):
    return [parameter for parameter in realisable(held.registry, space(held.registry)) if parameter.name == REALISATION][0]


def test_the_node_realisation_reads_options_of_one_action_whose_one_target_is_a_node_and_refuses_any_other(banks):
    started = time.perf_counter()
    stated, chosen = opened(banks, "stated"), opened(banks, "picked")
    where = "the realisation of options node cannot read the options of this bank: the kind "
    with pytest.raises(ValueError, match=where + "of task turn states the option wait, which does nothing and holds no action"):
        routed(stated, configured(stated, "node", graph=PLAIN))
    routed(chosen, configured(chosen, "node", graph=PLAIN))
    spot, number = dict(sort="node", kinds=["spot"]), dict(type="number", encoder="value")
    base = chosen.registry.schema.mapping()
    routes = "of the kind of task turn has no route to the prediction under the realisation of options node: the realisation"
    for action, entry, columns, message in (
        ("cut", dict(roles=dict(along=dict(spot, sort="edge")), parameters=[]), {}, where + "of action cut of the kind of task turn states the roles along of the sort edge, and the realisation reads options of one action whose one target is a node"),
        ("put", dict(roles=dict(at=spot, made=dict(sort="new", kinds=["mark"])), parameters=[]), {}, where + "of action put of the kind of task turn states the roles at of the sort node, made of the sort new"),
        ("stay", dict(roles=dict(at=spot), parameters=[]), {}, "the field kind " + routes + " reads no kind"),
        ("go", dict(roles=dict(to=spot), parameters=["speed"]), dict(speed=number), "the field parameter speed of go " + routes + " reads no parameters"),
    ):
        assert refused("node", varied(base, action, entry, columns)).startswith(message)
    assert refused("node", chosen.registry.schema) is None and refused("vector", stated.registry.schema) is None
    structure = dict(sequence=torch.zeros(1, dtype=torch.bool), action_option=torch.zeros(2, dtype=torch.long))
    structure.update(role_action=torch.arange(2), role_code=torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError, match="an option of the kind of task turn holds 2 actions, of the kinds go, go, and the realisation reads"):
        Node(chosen.registry)(dict(targets=torch.zeros(2, 4)), structure)
    assert {"node", "deaf"}.isdisjoint(offering(stated).options) and "vector" in offering(stated).options
    assert {"node", "vector", "deaf"} <= set(offering(chosen).options) and offering(chosen).options[0] == NONE
    LOG.info("node realisation refusals exit seconds=%.3f", time.perf_counter() - started)


class Recording(nn.Module):
    @timed("realisations.Recording.__init__")
    def __init__(self):
        super().__init__()
        self.held = []

    def forward(self, sinks, sources, edges, label):
        self.held.append((len(sinks), edges[0].tolist(), edges[1].tolist(), label.tolist()))
        return sinks


@timed("realisations.acted")
def acted(owner, options, codes, sequence, places):
    return dict(
        sequence=torch.tensor([sequence] * options),
        decision=torch.zeros(options, dtype=torch.long),
        action_option=torch.tensor(owner),
        action_place=torch.tensor(places),
        role_action=torch.tensor([action for action, _ in codes]),
        role_code=torch.tensor([code for _, code in codes]),
    )


def test_each_structural_realisation_puts_through_one_channel_the_elements_its_row_of_the_plan_states(banks, monkeypatch):
    started = time.perf_counter()
    held = opened(banks, "stated")
    slot, found = len(coded(held.registry.schema)[1]), {}

    @timed("realisations.recording")
    def recording(spec, hidden, labels):
        found.update(labels=labels, module=Recording())
        return found["module"]

    monkeypatch.setattr(channel, "received", recording)
    targets, actors = [(0, 0), (0, 3), (1, 1)], [(0, 0), (0, 3), (0, slot), (1, 1), (1, slot)]
    wanted = dict(
        action_nodes=(targets, slot, (2, [0, 1, 2], [0, 0, 1], [0, 3, 1])),
        actor_edge=(actors, slot + 1, (3, [0, 1, 3, 2, 2, 4], [0, 1, 2, 0, 1, 2], [0, 3, 1, slot, slot, slot])),
        set_edge=(actors, slot + 1, (2, [0, 1, 2, 3, 4], [0, 0, 0, 1, 1], [0, 3, slot, 1, slot])),
    )
    for name, (codes, labels, joined) in wanted.items():
        module = REALISATIONS.named(name).module(4, 3, dict(view="snapshot"), held.registry)
        fields = dict(kind=torch.ones(2, 4), targets=torch.ones(len(codes), 4), parameters=torch.ones(2, 3))
        options, actions = module(fields, acted([0, 0], 2, codes, False, [0, 1]))
        assert (found["labels"], found["module"].held) == (labels, [joined]), name
        assert options.shape == actions.shape == (2, 4) and torch.equal(options[1], module.empty.detach())
    base = held.registry.schema.mapping()
    bare = varied(base, "stay", dict(roles={}, parameters=[]), {})
    assert refused("actor_edge", bare).startswith(
        "the realisation of options actor_edge cannot read the options of this bank: the kind of action stay of the "
        "kind of task turn states no role, and the realisation joins the actor of an action to each of its targets"
    )
    assert refused("action_nodes", bare) is None and refused("set_edge", bare) is None
    LOG.info("structure of realisations exit seconds=%.3f", time.perf_counter() - started)


def test_through_each_family_that_reads_them_a_set_of_actions_encodes_alike_in_any_order_and_a_sequence_keeps_its_order(banks):
    started = time.perf_counter()
    held = opened(banks, "stated")
    slot, generator = len(coded(held.registry.schema)[1]), torch.Generator().manual_seed(5)
    kind, parameters = torch.randn(2, 8, generator=generator), torch.randn(2, 3, generator=generator)
    own = {0: (torch.randn(1, 8, generator=generator), [0]), 1: (torch.randn(2, 8, generator=generator), [3, 1])}
    actor, picked = torch.randn(1, 8, generator=generator), [0, 1, 1, 0]
    readers = [family for family in VIEWS if model_spec.STRUCTURE[family] is None]
    for family in readers:
        for name in STRUCTURAL:
            spec = configured(held, name, modelled(family), PLAIN)["model"]
            torch.manual_seed(5)
            module = REALISATIONS.named(name).module(8, 3, spec, held.registry).eval()
            codes, parts = [], []
            for at, action in enumerate(picked):
                rows, labels = own[action]
                parts += [rows] + ([actor] if realisation(name).actor else [])
                codes += [(at, code) for code in labels] + ([(at, slot)] if realisation(name).actor else [])
            fields = dict(kind=kind[picked], targets=torch.cat(parts), parameters=parameters[picked])
            encoded = []
            for sequence in (False, True):
                with torch.no_grad():
                    options = module(fields, acted([0, 0, 1, 1], 2, codes, sequence, [0, 1, 0, 1]))[0]
                encoded.append(torch.allclose(options[0], options[1], rtol=1e-6, atol=1e-6))
            assert encoded == [True, False], (family, name)
    LOG.info("order through families exit families=%d seconds=%.3f", len(readers), time.perf_counter() - started)


def test_a_family_that_passes_no_message_over_a_graph_reads_no_structural_realisation_and_a_search_never_offers_it(banks):
    from tuning.searches.selector import excluded, keyed, statics

    started = time.perf_counter()
    held = opened(banks, "stated")
    readers = [family for family in VIEWS if model_spec.STRUCTURE[family] is None]
    assert sorted(model_spec.STRUCTURE) == sorted(VIEWS) and readers == READERS
    assert [name for name in REALISATIONS.shipped if realisation(name).structure] == list(STRUCTURAL)
    for family, name in (("walks", "actor_edge"), ("memory", "set_edge"), ("signal", "action_nodes")):
        cfg = configured(held, name, modelled(family), PLAIN)
        reason = model_spec.STRUCTURE[family]
        stops = "the realisation of options %s puts options into the graph of the family of its model, and the family %s reads no option put there: %s"
        with pytest.raises(ValueError, match="^" + re.escape(stops % (name, family, reason)) + "$"):
            routed(held, cfg)
        with pytest.raises(ValueError, match="^" + re.escape("the family %s reads no option put into its graph: %s" % (family, reason))):
            network(cfg, held.fitted(cfg))
    spatial = dict(modelled("snapshot"), **{"snapshot/spatial": "none"})
    with pytest.raises(ValueError, match="the family snapshot reads no option put there: its spatial layer is none"):
        routed(held, configured(held, "set_edge", spatial, PLAIN))
    rng, parameters = np.random.default_rng(5), realisable(held.registry, space(held.registry))
    drawn = [model_spec.drawn(parameters, lambda parameter: parameter.uniform(rng)) for _ in range(300)]
    assert all(spec[REALISATION] not in STRUCTURAL or model_spec.unstructured(spec) is None for spec in drawn)
    assert {spec["view"] for spec in drawn} == set(VIEWS)
    assert {spec[REALISATION] for spec in drawn if spec["view"] in readers} >= set(STRUCTURAL)
    template = support.template()
    assert excluded(held.registry, configured(held, "actor_edge", modelled("walks"), PLAIN), {}, template) == [REALISATION]
    assert excluded(held.registry, configured(held, "actor_edge", modelled("tree"), PLAIN), {}, template) == []
    static, (choosing,) = statics(parameters, ()), [parameter for parameter in parameters if parameter.name == REALISATION]
    for family, reads in (("walks", False), ("tree", True)):
        fitted = model_spec.fitting(choosing, dict(view=family))
        assert (set(STRUCTURAL) <= set(fitted.options)) == reads and set(STRUCTURAL).isdisjoint(fitted.options) != reads
        assert keyed("model/", fitted, static) == "model/" + REALISATION + ("" if reads else "|" + "|".join(fitted.options))
    LOG.info("structure read by families exit seconds=%.3f", time.perf_counter() - started)


def test_bounds_or_a_random_realisation_that_would_leave_a_family_none_it_reads_are_refused_when_the_search_starts(banks):
    from tuning.searches.selector import check_bounds, excluded, random_suggest
    from tuning.searches.space import Space

    started = time.perf_counter()
    registry, template, fixed = opened(banks, "stated").registry, support.template(), "model/" + REALISATION
    unread = {"snapshot": model_spec.unstructured({"view": "snapshot", "snapshot/spatial": "none"})}
    unread.update((family, reason) for family, reason in model_spec.STRUCTURE.items() if reason is not None)
    families, reasons = ", ".join(unread), "; ".join("%s: %s" % pair for pair in unread.items())
    placing, readers = {REALISATION: ["action_nodes", "set_edge"]}, {"view": READERS, "snapshot/spatial": ["mean", "relational"]}
    bounded = "the bounds leave the model the realisations of options action_nodes, set_edge alone, each putting options into "
    bounded += "the graph of its family, and leave it the families %s, which read no option put there: %s" % (families, reasons)
    with pytest.raises(ValueError, match="^" + re.escape(bounded) + "$"):
        check_bounds(registry, placing, template)
    random = "the random parameter %s fixes one realisation of options for every model of a proposal, and its bounds let it "
    random += "fix action_nodes, actor_edge, set_edge, each putting options into the graph of its family, while they leave the "
    random += "model the families %s, which read no option put there: %s"
    with pytest.raises(ValueError, match="^" + re.escape(random % (fixed, families, reasons)) + "$"):
        Space(registry, {}, 60, (), template).random_names(["model/*"])
    check_bounds(registry, dict(placing, **readers), template)
    assert Space(registry, {REALISATION: [NONE, "vector"]}, 60, (), template).random_names([fixed]) == [fixed]
    space, rng = Space(registry, readers, 60, (), template), np.random.default_rng(5)
    assert space.random_names([fixed]) == [fixed]
    drawn = [(name, space.draw(rng, 7, {fixed: name})["model"]) for name in STRUCTURAL * 8]
    assert all(spec[REALISATION] == name for name, spec in drawn) and {spec["view"] for _, spec in drawn} == set(READERS)
    walks = random_suggest(registry, 3, 7, {"view": ["walks"]}, 60, template)[0]
    assert excluded(registry, walks, dict(placing, **readers), template) == ["view", REALISATION]
    LOG.info("structure refused at the start of a search exit seconds=%.3f", time.perf_counter() - started)


def test_tpe_records_each_family_under_the_realisations_it_reads_while_the_banks_of_decisions_stay_open(banks):
    from tuning.searches.selector import observation

    started = time.perf_counter()
    held = opened(banks, "stated")
    packed(held, "whole", 1)
    template, keys = support.template(), {}
    for family, name in (("tree", "set_edge"), ("walks", "vector")):
        record = observation(held.registry, configured(held, name, modelled(family), PLAIN), 1.0, {}, template)
        (keys[family],) = [key for key in record.params if key.startswith("model/" + REALISATION)]
        assert record.params[keys[family]] == name
    assert keys["tree"] == "model/" + REALISATION and keys["walks"].startswith("model/%s|%s|vector|" % (REALISATION, NONE))
    assert set(STRUCTURAL).isdisjoint(keys["walks"].split("|"))
    LOG.info("tpe records realisations exit seconds=%.3f", time.perf_counter() - started)

