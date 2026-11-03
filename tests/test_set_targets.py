import copy
import time

import numpy as np
import pytest
import support
import torch

from conftest import SPLITS, inline
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from graphspec.infra.hashing import token
from graphspec.infra.timing import timed
from graphspec.schema.schema import VERSION, Schema
from graphspec.schema.targets import TARGETS
from graphspec.sources import synthetic
from tuning.bank.bank import Deriver
from tuning.bank.build import build
from tuning.batches.batching import pack_batch
from tuning.families.model_spec import VIEWS
from tuning.families.models import build as network
from tuning.fits.losses import LOSSES, outcomes
from tuning.fits.training import fit
from tuning.statement import state

pytestmark = [pytest.mark.xdist_group("set_targets"), pytest.mark.extended]


@timed("set_targets.case")
def case():
    target = dict(kind="set", series="team", about=None, sets="team", utility=synthetic.valued(["team"]))
    schema = dict(
        version=VERSION,
        time=dict(model="clock", units=["tick"], ties="key"),
        observers=False,
        columns=dict(value=dict(type="number", encoder="value")),
        graph=dict(columns=[]),
        nodes=dict(person=dict(columns=["value"], holds=None)),
        sets=dict(team=dict(kinds=["person"], roles=["lead", "other"], members=[], columns=[])),
        edges={},
        tasks=dict(
            key="ticket", grouping=None, signals=1,
            references=dict(last=dict(sort="node", kinds=["team"]), who=dict(sort="node", kinds=["person"])),
            kinds=dict(next=dict(references=["last", "who"], columns=[], target=target, weight=False,
                                 absent="stop", event=["who", "last"], matures=None)),
        ),
    )
    people = ["p%d" % at for at in range(6)]
    targets = [dict(lead=[("person", people[at])], other=[("person", people[(at + 3) % 6])]) for at in range(4)]
    graph = dict(
        name="one", unit="tick", columns={}, nodes=dict(person=people, team=["t0", "t1"]),
        values=[
            dict(kind="person", form="state", parameters={}, rows=dict(key=people, value=list(range(6)),
                 since=[0] * 6, until=[None] * 6)),
            dict(kind="team", form="state", parameters={}, rows=dict(key=["t0", "t1"],
                 since=[0, 0], until=[None, None])),
        ],
        edges=[],
        sets=dict(team=dict(set=["t0", "t0", "t1", "t1"], member=people[:4], role=["lead", "other"] * 2)),
        tasks=dict(key=list(range(4)), kind=["next"] * 4, time=[1, 2, 3, 4],
                   split=["train"] * 2 + ["validation"] * 2, weight=[None] * 4,
                   references=dict(last=["t0"] * 4, who=people[:4]), target=targets, known=[2, 3, 4, 5],
                   signal=[[float(at)] for at in range(4)], signal_observed=[[True]] * 4),
    )
    return dict(schema=schema, graphs=[graph])


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    root = tmp_path_factory.mktemp("set-targets")
    path = root / "bank"
    build(synthetic.stage(root, "source", case()), path, 1, ("train",))
    return path, Deriver(path, "cpu", support.SEED, RESIDENT)


@timed("set_targets.configured")
def configured(held, family="snapshot"):
    return support.configuration(
        held.registry, dict(view=family),
        {"time/history": 2, "sets/team/storage": "node", "negatives/of": "set:team",
         "negatives/rule": "uniform", "negatives/count": 2},
        5, copy.deepcopy(SPLITS), support.SEVEN + ("graphspec.negatives",),
    )


def test_sets_round_trip_with_roles_and_member_kinds(bank):
    _, held = bank
    expected = case()["graphs"][0]["tasks"]["target"]
    assert [TARGETS.named("set").read([row]) for row in held.rows] == expected
    assert Schema(held.registry.schema.mapping()).mapping() == held.registry.schema.mapping()
    assert len(held.rows) == 4 and np.all(held.targets == 1)


def test_set_target_refuses_ambiguous_members_and_duplicate_roles():
    entry, target = TARGETS.named("set"), dict(series=("team",))
    for value in ({}, {"lead": []}, {"lead": ["p0"]}, {"lead": [("person", "p0")] * 2}):
        with pytest.raises(ValueError):
            entry.staged(value, target, "test set")
    stated = case()["schema"]
    stated["tasks"]["kinds"]["next"]["target"]["sets"] = "missing"
    with pytest.raises(ValueError, match="declared kind of set edge"):
        Schema(stated)


def test_sampled_sets_are_whole_candidates_with_one_positive_and_no_duplicate_members(bank):
    _, held = bank
    cfg = configured(held)
    blocks = held.blocks(0, 2, cfg)
    batch = pack_batch(held, [blocks[-1]] * 2, cfg, "cpu")
    marks = held.arrays(blocks[-1][0], cfg)["entity_token"]
    target = outcomes(held, batch, "cpu", None)["target"]
    assert len(batch["drawn"]) > 0 and "target" not in batch
    expected = {token("person:p%d" % at) & ((1 << 62) - 1) for at in range(6)}
    for row in batch["point_row"].unique():
        chosen = (batch["point_row"] == row).nonzero().flatten()
        assert target[chosen].sum() == 1 and len(chosen) > 1
        candidates = []
        for point in chosen:
            positions = batch["set_member"][batch["set_point"] == point]
            tokens = marks[batch["member_entity"][positions].numpy() % len(marks)].tolist()
            roles = batch["set_role"][batch["set_point"] == point].tolist()
            assert len(tokens) == 2 and len(set(tokens)) == 2 and set(tokens) <= expected
            candidates.append(tuple(sorted(zip(roles, tokens))))
        assert len(set(candidates)) == len(candidates)


@pytest.mark.parametrize("family", VIEWS)
def test_set_scores_ignore_member_order_and_read_roles_and_members(bank, family):
    _, held = bank
    cfg = configured(held, family)
    batch = pack_batch(held, held.blocks(1, 2, cfg)[-1:], cfg, "cpu")
    torch.manual_seed(9)
    model = network(cfg, held.fitted(cfg))
    model.eval()
    predicted = model(batch)
    reverse = torch.arange(len(batch["set_member"]) - 1, -1, -1)
    permuted = dict(batch, **{name: batch[name][reverse] for name in ("set_member", "set_point", "set_role")})
    torch.testing.assert_close(model(permuted), predicted)
    swapped = dict(batch, set_role=1 - batch["set_role"])
    assert not torch.allclose(model(swapped), predicted)
    changed = dict(batch, set_member=batch["set_member"].roll(1))
    assert not torch.allclose(model(changed), predicted)
    actual = outcomes(held, batch, "cpu", None)
    actual["target"] = actual["target"].to(predicted.dtype)
    loss = LOSSES.named("entropy")(predicted, dict(batch, **actual), held)
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())


def test_whole_sets_complete_a_fit(bank, tmp_path, statement, monkeypatch):
    path, held = bank
    inline(path, (), monkeypatch)
    cfg = dict(configured(held), epochs=1)
    given = state(dict(statement, loss=dict(set="entropy"), measures=["reciprocal", "hits1", "entropy"]))
    fit(path, tmp_path, cfg, held.blocks(0, 2, cfg), 8, time.perf_counter() + 10, given)
    result = read_json(tmp_path / "result.json")
    assert result["status"] == "complete" and 0 < result["optimizer_updates"] <= 20
    assert 0 < result["measures"]["reciprocal"]["value"] <= 1
