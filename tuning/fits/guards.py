import logging
import time

import numpy as np
import torch
from torch import nn

from graphspec.infra.timing import timed
from graphspec.operations.pipeline import emptied
from tuning.batches.batching import batches, pack_batch
from tuning.families.heads import NONE, REALISATION
from tuning.families.model_spec import OPTIONED
from tuning.families.options import FIELDS, coded, fielded, realisation, realised, slots
from tuning.families.periods import blocks as windows

LOG = logging.getLogger(__name__)
CLOSE = 1e-6
STATE = 1e-5


def unrouted(field, entry, removed):
    started = time.perf_counter()
    result = None
    if field["sort"] not in entry.reads:
        result = "the realisation reads no %s" % field["sort"]
    elif field["sort"] == "targets" and field["role"] != "new" and all(kind in removed for kind in field["kinds"]):
        result = "the pipeline removes every kind of node it may name, %s, the entry %s removing %s" % (
            ", ".join(field["kinds"]),
            removed[field["kinds"][0]],
            field["kinds"][0],
        )
    LOG.debug("guards unrouted exit field=%s seconds=%.6f", field["name"], time.perf_counter() - started)
    return result


def routed(bank, cfg):
    started = time.perf_counter()
    name = realised(cfg["model"])
    if name is None:
        LOG.debug("guards routed exit realisation=none seconds=%.4f", time.perf_counter() - started)
        return
    entry = realisation(name)
    if entry.back:
        raise ValueError(
            "the realisation of options %s states that messages pass from options back to the state; no family "
            "gives options a route back to the state, so a fit stating it stops" % name
        )
    if cfg["model"]["view"] not in OPTIONED:
        raise ValueError("the family %s reads no options and the model states the realisation %s" % (cfg["model"]["view"], name))
    schema, removed = bank.registry.schema, emptied(bank.registry, bank.stated(cfg)[1])
    for task in schema.decided():
        for field in fielded(schema, task):
            reason = unrouted(field, entry, removed)
            if reason is not None:
                raise ValueError(
                    "the field %s of the kind of task %s has no route to the prediction under the realisation of "
                    "options %s: %s" % (field["name"], task, name, reason)
                )
    LOG.info("guards routed exit realisation=%s seconds=%.4f", name, time.perf_counter() - started)


class Captured(nn.Module):
    @timed("guards.Captured.__init__")
    def __init__(self):
        super().__init__()
        self.held = None

    def forward(self, features, b, members):
        self.held = features.detach(), members.detach()
        return features.new_zeros(len(b["series"]))


def captured(net, b):
    started = time.perf_counter()
    head, catcher = net.head, Captured()
    net.head = catcher
    try:
        with torch.no_grad():
            net(b)
    finally:
        net.head = head
    LOG.debug("guards captured exit seconds=%.4f", time.perf_counter() - started)
    return catcher.held


@timed("guards.close")
def close(left, right):
    return np.abs(left - right) <= CLOSE * np.maximum(1.0, np.maximum(np.abs(left), np.abs(right)))


class Guard:
    @timed("guards.Guard.__init__")
    def __init__(self, bank, cfg, net, seed):
        self.registry, self.schema = bank.fitted(cfg), bank.registry.schema
        self.name, self.net, self.head, self.seed = realised(cfg["model"]), net, net.head, seed
        self.entry = realisation(self.name)
        self.kinds = list(self.schema.tasks["kinds"])
        self.aims = np.asarray(
            [
                -1 if entry["target"] is None else self.head.targets.index(entry["target"]["kind"])
                for entry in self.schema.tasks["kinds"].values()
            ],
            np.int64,
        )
        self.series = list(self.schema.series())
        self.generator = torch.Generator().manual_seed(seed)
        self.fields = {task: fielded(self.schema, task) for task in self.schema.decided()}
        self.actions = coded(self.schema)[0]
        self.moved = {(task, field["name"]): None for task, found in self.fields.items() for field in found}
        self.pairs, self.equal, self.options, self.sets = 0, 0, 0, 0

    @timed("guards.Guard.scored")
    def scored(self, features, b, fields):
        held = b["options"]
        options, actions = self.head.options.encoded(fields, b)
        task = self.head.task(features)[held["row"]]
        series = self.head.series.weight
        given = torch.cat((task, options), -1)
        values = given.new_zeros((len(options), len(series)))
        aimed = self.aims[np.asarray(held["kind"], np.int64)]
        for code, kind in enumerate(self.head.kinds):
            chosen = np.flatnonzero(aimed == code)
            if not len(chosen) or not self.head.decides[code]:
                continue
            rows = torch.as_tensor(np.repeat(chosen, len(series)), device=given.device)
            columns = torch.arange(len(series), device=given.device).repeat(len(chosen))
            values = values.index_put((rows, columns), kind(given[rows], series[columns], b, None))
        return values, options, actions

    def mask(self, b):
        started = time.perf_counter()
        kinds = np.asarray(b["options"]["kind"], np.int64)
        result = np.zeros((len(kinds), len(self.series)), bool)
        for place, code in enumerate(kinds.tolist()):
            stated = self.schema.tasks["kinds"][self.kinds[code]]["target"]["series"]
            result[place, [self.series.index(name) for name in stated]] = True
        LOG.debug("guard mask exit options=%d seconds=%.4f", len(kinds), time.perf_counter() - started)
        return result

    def state(self, plain, features, b):
        started = time.perf_counter()
        if self.entry.back or torch.allclose(plain, features, rtol=STATE, atol=STATE):
            LOG.debug("guard state exit seconds=%.4f", time.perf_counter() - started)
            return
        row = int(torch.nonzero(~torch.isclose(plain, features, rtol=STATE, atol=STATE).all(-1))[0])
        kind = self.kinds[int(b["row_kind"][b["selected"]][row])]
        raise ValueError(
            "the encoding of the state of a task of the kind %s changes when the options of its decisions are put "
            "beside it, and the realisation of options %s does not state that messages pass from options back to "
            "the state" % (kind, self.name)
        )

    def derived(self, features, b, members, mask):
        started = time.perf_counter()
        held = b["options"]
        fields = {key: value.detach().requires_grad_(True) for key, value in self.head.options.fields(b, members).items()}
        values = self.scored(features, b, fields)[0]
        weights = torch.randn(values.shape, generator=self.generator).to(values.device)
        weights = weights * torch.as_tensor(mask, device=values.device)
        found = torch.autograd.grad((values * weights).sum(), [fields[name] for name in FIELDS], allow_unused=True)
        grads = {
            name: (torch.zeros_like(fields[name]) if grad is None else grad).abs().cpu().numpy()
            for name, grad in zip(FIELDS, found)
        }
        owner = np.asarray(held["kind"], np.int64)
        acting = owner[np.asarray(held["action_option"].cpu(), np.int64)]
        roling = acting[np.asarray(held["role_action"].cpu(), np.int64)]
        kinds, codes = np.asarray(held["action_kind"].cpu(), np.int64), np.asarray(held["role_code"].cpu(), np.int64)
        columns = self.schema.parameters()
        for task, found in self.fields.items():
            code = self.kinds.index(task)
            for field in found:
                if field["sort"] == "kind":
                    rows, part = acting == code, grads["kind"]
                elif field["sort"] == "targets":
                    rows, part = (roling == code) & (codes == field["code"]), grads["targets"]
                else:
                    rows = (acting == code) & (kinds == field["code"])
                    part = grads["parameters"][:, slots(self.registry.encoding, columns, field["column"])]
                if rows.any():
                    key = (task, field["name"])
                    self.moved[key] = bool(self.moved[key]) or bool(part[rows].sum() > 0)
        LOG.debug("guard derived exit options=%d seconds=%.4f", len(owner), time.perf_counter() - started)
        return {name: value.detach() for name, value in fields.items()}

    def apart(self, features, b, fields, values, mask):
        started = time.perf_counter()
        held = b["options"]
        decision = np.asarray(held["decision"].cpu(), np.int64)
        owner = np.asarray(held["action_option"].cpu(), np.int64)
        acted = np.bincount(owner, minlength=len(decision)) > 0
        chosen = np.full(decision.max() + 1 if len(decision) else 0, -1, np.int64)
        for place in np.flatnonzero(acted)[::-1].tolist():
            chosen[decision[place]] = place
        moved = np.zeros(len(decision), bool)
        moved[chosen[chosen >= 0]] = True
        others = ~moved & (chosen[decision] >= 0)
        roles = moved[owner][np.asarray(held["role_action"].cpu(), np.int64)]
        rows = dict(kind=moved[owner], parameters=moved[owner], targets=roles)
        for name in FIELDS:
            changed = dict(fields)
            changed[name] = fields[name] + torch.as_tensor(rows[name], device=fields[name].device)[:, None]
            with torch.no_grad():
                again = self.scored(features, b, changed)[0].cpu().numpy()
            differ = others[:, None] & mask & ~close(again, values)
            if differ.any():
                kind = self.kinds[int(np.asarray(held["kind"])[np.flatnonzero(differ.any(1))[0]])]
                raise ValueError(
                    "the score of an option of the kind of task %s changes when the %s of another option of its "
                    "decision change, and the realisation of options %s does not state that it reads the options of "
                    "a decision together" % (kind, name, self.name)
                )
        LOG.debug("guard apart exit options=%d seconds=%.4f", len(decision), time.perf_counter() - started)

    def ordered(self, b, fields, options, actions):
        started = time.perf_counter()
        held = b["options"]
        owner = np.asarray(held["action_option"].cpu(), np.int64)
        count = np.bincount(owner, minlength=len(options))
        first = np.r_[0, np.cumsum(count)[:-1]].astype(np.int64)
        sets = ~np.asarray(held["sequence"].cpu(), bool) & (count > 1)
        if not sets.any():
            LOG.debug("guard ordered exit sets=0 seconds=%.4f", time.perf_counter() - started)
            return
        place = np.arange(len(owner)) - first[owner]
        turned = np.where(sets[owner], first[owner] + count[owner] - 1 - place, np.arange(len(owner)))
        inverse = np.argsort(turned)
        role = inverse[np.asarray(held["role_action"].cpu(), np.int64)]
        order = np.argsort(role, kind="stable")
        device = fields["kind"].device
        index, by = torch.as_tensor(turned, device=device), torch.as_tensor(order, device=device)
        changed = dict(kind=fields["kind"][index], parameters=fields["parameters"][index], targets=fields["targets"][by])
        turning = dict(held, role_action=torch.as_tensor(role[order], device=device), role_code=held["role_code"][by])
        with torch.no_grad():
            again, acting = self.head.options.encoded(changed, dict(b, options=turning))
        same = torch.isclose(again, options, rtol=CLOSE, atol=CLOSE).all(-1).cpu().numpy()
        each = torch.isclose(acting[torch.as_tensor(inverse, device=device)], actions, rtol=CLOSE, atol=CLOSE)
        np.logical_and.at(same, owner, each.all(-1).cpu().numpy())
        wrong = np.flatnonzero(sets & ~same)
        self.sets += int(sets.sum())
        if len(wrong):
            kind = self.kinds[int(np.asarray(held["kind"])[wrong[0]])]
            named = [self.actions[code][1] for code in np.asarray(held["action_kind"].cpu())[owner == wrong[0]].tolist()]
            raise ValueError(
                "the encoding of an option of the kind of task %s whose actions, %s, are a set changes with the "
                "order of its actions under the realisation of options %s; a set gives the same encoding in any order"
                % (kind, ", ".join(named), self.name)
            )
        LOG.debug("guard ordered exit sets=%d seconds=%.4f", int(sets.sum()), time.perf_counter() - started)

    def alike(self, b, values, mask):
        started = time.perf_counter()
        held = b["options"]
        decision, content = np.asarray(held["decision"].cpu(), np.int64), np.asarray(held["content"], np.int64)
        for chosen in np.split(np.argsort(decision, kind="stable"), np.flatnonzero(np.diff(np.sort(decision))) + 1):
            if len(chosen) < 2:
                continue
            left, right = np.triu_indices(len(chosen), 1)
            left, right = chosen[left], chosen[right]
            differing = content[left] != content[right]
            equal = (close(values[left], values[right]) | ~mask[left]).all(1)
            self.pairs += int(differing.sum())
            self.equal += int((differing & equal).sum())
        LOG.debug("guard alike exit pairs=%d seconds=%.4f", self.pairs, time.perf_counter() - started)

    def checked(self, b, plain):
        started = time.perf_counter()
        features, members = captured(self.net, b)
        self.state(captured(self.net, plain)[0], features, b)
        if len(b["options"]["sequence"]):
            mask = self.mask(b)
            fields = self.derived(features, b, members, mask)
            with torch.no_grad():
                values, options, actions = self.scored(features, b, fields)
            values = values.cpu().numpy()
            if not self.entry.together:
                self.apart(features, b, fields, values, mask)
            self.ordered(b, fields, options, actions)
            self.alike(b, values, mask)
            self.options += len(values)
        LOG.debug("guard checked exit seconds=%.4f", time.perf_counter() - started)


    def concluded(self, alike):
        started = time.perf_counter()
        still = [(task, name) for (task, name), moved in self.moved.items() if moved is False]
        if still:
            task, name = still[0]
            raise ValueError(
                "with the weights drawn by the seed %d, the prediction does not move with the field %s of the kind of "
                "task %s for any option of the splits scored on, under the realisation of options %s"
                % (self.seed, name, task, self.name)
            )
        share = self.equal / self.pairs if self.pairs else None
        if share is not None and share > alike:
            raise ValueError(
                "%d of %d pairs of options of different content of the splits scored on have equal predictions "
                "under the realisation of options %s, a share of %.4f above the statement's alike %g"
                % (self.equal, self.pairs, self.name, share, alike)
            )
        result = dict(
            realisation=self.name,
            options=self.options,
            sets=self.sets,
            pairs=self.pairs,
            alike=self.equal,
            share=share,
            unseen=sorted("%s of %s" % (name, task) for (task, name), moved in self.moved.items() if moved is None),
        )
        LOG.debug("guard concluded exit pairs=%d seconds=%.4f", self.pairs, time.perf_counter() - started)
        return result


def guarded(bank, cfg, net, stated, size):
    started = time.perf_counter()
    if realised(cfg["model"]) is None:
        LOG.debug("guards guarded exit realisation=none seconds=%.4f", time.perf_counter() - started)
        return None
    guard, history = Guard(bank, cfg, net, stated.seed), cfg["graph"]["time/history"]
    plain = dict(cfg, model=dict(cfg["model"], **{REALISATION: NONE}))
    groups = batches(windows(bank, bank.blocks(1, history, cfg), cfg), size, bank.sizes(cfg), None, frame_budget=1024)
    training = net.training
    net.eval()
    try:
        for group in groups:
            guard.checked(pack_batch(bank, group, cfg, stated.device), pack_batch(bank, group, plain, stated.device))
    finally:
        net.train(training)
    result = dict(guard.concluded(stated.alike), seconds=time.perf_counter() - started)
    LOG.info(
        "guards guarded exit realisation=%s pairs=%d alike=%d seconds=%.3f",
        guard.name,
        guard.pairs,
        guard.equal,
        result["seconds"],
    )
    return result
