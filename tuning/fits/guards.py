import logging
import time
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from graphspec.infra.timing import timed
from graphspec.operations.pipeline import emptied
from tuning.bank.forks import Forks, made, trunk
from tuning.bank.options import offered
from tuning.batches.batching import batches, pack_batch, pack_parts
from tuning.families.heads import NONE, REALISATION, coded, fielded, realisation, refused
from tuning.families.model_spec import NEEDS, edits, ignored, realised, unedited
from tuning.families.options import slots
from tuning.families.periods import blocks as windows

LOG = logging.getLogger(__name__)
CLOSE = 1e-6
STATE = 1e-5
EQUAL, BACK = "equal", "back"
MADE = dict(targets="the end of a fact", kind="the relation of a fact", parameters="a value of a fact")


@timed("guards.withheld")
def withheld(field):
    if field["sort"] == "kind":
        return ("kind",)
    if field["sort"] == "targets":
        return ("target", field["code"])
    return ("parameter", field["code"], field["column"])


def routed(bank, cfg):
    started = time.perf_counter()
    name = realised(cfg["model"], bank.registry.schema)
    if name is None:
        LOG.debug("guards routed exit realisation=none seconds=%.4f", time.perf_counter() - started)
        return
    back, spec = realisation(name).back, cfg["model"]
    reason = ignored(spec, name) if not back or not edits(spec) else None
    if reason is not None:
        raise ValueError(
            "the realisation of options %s puts options into the graph of the family of its model, and the family "
            "%s reads no option put there: %s" % (name, spec["view"], reason)
        )
    schema = bank.registry.schema
    reason = refused(name, schema)
    if reason is not None:
        raise ValueError(reason)
    removed = emptied(bank.registry, bank.stated(cfg)[1])
    for task in schema.decided():
        for field in fielded(schema, task):
            where = "the field %s of the kind of task %s has no route to the prediction under the realisation of options %s"
            where %= (field["name"], task, name)
            if field["sort"] == "targets" and field["role"] != "new" and all(kind in removed for kind in field["kinds"]):
                raise ValueError(
                    "%s: the pipeline removes every kind of node it may name, %s, the entry %s removing %s"
                    % (where, ", ".join(field["kinds"]), removed[field["kinds"][0]], field["kinds"][0])
                )
            if back and NEEDS[field["sort"]] not in edits(spec):
                raise ValueError(
                    "%s: an edit makes it %s, which the family %s does not read; %s"
                    % (where, MADE[field["sort"]], spec["view"], unedited(spec))
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
        self.name, self.net, self.head, self.seed = realised(cfg["model"], self.schema), net, net.head, seed
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
        if torch.allclose(plain, features, rtol=STATE, atol=STATE):
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
        found = torch.autograd.grad((values * weights).sum(), list(fields.values()), allow_unused=True)
        grads = {
            name: (torch.zeros_like(fields[name]) if grad is None else grad).abs().cpu().numpy()
            for name, grad in zip(fields, found)
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

    def kept(self, features, b, changed, values, mask, others, name):
        started = time.perf_counter()
        with torch.no_grad():
            again = self.scored(features, b, changed)[0].cpu().numpy()
        differ = others[:, None] & mask & ~close(again, values)
        if differ.any():
            kind = self.kinds[int(np.asarray(b["options"]["kind"])[np.flatnonzero(differ.any(1))[0]])]
            raise ValueError(
                "the score of an option of the kind of task %s changes when the %s of another option of its "
                "decision change, and the realisation of options %s does not state that it reads the options of "
                "a decision together" % (kind, name, self.name)
            )
        LOG.debug("guard kept exit moved=%s seconds=%.4f", name, time.perf_counter() - started)

    def apart(self, features, b, members, fields, values, mask):
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
        for name in fields:
            changed = dict(fields)
            changed[name] = fields[name] + torch.as_tensor(rows[name], device=fields[name].device)[:, None]
            self.kept(features, b, changed, values, mask, others, name)
        ends = held["role_member"][torch.as_tensor(roles, device=members.device)].flatten()
        ends = ends[ends >= 0]
        shifted = members.index_add(0, ends, members.new_ones((len(ends), members.shape[1])))
        with torch.no_grad():
            changed = self.head.options.fields(b, shifted)
        self.kept(features, b, changed, values, mask, others, "members named by the targets")
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
        device = held["action_option"].device
        index, by = torch.as_tensor(turned, device=device), torch.as_tensor(order, device=device)
        changed = {name: value[by if name == "targets" else index] for name, value in fields.items()}
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
        decision, content = np.asarray(torch.as_tensor(held["decision"]).cpu(), np.int64), np.asarray(held["content"], np.int64)
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
                self.apart(features, b, members, fields, values, mask)
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
            state=BACK if self.entry.back else EQUAL,
        )
        LOG.debug("guard concluded exit pairs=%d seconds=%.4f", self.pairs, time.perf_counter() - started)
        return result


class Forked(Guard):
    @timed("guards.Forked.__init__")
    def __init__(self, bank, cfg, net, seed, device):
        super().__init__(bank, cfg, net, seed)
        self.bank, self.cfg, self.device, self.found = bank, cfg, device, offered(bank)
        self.roles = coded(self.schema)[1]

    def valued(self, b):
        started = time.perf_counter()
        held = b["options"]
        with torch.no_grad():
            features = captured(self.net, b)[0]
            task = self.head.task(features)[torch.as_tensor(held["row"], device=features.device)]
            series = self.head.series.weight
            values = task.new_zeros((len(task), len(series)))
            aimed = self.aims[np.asarray(held["kind"], np.int64)]
            for code, kind in enumerate(self.head.kinds):
                chosen = np.flatnonzero(aimed == code)
                if not len(chosen):
                    continue
                rows = torch.as_tensor(np.repeat(chosen, len(series)), device=task.device)
                columns = torch.arange(len(series), device=task.device).repeat(len(chosen))
                values = values.index_put((rows, columns), kind(task[rows], series[columns], b, None))
        LOG.debug("forked guard valued exit options=%d seconds=%.4f", len(task), time.perf_counter() - started)
        return values.cpu().numpy()

    def holding(self, option, task, field):
        started = time.perf_counter()
        found = self.found
        first, count = int(found.actions[0][option]), int(found.actions[1][option])
        kinds = found.action["kind"][first : first + count]
        result = int(found.option["kind"][option]) == self.kinds.index(task) and count > 0
        if result and field["sort"] == "targets":
            owner = self.actions.index(self.roles[field["code"]][:2])
            result = bool((kinds == owner).any())
        elif result and field["sort"] == "parameters":
            column = self.schema.parameters().index(field["column"])
            values = found.action["values"][first : first + count][kinds == field["code"], column]
            result = bool((~np.isnan(values)).any())
        LOG.debug("forked guard holding exit option=%d held=%s seconds=%.4f", option, result, time.perf_counter() - started)
        return result

    def moving(self, block, move):
        started = time.perf_counter()
        moved = (replace(block[0], move=move),) + tuple(block[1:])
        result = self.valued(pack_parts(self.bank, [moved], self.cfg, self.device))
        LOG.debug("forked guard moving exit option=%d seconds=%.4f", block[0].option, time.perf_counter() - started)
        return result

    def derived(self, b, values, mask):
        started = time.perf_counter()
        held = b["options"]
        for task, found in self.fields.items():
            for field in found:
                key = (task, field["name"])
                for place in range(len(held["row"])) if not self.moved[key] else ():
                    if not self.holding(int(held["option"][place]), task, field):
                        continue
                    again = self.moving(held["block"][place], withheld(field))[0]
                    self.moved[key] = bool(((~close(again, values[place])) & mask[place]).any())
                    if self.moved[key]:
                        break
        LOG.debug("forked guard derived exit options=%d seconds=%.4f", len(values), time.perf_counter() - started)

    def apart(self, b, values, mask):
        started = time.perf_counter()
        held = b["options"]
        decision, chosen = np.asarray(held["decision"], np.int64), {}
        for place in range(len(decision)):
            if int(decision[place]) not in chosen and int(self.found.actions[1][int(held["option"][place])]) > 0:
                chosen[int(decision[place])] = place
        moved = set(chosen.values())
        others = np.asarray([place not in moved and int(decision[place]) in chosen for place in range(len(decision))])
        if not others.any():
            LOG.debug("forked guard apart exit decisions=0 seconds=%.4f", time.perf_counter() - started)
            return
        blocks = [
            (replace(block[0], move=("kind",)),) + tuple(block[1:]) if place in moved else block
            for place, block in enumerate(held["block"])
        ]
        again = self.valued(pack_parts(self.bank, blocks, self.cfg, self.device))
        differ = others[:, None] & mask & ~close(again, values)
        if differ.any():
            kind = self.kinds[int(np.asarray(held["kind"])[np.flatnonzero(differ.any(1))[0]])]
            raise ValueError(
                "the score of an option of the kind of task %s changes when the facts of another option of its "
                "decision are withheld, and the realisation of options %s does not state that it reads the options "
                "of a decision together" % (kind, self.name)
            )
        LOG.debug("forked guard apart exit decisions=%d seconds=%.4f", len(chosen), time.perf_counter() - started)

    def ordered(self, b):
        started = time.perf_counter()
        held, found = b["options"], self.found
        for place, option in enumerate(np.asarray(held["option"], np.int64).tolist()):
            first, count = int(found.actions[0][option]), int(found.actions[1][option])
            if bool(held["sequence"][place]) or count < 2:
                continue
            part = held["block"][place][0]
            history = trunk(self.bank, self.cfg, part.graph, part.step)
            stated = list(range(first, first + count))
            edits = [made(self.bank, self.cfg, history, part, order) for order in (stated, stated[::-1])]
            same = all(
                np.array_equal(edits[0].added[side][name], edits[1].added[side][name], equal_nan=name in ("x", "values"))
                for side in edits[0].added
                for name in edits[0].added[side]
            )
            self.sets += 1
            if not same:
                kind = self.kinds[int(held["kind"][place])]
                named = [self.actions[code][1] for code in found.action["kind"][first : first + count].tolist()]
                raise ValueError(
                    "the facts of an option of the kind of task %s whose actions, %s, are a set change with the order "
                    "of its actions under the realisation of options %s; a set makes the same facts in any order"
                    % (kind, ", ".join(named), self.name)
                )
        LOG.debug("forked guard ordered exit sets=%d seconds=%.4f", self.sets, time.perf_counter() - started)

    def checked(self, b):
        started = time.perf_counter()
        if len(b["options"]["row"]):
            values = self.valued(b)
            mask = self.mask(b)
            self.derived(b, values, mask)
            self.apart(b, values, mask)
            self.ordered(b)
            self.alike(b, values, mask)
            self.options += len(values)
        LOG.debug("forked guard checked exit seconds=%.4f", time.perf_counter() - started)


def guarded(bank, cfg, net, stated, size):
    started = time.perf_counter()
    name = realised(cfg["model"], bank.registry.schema)
    if name is None:
        LOG.debug("guards guarded exit realisation=none seconds=%.4f", time.perf_counter() - started)
        return None
    back, history = realisation(name).back, cfg["graph"]["time/history"]
    guard = Forked(bank, cfg, net, stated.seed, stated.device) if back else Guard(bank, cfg, net, stated.seed)
    plain = dict(cfg, model=dict(cfg["model"], **{REALISATION: NONE}))
    groups = batches(windows(bank, bank.blocks(1, history, cfg), cfg), size, bank.sizes(cfg), None, frame_budget=1024)
    training = net.training
    net.eval()
    try:
        for group in groups:
            if back:
                with Forks(bank):
                    guard.checked(pack_batch(bank, group, cfg, stated.device))
            else:
                guard.checked(pack_batch(bank, group, cfg, stated.device), pack_batch(bank, group, plain, stated.device))
    finally:
        net.train(training)
    if back:
        LOG.info(
            "guards guarded state realisation=%s: messages pass from options back to the state, so the state of an "
            "option is its own fork's and is not compared with the state without options",
            name,
        )
    result = dict(guard.concluded(stated.alike), seconds=time.perf_counter() - started)
    LOG.info(
        "guards guarded exit realisation=%s pairs=%d alike=%d seconds=%.3f",
        guard.name,
        guard.pairs,
        guard.equal,
        result["seconds"],
    )
    return result
