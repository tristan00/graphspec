import logging
import time
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from graphspec.infra.timing import timed
from graphspec.operations.pipeline import emptied, unrelated
from tuning.bank.forks import Forks, made, trunk
from tuning.bank.options import offered
from tuning.batches.batching import batches, pack_batch, pack_parts
from tuning.families.heads import NONE, REALISATION, coded, fielded, realisation, refused
from tuning.families.model_spec import DIRECTION, NEEDS, edits, ignored, realised, sending, unedited
from tuning.families.options import slots
from tuning.families.periods import blocks as windows

LOG = logging.getLogger(__name__)
CLOSE = 1e-6
STATE = SUMMED = 1e-5
EQUAL, BACK = "equal", "back"
MADE = dict(targets="the end of a fact", kind="the relation of a fact", parameters="a value of a fact")
ORDERED = (
    "the encoding of an option of the kind of task %s whose actions, %s, are a set changes when its actions are %s, "
    "under the realisation of options %s; a set gives the same encoding in any order"
)
APART = (
    "the score of an option of the kind of task %s differs from its score where it is the only option of its "
    "decision, and the realisation of options %s does not state that it reads the options of a decision together"
)
PLACED = (
    "the score of an option of the kind of task %s changes when the options of its decision are %s, under the "
    "realisation of options %s; permuting the options of a decision permutes their scores"
)


@timed("guards.swapped")
def swapped(place, count):
    return np.where(count < 2, place, np.where(place < 2, 1 - place, place))


@timed("guards.rotated")
def rotated(place, count):
    return (place + 1) % count


ORDERS = dict(swapped=swapped, rotated=rotated)


@timed("guards.turned")
def turned(group, step):
    order = np.argsort(group, kind="stable")
    grouped = group[order]
    first = np.searchsorted(grouped, grouped)
    count = np.bincount(grouped)[grouped]
    result = np.arange(len(group))
    result[order] = order[first + step(np.arange(len(order)) - first, count)]
    return result


@timed("guards.places")
def places(group):
    order = np.argsort(group, kind="stable")
    grouped = group[order]
    result = np.empty(len(group), np.int64)
    result[order] = np.arange(len(order)) - np.searchsorted(grouped, grouped)
    return result


def rearranged(held, fields, moved, acts):
    started = time.perf_counter()
    device = held["action_option"].device
    inverse = np.full(len(held["sequence"]), -1, np.int64)
    inverse[moved] = np.arange(len(moved))
    acting = np.full(len(held["action_option"]), -1, np.int64)
    acting[acts] = np.arange(len(acts))
    roled = acting[np.asarray(held["role_action"].cpu(), np.int64)]
    kept = np.flatnonzero(roled >= 0)
    roles = kept[np.argsort(roled[kept], kind="stable")]
    index = {name: torch.as_tensor(found, device=device) for name, found in (("option", moved), ("action", acts), ("role", roles))}
    owner = inverse[np.asarray(held["action_option"].cpu(), np.int64)][acts]
    decision = np.unique(np.asarray(held["decision"].cpu(), np.int64)[moved], return_inverse=True)[1]
    result = dict(
        held,
        row=held["row"][index["option"]],
        decision=torch.as_tensor(decision.reshape(-1), device=device),
        sequence=held["sequence"][index["option"]],
        kind=np.asarray(held["kind"])[moved],
        content=np.asarray(held["content"])[moved],
        action_option=torch.as_tensor(owner, device=device),
        role_action=torch.as_tensor(roled[roles], device=device),
        **{name: held[name][index["action"]] for name in ("action_kind", "action_place", "action_values")},
        **{name: held[name][index["role"]] for name in ("role_code", "role_member", "role_relation")},
    )
    changed = {name: value[index["role" if name == "targets" else "action"]] for name, value in fields.items()}
    LOG.debug("guards rearranged exit options=%d seconds=%.4f", len(moved), time.perf_counter() - started)
    return result, changed


@timed("guards.withheld")
def withheld(field):
    if field["sort"] == "kind":
        return ("kind",)
    if field["sort"] == "targets":
        return ("target", field["code"])
    return ("parameter", field["code"], field["column"])


def cut(bank, entry, task, field, removed, unkept, directed):
    started = time.perf_counter()
    kinds, relations, held, result = field["named"], field["relations"], None, None
    if entry.back:
        schema = bank.registry.schema
        held = entry.mapping[task][field["action"]][field["part"]]
        relations = (held["relation"],) if held["relation"] in schema.edges else ()
        kinds += tuple(schema.tasks["references"][schema.tasks["kinds"][task]["target"]["actors"]]["kinds"])
    elif field["role"] == "new":
        kinds = ()
    lost = [kind for kind in kinds if kind in removed]
    gone = [relation for relation in relations if relation in unkept]
    if lost:
        result = "the pipeline removes the kind of node %s, which it may name, the entry %s removing it" % (
            lost[0],
            removed[lost[0]],
        )
    elif gone:
        step, parameter, value = unkept[gone[0]]
        result = "the pipeline removes the relation %s, which it may name, by %s %s of the entry %s" % (
            gone[0],
            parameter,
            value,
            step,
        )
    elif held is not None and directed and sending(bank.registry, entry, held):
        result = (
            "an edit makes it facts of the relation %s, which has a direction, from the actor, and the setting %s "
            "directed reads a relation from its source to its target alone, so the actor reads none of them"
            % (held["relation"], DIRECTION)
        )
    LOG.debug("guards cut exit field=%s cut=%s seconds=%.4f", field["name"], result is not None, time.perf_counter() - started)
    return result


def routed(bank, cfg):
    started = time.perf_counter()
    name = realised(cfg["model"], bank.registry.schema)
    if name is None:
        LOG.debug("guards routed exit realisation=none seconds=%.4f", time.perf_counter() - started)
        return
    entry, spec = realisation(name), cfg["model"]
    back = entry.back
    reason = ignored(spec, name) if not back or not edits(spec) else None
    if reason is not None:
        raise ValueError(
            "the realisation of options %s puts options into the graph of the family of its model, and the family "
            "%s reads no option put there: %s" % (name, spec["view"], reason)
        )
    schema = bank.registry.schema
    reason = refused(name, bank.registry)
    if reason is not None:
        raise ValueError(reason)
    _, steps, settings, _ = bank.stated(cfg)
    removed, unkept = emptied(bank.registry, steps), unrelated(bank.registry, steps)
    directed = DIRECTION in settings and settings[DIRECTION] == "directed"
    for task in schema.decided():
        for field in fielded(schema, task):
            where = "the field %s of the kind of task %s has no route to the prediction under the realisation of options %s"
            where %= (field["name"], task, name)
            reason = cut(bank, entry, task, field, removed, unkept, directed) if field["sort"] == "targets" else None
            if reason is not None:
                raise ValueError("%s: %s" % (where, reason))
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
def close(left, right, tolerance):
    return np.abs(left - right) <= tolerance * np.maximum(1.0, np.maximum(np.abs(left), np.abs(right)))


class Guard:
    @timed("guards.Guard.__init__")
    def __init__(self, bank, cfg, net, seed):
        self.bank, self.cfg, self.registry, self.schema = bank, cfg, bank.fitted(cfg), bank.registry.schema
        self.name, self.net, self.head, self.seed = realised(cfg["model"], self.schema), net, net.head, seed
        self.splits = cfg["splits"]["scored"]
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
        self.actions, self.roles = coded(self.schema)
        self.moved = {(task, field["name"]): None for task, found in self.fields.items() for field in found}
        self.finite = dict(self.moved)
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

    def located(self, held):
        started = time.perf_counter()
        kind, owner = np.asarray(held["kind"], np.int64), np.asarray(held["action_option"].cpu(), np.int64)
        roled = owner[np.asarray(held["role_action"].cpu(), np.int64)]
        kinds, codes = np.asarray(held["action_kind"].cpu(), np.int64), np.asarray(held["role_code"].cpu(), np.int64)
        columns, result = self.schema.parameters(), []
        for task, found in self.fields.items():
            code = self.kinds.index(task)
            for field in found:
                key = (task, field["name"])
                if field["sort"] == "kind":
                    result.append((key, "kind", kind[owner] == code, owner, None))
                elif field["sort"] == "targets":
                    result.append((key, "targets", (kind[roled] == code) & (codes == field["code"]), roled, None))
                else:
                    place = slots(self.registry.encoding, columns, field["column"])
                    result.append((key, "parameters", (kind[owner] == code) & (kinds == field["code"]), owner, place))
        LOG.debug("guard located exit fields=%d seconds=%.4f", len(result), time.perf_counter() - started)
        return result

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
        for key, name, rows, _, columns in self.located(held):
            if rows.any():
                part = grads[name] if columns is None else grads[name][:, columns]
                self.moved[key] = bool(self.moved[key]) or bool(part[rows].sum() > 0)
        LOG.debug("guard derived exit options=%d seconds=%.4f", len(values), time.perf_counter() - started)
        return {name: value.detach() for name, value in fields.items()}

    def shifted(self, features, b, fields, values, mask):
        started = time.perf_counter()
        held = b["options"]
        decision = np.asarray(held["decision"].cpu(), np.int64)
        for key, name, rows, option, columns in self.located(held):
            if self.finite[key] or not rows.any():
                continue
            first = np.full(decision.max() + 1, len(decision), np.int64)
            np.minimum.at(first, decision[option[rows]], option[rows])
            picked = first[first < len(decision)]
            cells = np.zeros(tuple(fields[name].shape), bool)
            cells[np.ix_(rows & np.isin(option, picked), np.arange(cells.shape[1]) if columns is None else columns)] = True
            moved = torch.randn(cells.shape, generator=self.generator) * torch.as_tensor(cells)
            with torch.no_grad():
                again = self.scored(features, b, dict(fields, **{name: fields[name] + moved.to(fields[name].device)}))[0]
            self.finite[key] = bool((mask[picked] & ~close(again.cpu().numpy()[picked], values[picked], CLOSE)).any())
        LOG.debug("guard shifted exit options=%d seconds=%.4f", len(decision), time.perf_counter() - started)

    def apart(self, features, b, fields, values, mask):
        started = time.perf_counter()
        held = b["options"]
        owner, slot = np.asarray(held["action_option"].cpu(), np.int64), places(np.asarray(held["decision"].cpu(), np.int64))
        alone = np.zeros_like(values)
        for place in np.unique(slot).tolist():
            chosen = np.flatnonzero(slot == place)
            lone, changed = rearranged(held, fields, chosen, np.flatnonzero(np.isin(owner, chosen)))
            with torch.no_grad():
                alone[chosen] = self.scored(features, dict(b, options=lone), changed)[0].cpu().numpy()
        wrong = np.flatnonzero((mask & ~close(alone, values, SUMMED)).any(1))
        if len(wrong):
            raise ValueError(APART % (self.kinds[int(np.asarray(held["kind"])[wrong[0]])], self.name))
        LOG.debug("guard apart exit options=%d slots=%d seconds=%.4f", len(slot), len(np.unique(slot)), time.perf_counter() - started)

    def placed(self, features, b, fields, values, mask):
        started = time.perf_counter()
        held = b["options"]
        decision = np.asarray(held["decision"].cpu(), np.int64)
        owner = np.asarray(held["action_option"].cpu(), np.int64)
        for name, step in ORDERS.items():
            moved = turned(decision, step)
            if (moved == np.arange(len(moved))).all():
                continue
            inverse = np.empty(len(moved), np.int64)
            inverse[moved] = np.arange(len(moved))
            turning, changed = rearranged(held, fields, moved, np.argsort(inverse[owner], kind="stable"))
            with torch.no_grad():
                again = self.scored(features, dict(b, options=turning), changed)[0].cpu().numpy()
            wrong = np.flatnonzero((mask[moved] & ~close(again, values[moved], SUMMED)).any(1))
            if len(wrong):
                raise ValueError(PLACED % (self.kinds[int(np.asarray(turning["kind"])[wrong[0]])], name, self.name))
        LOG.debug("guard placed exit options=%d seconds=%.4f", len(decision), time.perf_counter() - started)

    def ordered(self, b, fields, options, actions):
        started = time.perf_counter()
        held = b["options"]
        owner = np.asarray(held["action_option"].cpu(), np.int64)
        count = np.bincount(owner, minlength=len(options))
        sets = ~np.asarray(held["sequence"].cpu(), bool) & (count > 1)
        if not sets.any():
            LOG.debug("guard ordered exit sets=0 seconds=%.4f", time.perf_counter() - started)
            return
        group = np.where(sets[owner], owner, len(options) + np.arange(len(owner)))
        self.sets += int(sets.sum())
        for name, step in ORDERS.items():
            acts = turned(group, step)
            turning, changed = rearranged(held, fields, np.arange(len(options)), acts)
            with torch.no_grad():
                again, acting = self.head.options.encoded(changed, dict(b, options=dict(turning, action_place=held["action_place"])))
            inverse = torch.as_tensor(np.argsort(acts), device=acting.device)
            same = torch.isclose(again, options, rtol=SUMMED, atol=SUMMED).all(-1).cpu().numpy()
            each = torch.isclose(acting[inverse], actions, rtol=SUMMED, atol=SUMMED)
            np.logical_and.at(same, owner, each.all(-1).cpu().numpy())
            wrong = np.flatnonzero(sets & ~same)
            if len(wrong):
                kind = self.kinds[int(np.asarray(held["kind"])[wrong[0]])]
                named = [self.actions[code][1] for code in np.asarray(held["action_kind"].cpu())[owner == wrong[0]].tolist()]
                raise ValueError(ORDERED % (kind, ", ".join(named), name, self.name))
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
            equal = (close(values[left], values[right], CLOSE) | ~mask[left]).all(1)
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
            self.shifted(features, b, fields, values, mask)
            if not self.entry.together:
                self.apart(features, b, fields, values, mask)
            self.placed(features, b, fields, values, mask)
            self.ordered(b, fields, options, actions)
            self.alike(b, values, mask)
            self.options += len(values)
        LOG.debug("guard checked exit seconds=%.4f", time.perf_counter() - started)


    def banked(self):
        started = time.perf_counter()
        found, columns = offered(self.bank), self.schema.parameters()
        decision = found.decision[self.bank.members(self.cfg, 1)]
        owner, kind = found.action["option"], found.action["kind"]
        scored = np.isin(found.option["decision"], decision[decision >= 0])[owner]
        result = {}
        for task, fields in self.fields.items():
            mine = scored & (found.option["kind"][owner] == self.kinds.index(task))
            for field in fields:
                held = mine
                if field["sort"] == "targets":
                    held = mine & (kind == self.actions.index(self.roles[field["code"]][:2]))
                elif field["sort"] == "parameters":
                    values = found.action["values"][:, columns.index(field["column"])]
                    held = mine & (kind == field["code"]) & ~np.isnan(values)
                result[(task, field["name"])] = len(np.unique(owner[held]))
        LOG.debug("guard banked exit fields=%d seconds=%.4f", len(result), time.perf_counter() - started)
        return result

    def acting(self, task, field):
        started = time.perf_counter()
        targets = [held for held in self.fields[task] if held["sort"] == "targets"]
        if field["sort"] == "targets":
            targets = [field]
        elif field["sort"] == "parameters":
            targets = [held for held in targets if held["action"] == self.actions[field["code"]][1]]
        names = {name for held in targets for name in held["named"] + held["relations"]}
        if self.entry.back:
            names |= {self.entry.mapping[task][held["action"]][held["part"]]["relation"] for held in targets}
        found = []
        for step in self.bank.stated(self.cfg)[1]:
            defaults = {parameter.name: parameter.default for parameter in step.operation.stated(self.bank.registry)}
            for parameter, value in step.parameters.items():
                if names & set(parameter.split("/")) and (parameter not in defaults or defaults[parameter] != value):
                    found.append("%s %s of the entry %s" % (parameter, value, step.name))
        result = "the settings in force on its kinds and relations are %s" % ", ".join(found) if found else (
            "no setting of the pipeline names its kinds or relations"
        )
        LOG.debug("guard acting exit settings=%d seconds=%.4f", len(found), time.perf_counter() - started)
        return result

    def concluded(self, alike):
        started = time.perf_counter()
        if not self.options:
            raise ValueError(
                "the splits scored on, %s, hold no option of a decision under the realisation of options %s, so nothing "
                "can show that the model reads options; a fit stating a realisation scores on options"
                % (", ".join(self.splits), self.name)
            )
        banked = self.banked() if None in self.moved.values() else {}
        still = [(task, name) for (task, name), moved in self.moved.items() if moved is None and banked[task, name]]
        if still:
            task, name = still[0]
            (field,) = [held for held in self.fields[task] if held["name"] == name]
            raise ValueError(
                "the field %s of the kind of task %s is held by %d options of the splits scored on, %s, and by none of "
                "the options the batches of those splits hold under the realisation of options %s, since the variant "
                "shows none of them at the step of its decision: %s"
                % (name, task, banked[task, name], ", ".join(self.splits), self.name, self.acting(task, field))
            )
        still = [(task, name) for (task, name), moved in self.moved.items() if moved is False]
        if still:
            task, name = still[0]
            raise ValueError(
                "with the weights drawn by the seed %d, the prediction does not move with the field %s of the kind of "
                "task %s for any option of the splits scored on, under the realisation of options %s"
                % (self.seed, name, task, self.name)
            )
        still = [(task, name) for (task, name), moved in self.finite.items() if moved is False]
        if still:
            task, name = still[0]
            raise ValueError(
                "with the moves drawn by the seed %d, the score of no option of the splits scored on changes when its "
                "own field %s of the kind of task %s moves by a finite amount, under the realisation of options %s"
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
        self.device, self.found, self.finite = device, offered(bank), self.moved

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
                    self.moved[key] = bool(((~close(again, values[place], CLOSE)) & mask[place]).any())
                    if self.moved[key]:
                        break
        LOG.debug("forked guard derived exit options=%d seconds=%.4f", len(values), time.perf_counter() - started)

    def apart(self, b, values, mask):
        started = time.perf_counter()
        held = b["options"]
        slot, blocks = places(np.asarray(held["decision"], np.int64)), held["block"]
        alone = np.zeros_like(values)
        for place in np.unique(slot).tolist():
            chosen = np.flatnonzero(slot == place)
            alone[chosen] = self.valued(pack_parts(self.bank, [blocks[at] for at in chosen.tolist()], self.cfg, self.device))
        wrong = np.flatnonzero((mask & ~close(alone, values, SUMMED)).any(1))
        if len(wrong):
            raise ValueError(APART % (self.kinds[int(np.asarray(held["kind"])[wrong[0]])], self.name))
        LOG.debug("forked guard apart exit options=%d slots=%d seconds=%.4f", len(slot), len(np.unique(slot)), time.perf_counter() - started)

    def placed(self, b, values, mask):
        started = time.perf_counter()
        held = b["options"]
        decision, blocks = np.asarray(held["decision"], np.int64), held["block"]
        for name, step in ORDERS.items():
            moved = turned(decision, step)
            if (moved == np.arange(len(moved))).all():
                continue
            again = self.valued(pack_parts(self.bank, [blocks[at] for at in moved.tolist()], self.cfg, self.device))
            wrong = np.flatnonzero((mask[moved] & ~close(again, values[moved], SUMMED)).any(1))
            if len(wrong):
                raise ValueError(PLACED % (self.kinds[int(np.asarray(held["kind"])[moved[wrong[0]]])], name, self.name))
        LOG.debug("forked guard placed exit options=%d seconds=%.4f", len(decision), time.perf_counter() - started)

    def ordered(self, b):
        started = time.perf_counter()
        held, found = b["options"], self.found
        for place, option in enumerate(np.asarray(held["option"], np.int64).tolist()):
            first, count = int(found.actions[0][option]), int(found.actions[1][option])
            if bool(held["sequence"][place]) or count < 2:
                continue
            part = held["block"][place][0]
            history = trunk(self.bank, self.cfg, part.graph, part.step)
            stated = np.arange(first, first + count)
            edit = made(self.bank, self.cfg, history, part, stated.tolist())
            self.sets += 1
            for name, step in ORDERS.items():
                again = made(self.bank, self.cfg, history, part, stated[step(np.arange(count), count)].tolist())
                same = all(
                    np.array_equal(edit.added[side][array], again.added[side][array], equal_nan=array in ("x", "values"))
                    for side in edit.added
                    for array in edit.added[side]
                )
                if not same:
                    kind = self.kinds[int(held["kind"][place])]
                    named = [self.actions[code][1] for code in found.action["kind"][first : first + count].tolist()]
                    raise ValueError(
                        "the facts of an option of the kind of task %s whose actions, %s, are a set change when its "
                        "actions are %s, under the realisation of options %s; a set makes the same facts in any order"
                        % (kind, ", ".join(named), name, self.name)
                    )
        LOG.debug("forked guard ordered exit sets=%d seconds=%.4f", self.sets, time.perf_counter() - started)

    def checked(self, b):
        started = time.perf_counter()
        if len(b["options"]["row"]):
            values = self.valued(b)
            mask = self.mask(b)
            self.derived(b, values, mask)
            self.apart(b, values, mask)
            self.placed(b, values, mask)
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
