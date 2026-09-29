import copy
import logging
import time
from types import SimpleNamespace

import numpy as np
import torch

from graphspec import segments, transform
from graphspec.infra.arrays import Bundle, Resident, views
from graphspec.resources import DEFAULT_LIMITS
from graphspec.truth import Dense, Graph

from .batching import recipe_batch
from .data import DTYPES
from .tensors import prepare, resolve
from .turns import blocks as turn_blocks

LOG = logging.getLogger(__name__)
MARK, AT = "#", "@"
OBSERVED = ("obs_entity", "obs_step", "obs_ref", "obs_anchor")
STATED = ("fact_source", "fact_target", "fact_code", "fact_step", "fact_present", "fact_ref")
LABELLED = ("label_entity", "label_step", "label_row", "label_id")
ASKED = ("query_actor", "query_subject", "query_context")
PER_STEP = ("period", "step_campaign", "query_key", "query_category", "query_x", "query_signals") + ASKED
TOLERANCE = 1e-4
WALK, EACH, GROW = 0.01, 0.3, 0.0015


class Memory(dict):
    def read(self, key):
        return self[key]


class Offered:
    def __init__(self, bank, unit, last):
        self.bank, self.unit, self.last, self.held, self.residents = bank, unit, last, {}, {}

    def __getattr__(self, name):
        return getattr(self.bank, name)

    def resident(self, index, cfg, device):
        key = index, str(device)
        if key not in self.residents:
            self.residents[key] = Resident(Memory(self.held[index]), device, DTYPES)
        return self.residents[key]

    def arrays(self, index, cfg):
        return Memory(self.held[index])

    def points(self, index, cfg):
        rows, steps = self.bank.points(self.unit, cfg)
        kept = steps <= self.last
        return rows[kept], steps[kept]

    def tokens(self, held):
        return np.where(self.bank.shared_kinds()[held.numpy("entity_kind")], held.numpy("entity_token"), -1)


def compact(table, reference):
    used = np.unique(reference[reference >= 0])
    place = np.searchsorted(used, np.maximum(reference, 0))
    return np.asarray(table)[used], np.where(reference >= 0, place, -1).astype(reference.dtype)


def sliced(dense, low, high):
    arrays = dict(dense.arrays)
    for names, moment in ((OBSERVED, "obs_step"), (STATED, "fact_step"), (LABELLED, "label_step")):
        steps = np.asarray(dense.arrays[moment])
        chosen = (steps >= low) & (steps < high)
        for name in names:
            arrays[name] = np.asarray(dense.arrays[name])[chosen]
        arrays[moment] = arrays[moment] - low
    for name in PER_STEP:
        arrays[name] = np.asarray(dense.arrays[name])[low:high]
    arrays["x_table"], arrays["obs_ref"] = compact(dense.x_table, arrays["obs_ref"])
    arrays["value_table"], arrays["fact_ref"] = compact(dense.value_table, arrays["fact_ref"])
    return Dense(dense.name, [], high - low, **arrays)


def turn_start(dense, spec, step):
    if spec["time/periods"] == "every":
        return step - step % spec["time/period_length"]
    period = np.asarray(dense.period)
    start = step
    while start > 0 and period[start - 1] == period[step]:
        start -= 1
    return start


def placed(dense, offer):
    known = {
        (int(kind), int(token)): entity
        for entity, (kind, token) in enumerate(zip(np.asarray(dense.kinds), np.asarray(dense.entity_token)))
    }
    kinds, tokens = np.asarray(offer.kinds), np.asarray(offer.entity_token)
    place, fresh = np.empty(len(kinds), np.int64), []
    for entity, key in enumerate(zip(kinds.tolist(), tokens.tolist())):
        if key not in known:
            known[key] = len(dense.kinds) + len(fresh)
            fresh.append(entity)
        place[entity] = known[key]
    return place, kinds[fresh], tokens[fresh]


def swapped(dense, offer, low, step):
    place, kinds, tokens = placed(dense, offer)
    piece = sliced(dense, low, step + 1)
    arrays, last = dict(piece.arrays), step - low
    seen = np.asarray(piece.obs_step) < last
    stated = (np.asarray(piece.fact_step) < last) | ~np.asarray(piece.fact_present)
    order = np.argsort(place[np.asarray(offer.obs_entity)], kind="stable")
    given = dict(
        obs_entity=place[np.asarray(offer.obs_entity)][order],
        obs_step=np.full(len(order), last),
        obs_ref=len(piece.x_table) + np.asarray(offer.obs_ref)[order],
        obs_anchor=np.asarray(offer.obs_anchor)[order],
    )
    for name in OBSERVED:
        arrays[name] = np.r_[arrays[name][seen], given[name]].astype(arrays[name].dtype)
    present = np.asarray(offer.fact_present)
    reference = np.asarray(offer.fact_ref)[present]
    added = dict(
        fact_source=place[np.asarray(offer.fact_source)[present]],
        fact_target=place[np.asarray(offer.fact_target)[present]],
        fact_code=np.asarray(offer.fact_code)[present],
        fact_step=np.full(int(present.sum()), last),
        fact_present=np.ones(int(present.sum()), bool),
        fact_ref=np.where(reference >= 0, len(piece.value_table) + reference, -1),
    )
    earlier = np.asarray(piece.fact_step)[stated] < last
    for name in STATED:
        kept = arrays[name][stated]
        arrays[name] = np.r_[kept[earlier], added[name], kept[~earlier]].astype(arrays[name].dtype)
    for name in PER_STEP:
        if name.startswith("query_"):
            value = np.asarray(offer.arrays[name])[:1]
            arrays[name] = np.concatenate(
                (arrays[name][:last], place[value].astype(value.dtype) if name in ASKED else value)
            )
    arrays["x_table"] = np.concatenate((piece.x_table, np.asarray(offer.x_table)))
    arrays["value_table"] = np.concatenate((piece.value_table, np.asarray(offer.value_table)))
    arrays["kinds"] = np.r_[np.asarray(dense.kinds), kinds].astype(np.asarray(dense.kinds).dtype)
    arrays["entity_token"] = np.r_[np.asarray(dense.entity_token), tokens]
    arrays["entity_connected"] = np.zeros(len(arrays["kinds"]), bool)
    return Dense(dense.name, [], piece.steps, **arrays), kinds, tokens


def recorded(dense, offer, step):
    place, kinds, _ = placed(dense, offer)
    if len(kinds):
        return False
    seen = np.asarray(dense.obs_step) == step
    table, given = np.asarray(dense.x_table), np.asarray(offer.x_table)
    first = np.argsort(np.asarray(dense.obs_entity)[seen], kind="stable")
    second = np.argsort(place[np.asarray(offer.obs_entity)], kind="stable")
    stated = (np.asarray(dense.fact_step) == step) & np.asarray(dense.fact_present)
    present = np.asarray(offer.fact_present)

    def keyed(code, source, target):
        return sorted(zip(code.tolist(), source.tolist(), target.tolist()))

    return (
        np.array_equal(np.asarray(dense.obs_entity)[seen][first], place[np.asarray(offer.obs_entity)][second])
        and np.array_equal(
            table[np.asarray(dense.obs_ref)[seen][first]], given[np.asarray(offer.obs_ref)[second]]
        )
        and keyed(
            np.asarray(dense.fact_code)[stated],
            np.asarray(dense.fact_source)[stated],
            np.asarray(dense.fact_target)[stated],
        )
        == keyed(
            np.asarray(offer.fact_code)[present],
            place[np.asarray(offer.fact_source)[present]],
            place[np.asarray(offer.fact_target)[present]],
        )
    )


def period_groups(period):
    return 1 + int(np.count_nonzero(np.diff(period))) if len(period) else 0


class Line:
    def __init__(self, members, registry, spec, groups, categories):
        self.registry, self.spec, self.groups, self.categories = registry, spec, groups, categories
        if spec["graphs/join"] == "single":
            whole = segments.Plan(members, registry, spec)
            self.maps, self.truth = whole.maps, whole.truth_periods
            self.kinds, self.tokens, self.connected = whole.kinds, whole.tokens, whole.connected
        else:
            (only,) = members
            self.maps, self.truth = [np.arange(len(only.kinds))], np.zeros(2, np.int64)
            self.kinds = np.asarray(only.kinds).astype(np.int64)
            self.tokens = np.asarray(only.entity_token).astype(np.uint64)
            self.connected = np.zeros(len(self.kinds), bool)
        self.steps = self.periods = self.turns = self.rows = self.values = 0
        self.seen = np.zeros(len(self.kinds), bool)
        self.moved, self.shown, self.pieces, self.tables = {}, {}, [], []
        self.context = segments.Context(SimpleNamespace(step_offsets=[0]), spec, groups)

    def fork(self):
        other = copy.copy(self)
        other.seen, other.moved, other.shown = self.seen.copy(), dict(self.moved), dict(self.shown)
        other.context = copy.copy(self.context)
        other.context.attributes = dict(self.context.attributes)
        other.context.turn_starts = list(self.context.turn_starts)
        other.context.turn_ends = list(self.context.turn_ends)
        other.context.history = list(self.context.history)
        other.pieces, other.tables = list(self.pieces), list(self.tables)
        return other

    def add(self, piece, owner, carried, kinds=(), tokens=()):
        entities = len(self.kinds)
        if len(kinds):
            self.kinds = np.r_[self.kinds, np.asarray(kinds, np.int64)]
            self.tokens = np.r_[self.tokens, np.asarray(tokens, np.uint64)]
            self.connected = np.r_[self.connected, np.zeros(len(kinds), bool)]
            self.seen = np.r_[self.seen, np.zeros(len(kinds), bool)]
        every = self.spec["time/periods"] == "every"
        plan = SimpleNamespace(
            maps=[np.r_[self.maps[owner], entities + np.arange(len(kinds))]],
            kinds=self.kinds,
            tokens=self.tokens,
            connected=self.connected,
            step_offsets=[self.steps],
            truth_periods=[self.truth[owner]],
            period_offsets=[self.periods if every else self.truth[owner]],
            group_offsets=[self.periods if every else self.turns],
            x_offsets=[self.rows, self.rows + len(piece.x_table)],
            value_offsets=[self.values, self.values + len(piece.value_table)],
            seen=self.seen,
        )
        carry, dense = segments.segment(plan, 0, piece, self.moved)
        if carried:
            dense = dense.replace(entity_connected=np.ones(len(dense.kinds), bool))
        known = self.seen[carry.glob]
        self.seen[carry.glob[dense.obs_entity]] = True
        graph = transform.derive(dense, self.registry, self.spec, DEFAULT_LIMITS, carry, known)
        self.seen[carry.glob[graph.obs_entity]] = True
        self.context.steps = self.steps + piece.steps
        self.pieces.append(
            segments.represent_segment(
                carry,
                graph,
                self.registry,
                self.spec,
                self.groups,
                self.categories,
                transform.SEED,
                DEFAULT_LIMITS,
                self.shown,
                self.context,
            )
        )
        self.tables.append(np.asarray(graph.x_table)[: len(piece.x_table)])
        self.steps += piece.steps
        self.rows += len(piece.x_table)
        self.values += len(piece.value_table)
        self.periods += -(-piece.steps // self.spec["time/period_length"]) if every else 0
        self.turns += period_groups(piece.period)

    def arrays(self):
        names = [name for name in self.pieces[0] if name != "mixed_member_offsets"]
        found = {name: np.concatenate([piece[name] for piece in self.pieces]) for name in names}
        if "mixed" in self.groups:
            found["mixed_member_offsets"] = np.concatenate(
                [piece["mixed_member_offsets"] for piece in self.pieces]
                + [np.asarray([self.context.holders0], np.int64)]
            )
        plan = SimpleNamespace(kinds=self.kinds, tokens=self.tokens)
        return segments.stitched_arrays(
            plan, found, self.tables, self.registry, self.spec, self.groups, DEFAULT_LIMITS, self.context
        )


def decisions(bank):
    started = time.perf_counter()
    index = {graph["name"]: graph["index"] for graph in bank.manifest["graphs"]}
    found = {}
    for name, graph in index.items():
        if AT in name:
            base, rest = name.split(AT)
            key, block = rest.split(MARK)
            found.setdefault((index[base], int(key)), {})[int(block)] = graph
    LOG.info("decisions exit seconds=%.3f decisions=%d", time.perf_counter() - started, len(found))
    return found


def packed(source, blocks, cfg):
    bundle = Bundle("cpu")
    tree = prepare(recipe_batch(source, blocks, cfg), bundle)
    arrays, spans, size = bundle.layout()
    buffer = torch.zeros(size, dtype=torch.uint8)
    bundle.write(buffer.numpy(), arrays)
    return resolve(tree, source, cfg, views(buffer, spans), "cpu")


def ordered(bank, cfg, history):
    wanted = decisions(bank)
    joining = bank.joining(cfg)
    steps = {graph["index"]: graph["steps"] for graph in bank.manifest["graphs"]}
    found, reach = [], []
    for unit in sorted({block[0] for block in bank.blocks(1, history, cfg)}):
        members = bank.groups(cfg)[unit] if joining else [unit]
        before = 0
        for position, member in enumerate(members):
            keys = sorted(key for owner, key in wanted if owner == member)
            found.extend((unit, position, key) for key in keys)
            reach.extend(before + steps[member] * (rank + 1) / (len(keys) + 1) for rank in range(len(keys)))
            before += steps[member]
    return found, reach


def filled(chosen, reach, limit):
    shares, spent, unit, last = [[]], 0.0, None, 0.0
    for item, steps in zip(chosen, reach):
        cost = EACH + GROW * steps + WALK * (steps - last if item[0] == unit else steps)
        if shares[-1] and spent + cost > limit:
            shares.append([])
            spent, cost = 0.0, EACH + GROW * steps + WALK * steps
        shares[-1].append(item)
        spent, unit, last = spent + cost, item[0], steps
    return shares


def shared(chosen, reach, count):
    low, high = 0.0, sum(EACH + (GROW + WALK) * steps for steps in reach)
    for _ in range(40):
        middle = (low + high) / 2
        if len(filled(chosen, reach, middle)) <= count:
            high = middle
        else:
            low = middle
    return filled(chosen, reach, high)


def batches(bank, cfg, history, chosen):
    started = time.perf_counter()
    _, spec, groups = bank.variant(cfg)
    wanted = decisions(bank)
    joining = bank.joining(cfg)
    windows = turn_blocks(bank, bank.blocks(1, history, cfg), cfg)
    found = []
    for unit in sorted({unit for unit, _, _ in chosen}):
        members = bank.groups(cfg)[unit] if joining else [unit]
        here = [item for item in chosen if item[0] == unit]
        final = max(position for _, position, _ in here)
        loaded = [Graph.load(bank.truth(member)).dense() for member in members[: final + 1]]
        offsets = np.r_[0, np.cumsum([graph.steps for graph in loaded])]
        line = Line(
            [Graph.load(bank.truth(member), mapped=True) for member in members],
            bank.registry,
            spec,
            groups,
            bank.categories,
        )
        for position, dense in enumerate(loaded):
            keys, done = np.asarray(dense.query_key), 0
            for key in [key for _, owner, key in here if owner == position]:
                step = int(np.flatnonzero(keys == key)[0])
                moment = int(offsets[position]) + step
                low = turn_start(dense, spec, step)
                if low > done:
                    line.add(sliced(dense, done, low), position, True)
                    done = low
                source = Offered(bank, unit, moment)
                offers = wanted[members[position], key]
                for block in sorted(offers):
                    offer = Graph.load(bank.truth(offers[block])).dense()
                    branch = line.fork()
                    if block == 0:
                        if not recorded(dense, offer, step):
                            raise ValueError(
                                "the stored block of %s at %d differs from its run" % (dense.name, key)
                            )
                        branch.add(sliced(dense, done, step + 1), position, True)
                    else:
                        piece, kinds, tokens = swapped(dense, offer, done, step)
                        branch.add(piece, position, True, kinds, tokens)
                    source.held[block] = branch.arrays()
                window = next(item for item in windows if item[0] == unit and item[3] == moment)
                batch = packed(source, [(block,) + tuple(window[1:]) for block in sorted(offers)], cfg)
                batch.update(graph=unit, graphs=[unit] * len(offers))
                found.append((len(offers), batch))
            if position < final:
                line.add(sliced(dense, done, dense.steps), position, False)
    LOG.info("offers batches exit seconds=%.3f decisions=%d", time.perf_counter() - started, len(found))
    return found


def moved(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: moved(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(moved(item, device) for item in value)
    return value


def predict(bank, cfg, net, device, history, packer):
    started = time.perf_counter()
    shares = shared(*ordered(bank, cfg, history), len(packer.processes))
    count = len(shares)
    rows, values, spent = [], [], 0.0
    for share in packer.offered(shares, history):
        for blocks, batch in share:
            clock = time.perf_counter()
            batch = moved(batch, device)
            with torch.inference_mode():
                found = net(batch)
            if not torch.isfinite(found).all():
                raise FloatingPointError("nonfinite predictions for drawn blocks")
            center = found.new_tensor(bank.center)[batch["series"]]
            scale = found.new_tensor(bank.scale)[batch["series"]]
            rows.append(np.asarray(batch["rows"]).reshape(blocks, -1))
            values.append((found * scale + center).cpu().numpy().reshape(blocks, -1))
            spent += time.perf_counter() - clock
    LOG.info(
        "offers predict exit seconds=%.3f decisions=%d workers=%d forward=%.2f",
        time.perf_counter() - started,
        len(rows),
        count,
        spent,
    )
    return rows, values


def spread(bank, rows, values, held, predictions):
    started = time.perf_counter()
    known = dict(zip(held.tolist(), predictions.tolist()))
    found = {name: [] for name in bank.series_names}
    worst = 0.0
    for points, paths in zip(rows, values):
        points = np.asarray(points)
        if not (points == points[0]).all():
            raise ValueError("drawn blocks of one decision were predicted at different points")
        expected = np.asarray([known[int(row)] for row in points[0]])
        scale = bank.scale[bank.series[points[0]]]
        worst = max(worst, float((np.abs(paths[0] - expected) / scale).max()))
        for code, name in enumerate(bank.series_names):
            member = bank.series[points[0]] == code
            if member.any():
                found[name].append(float(np.ptp(paths[:, member], 0).mean()) / float(bank.scale[code]))
    if worst > TOLERANCE:
        raise ValueError("the recorded block does not reproduce its validation prediction: gap %.3g" % worst)
    LOG.info(
        "offers spread exit seconds=%.3f decisions=%d recorded_gap=%.3g",
        time.perf_counter() - started,
        len(rows),
        worst,
    )
    return {name: float(np.mean(entries)) for name, entries in found.items()}, worst
