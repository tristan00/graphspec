import logging
import time

import numpy as np

from graphspec.infra.timing import timed
from graphspec.schema.references import MEMBERS
from graphspec.views.graph import LABELS, PER_STEP, TASKS, Dense, joining

LOG = logging.getLogger(__name__)
OBSERVED = ("obs_entity", "obs_step", "obs_ref", "obs_anchor")
STATED = ("fact_source", "fact_target", "fact_code", "fact_step", "fact_present", "fact_ref")
PARTS = dict(
    entities=("kind", "token"),
    removed=dict(observations=("entity", "step"), facts=("code", "source", "target", "step")),
    added=dict(
        observations=("entity", "step", "x"), facts=("source", "target", "code", "step", "present", "values")
    ),
)


@timed("edits.compact")
def compact(table, reference):
    reference = np.asarray(reference)
    used = np.unique(reference[reference >= 0]).astype(np.int64)
    place = np.searchsorted(used, np.maximum(reference, 0))
    return np.asarray(table)[used], np.where(reference >= 0, place, -1).astype(reference.dtype), used


def cut(graph, low, high):
    started = time.perf_counter()
    if not 0 <= low < high <= graph.steps:
        raise ValueError(
            "a slice of graph %s holds the steps from %d to %d and the graph holds the steps from 0 to %d"
            % (graph.name, low, high, graph.steps)
        )
    arrays = dict(graph.arrays)
    for names, moment in ((OBSERVED, "obs_step"), (STATED, "fact_step")):
        steps = np.asarray(graph.arrays[moment])
        chosen = (steps >= low) & (steps < high)
        for name in names:
            arrays[name] = np.asarray(graph.arrays[name])[chosen]
        arrays[moment] = (arrays[moment] - low).astype(steps.dtype)
    moments = np.asarray(graph.task_step)
    first, last = np.searchsorted(moments, low), np.searchsorted(moments, high)
    for name in TASKS:
        arrays[name] = np.asarray(graph.arrays[name])[first:last]
    arrays["task_step"] = (arrays["task_step"] - low).astype(moments.dtype)
    for names, owner in ((MEMBERS, "member_task"), (LABELS, "label_task")):
        tasks = np.asarray(graph.arrays[owner])
        chosen = (tasks >= first) & (tasks < last)
        for name in names:
            arrays[name] = np.asarray(graph.arrays[name])[chosen]
        arrays[owner] = (arrays[owner] - first).astype(tasks.dtype)
    for name in PER_STEP:
        arrays[name] = np.asarray(graph.arrays[name])[low:high]
    arrays["x_table"], arrays["obs_ref"], rows = compact(graph.x_table, arrays["obs_ref"])
    arrays["value_table"], arrays["fact_ref"], values = compact(graph.value_table, arrays["fact_ref"])
    result = Dense(graph.name, graph.keys, high - low, **arrays)
    LOG.debug(
        "cut exit graph=%s low=%d high=%d seconds=%.4f", graph.name, low, high, time.perf_counter() - started
    )
    return result, rows, values, int(first)


@timed("edits.sliced")
def sliced(graph, high):
    return cut(graph, 0, high)[0]


@timed("edits.columns")
def columns(stated, names, where):
    if type(stated) is not dict or set(stated) != set(names):
        raise ValueError("an edit states %s as a mapping of %s, found %r" % (where, ", ".join(names), stated))
    found = {name: np.asarray(stated[name]) for name in names}
    if len({len(value) for value in found.values()}) > 1:
        raise ValueError("an edit states %s with columns of different lengths" % where)
    return found


class Edit:
    def __init__(self, entities, removed, added):
        started = time.perf_counter()
        if type(removed) is not dict or set(removed) != {"observations", "facts"}:
            raise ValueError("an edit states what it removes as a mapping of observations and facts")
        if type(added) is not dict or set(added) != {"observations", "facts"}:
            raise ValueError("an edit states what it adds as a mapping of observations and facts")
        self.entities = columns(entities, PARTS["entities"], "the entities it adds")
        self.removed = {
            part: columns(removed[part], names, "the %s it removes" % part)
            for part, names in PARTS["removed"].items()
        }
        self.added = {
            part: columns(added[part], names, "the %s it adds" % part) for part, names in PARTS["added"].items()
        }
        LOG.debug("edit exit seconds=%.4f", time.perf_counter() - started)

    @timed("edits.Edit.moments")
    def moments(self):
        return np.concatenate(
            [self.removed[part]["step"] for part in self.removed]
            + [self.added[part]["step"] for part in self.added]
        ).astype(np.int64)

    def moved(self, offset):
        started = time.perf_counter()
        result = Edit(
            self.entities,
            {part: dict(held, step=held["step"] + offset) for part, held in self.removed.items()},
            {part: dict(held, step=held["step"] + offset) for part, held in self.added.items()},
        )
        LOG.debug("moved exit offset=%d seconds=%.4f", offset, time.perf_counter() - started)
        return result


@timed("edits.matched")
def matched(left, right):
    left, right = np.asarray(left, np.int64), np.asarray(right, np.int64)
    if not len(left) or not len(right):
        return np.zeros(len(left), bool), np.zeros(len(right), bool)
    _, inverse = np.unique(np.r_[left, right], axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    size = int(inverse.max()) + 1
    found = np.zeros(size, bool), np.zeros(size, bool)
    found[0][inverse[: len(left)]], found[1][inverse[len(left) :]] = True, True
    return found[1][inverse[: len(left)]], found[0][inverse[len(left) :]]


@timed("edits.ordered")
def ordered(arrays, key):
    order = np.argsort(key, kind="stable")
    return {name: value[order] for name, value in arrays.items()}


@timed("edits.joins")
def joins(edge):
    found = "; ".join("%s to %s" % pair for pair in edge.pairs)
    return found + " in either order" if edge.symmetric else found


def admitted(graph, edit, registry):
    started = time.perf_counter()
    made = np.asarray(edit.entities["kind"], np.int64)
    strange = np.flatnonzero((made < 0) | (made >= len(registry.nodes)))
    if len(strange):
        raise ValueError(
            "an edit of graph %s adds in its row %d of entities an entity of the kind %d, and its registry holds the "
            "kinds 0 to %d" % (graph.name, strange[0], made[strange[0]], len(registry.nodes) - 1)
        )
    kinds = np.r_[np.asarray(graph.kinds, np.int64), made]
    for change, part, names in (
        ("removes", "observations", ("entity",)),
        ("removes", "facts", ("source", "target")),
        ("adds", "observations", ("entity",)),
        ("adds", "facts", ("source", "target")),
    ):
        for name in names:
            held = np.asarray((edit.removed if change == "removes" else edit.added)[part][name], np.int64)
            outside = np.flatnonzero((held < 0) | (held >= len(kinds)))
            if len(outside):
                raise ValueError(
                    "an edit of graph %s names in its row %d of the %s it %s the entity %d as its %s, and the graph "
                    "holds the entities 0 to %d, with the %d the edit adds"
                    % (graph.name, outside[0], part, change, held[outside[0]], name, len(graph.kinds) - 1, len(made))
                )
    added = edit.added["facts"]
    code, source, target = (np.asarray(added[name], np.int64) for name in ("code", "source", "target"))
    unknown = np.flatnonzero((code < 0) | (code >= registry.relations))
    unknown = unknown if len(unknown) else np.flatnonzero(~registry.canonical[code])
    if len(unknown):
        raise ValueError(
            "an edit of graph %s adds in its row %d of facts a fact of the code %d, which names no relation of its "
            "registry; a fact names a relation by its canonical code" % (graph.name, unknown[0], code[unknown[0]])
        )
    nodes = len(registry.nodes)
    lost = np.flatnonzero(~np.isin((code * nodes + kinds[source]) * nodes + kinds[target], joining(registry)))
    if len(lost):
        row = lost[0]
        edge = registry.by_code[int(code[row])]
        raise ValueError(
            "an edit of graph %s adds in its row %d of facts a fact of %s from the entity %d of the kind %s to the "
            "entity %d of the kind %s, and %s joins %s"
            % (
                graph.name,
                row,
                edge.name,
                source[row],
                registry.nodes[kinds[source[row]]],
                target[row],
                registry.nodes[kinds[target[row]]],
                edge.name,
                joins(edge),
            )
        )
    LOG.debug("admitted exit graph=%s facts=%d seconds=%.4f", graph.name, len(code), time.perf_counter() - started)


def edited(graph, edit, registry):
    started = time.perf_counter()
    admitted(graph, edit, registry)
    moments = edit.moments()
    if len(moments) and (moments.min() < 0 or moments.max() >= graph.steps):
        raise ValueError(
            "an edit of graph %s names the steps from %d to %d and the graph holds the steps from 0 to %d"
            % (graph.name, moments.min(), moments.max(), graph.steps - 1)
        )
    kinds, tokens = edit.entities["kind"], edit.entities["token"]
    held = np.stack((np.asarray(graph.kinds, np.int64), np.asarray(graph.entity_token).view(np.int64)), 1)
    fresh = np.stack((kinds.astype(np.int64), tokens.astype(np.uint64).view(np.int64)), 1)
    known, _ = matched(fresh, held)
    if known.any():
        raise ValueError(
            "an edit of graph %s adds the entity of the kind %d and token %d, which the graph holds"
            % (graph.name, kinds[known][0], tokens[known][0])
        )
    arrays = dict(graph.arrays)
    arrays["kinds"] = np.r_[graph.kinds, kinds].astype(graph.kinds.dtype)
    arrays["entity_token"] = np.r_[graph.entity_token, tokens].astype(graph.entity_token.dtype)
    arrays["entity_connected"] = np.r_[graph.entity_connected, np.zeros(len(kinds), bool)]
    removed = edit.removed["observations"]
    present, lost = matched(
        np.stack((removed["step"], removed["entity"]), 1), np.stack((graph.obs_step, graph.obs_entity), 1)
    )
    if not present.all():
        raise ValueError(
            "an edit of graph %s removes the observation of the entity %d at the step %d, which the graph does "
            "not hold" % (graph.name, removed["entity"][~present][0], removed["step"][~present][0])
        )
    removed = edit.removed["facts"]
    named = ("fact_step", "fact_code", "fact_source", "fact_target")
    present, gone = matched(
        np.stack([removed[name.split("_")[1]] for name in named], 1),
        np.stack([graph.arrays[name] for name in named], 1),
    )
    if not present.all():
        raise ValueError(
            "an edit of graph %s removes the fact of the relation %d from %d to %d at the step %d, which the graph "
            "does not hold"
            % tuple([graph.name] + [removed[name][~present][0] for name in ("code", "source", "target", "step")])
        )
    added = edit.added["observations"]
    x = np.asarray(added["x"])
    if x.ndim != 2 or x.shape[1] != graph.x_table.shape[1]:
        raise ValueError(
            "an edit of graph %s adds values of observations of the shape %s and the graph holds them %d wide"
            % (graph.name, x.shape, graph.x_table.shape[1])
        )
    grown = dict(
        obs_entity=added["entity"],
        obs_step=added["step"],
        obs_ref=len(graph.x_table) + np.arange(len(x)),
        obs_anchor=np.zeros(len(x), bool),
    )
    kept = ~lost
    observed = {
        name: np.r_[np.asarray(graph.arrays[name])[kept], grown[name]].astype(graph.arrays[name].dtype)
        for name in OBSERVED
    }
    arrays.update(ordered(observed, observed["obs_step"]))
    arrays["x_table"] = np.concatenate((graph.x_table, x.astype(graph.x_table.dtype)))
    added = edit.added["facts"]
    values = np.asarray(added["values"], np.float64)
    if values.ndim != 2 or values.shape[1] != graph.value_table.shape[1]:
        raise ValueError(
            "an edit of graph %s adds values of facts of the shape %s and the graph holds them %d wide"
            % (graph.name, values.shape, graph.value_table.shape[1])
        )
    valued = ~np.isnan(values).all(1)
    grown = dict(
        fact_source=added["source"],
        fact_target=added["target"],
        fact_code=added["code"],
        fact_step=added["step"],
        fact_present=added["present"].astype(bool),
        fact_ref=np.where(valued, len(graph.value_table) + np.cumsum(valued) - 1, -1),
    )
    kept = ~gone
    stated = {
        name: np.r_[np.asarray(graph.arrays[name])[kept], grown[name]].astype(graph.arrays[name].dtype)
        for name in STATED
    }
    arrays.update(ordered(stated, stated["fact_step"].astype(np.int64) * 2 + ~stated["fact_present"]))
    arrays["value_table"] = np.concatenate((graph.value_table, values[valued].astype(graph.value_table.dtype)))
    result = Dense(graph.name, graph.keys, graph.steps, **arrays)
    LOG.debug(
        "edited exit graph=%s entities=%d observations=%d facts=%d seconds=%.4f",
        graph.name,
        len(kinds),
        len(x),
        len(values),
        time.perf_counter() - started,
    )
    return result
