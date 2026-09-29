import logging
import time

import numpy as np

from graphspec.infra.timing import timed

from . import clock, spans
from .clock import ALWAYS, NEVER, SEVERAL
from .names import Names
from .truth import GROUPS, Graph, encoded, ranks, verify

LOG = logging.getLogger(__name__)
STEPS = Names("choice of steps")


@timed("view.sequence")
def sequence(observer, moments, looking, queries, labels, label_step):
    order = np.argsort(label_step, kind="stable")
    return dict(
        observer=int(observer),
        moments=np.asarray(moments, np.int64),
        looking=np.asarray(looking, np.int64),
        queries=np.asarray(queries, np.int64),
        labels=np.asarray(labels, np.int64)[order],
        label_step=np.asarray(label_step, np.int32)[order],
    )


@timed("view.at")
def at(graph, moments, times):
    found = np.searchsorted(moments, times, "right") - 1
    absent = (found < 0) | (moments[np.maximum(found, 0)] != times)
    if absent.any():
        raise ValueError(
            "a label of graph %s sits at %s, which no step of the view shows"
            % (graph, clock.shown(np.asarray(times)[absent][0]))
        )
    return found


class Tasks:
    @timed("view.tasks.serves")
    def serves(self, registry):
        return bool(registry.extras["queries"])

    def __call__(self, facts, registry):
        started = time.perf_counter()
        looking = np.asarray(facts.query_observer)
        if not len(looking):
            raise ValueError("graph %s holds no task, and the steps of the view are its tasks" % facts.name)
        owner, moments = np.asarray(facts.label_query), np.asarray(facts.query_time)
        loose = np.flatnonzero(owner < 0)
        if len(loose) and np.any(looking >= 0):
            raise ValueError(
                "graph %s holds targets on nodes beside tasks that name observers; "
                "a target on a node names no observer" % facts.name
            )
        result = []
        for observer in np.unique(looking).tolist():
            chosen = np.flatnonzero(looking == observer)
            place = np.full(len(looking), -1, np.int64)
            place[chosen] = np.arange(len(chosen))
            held = np.flatnonzero((owner >= 0) & (place[np.maximum(owner, 0)] >= 0))
            labels = np.r_[held, loose]
            steps = np.r_[
                place[owner[held]], at(facts.name, moments[chosen], np.asarray(facts.label_time)[loose])
            ]
            result.append(sequence(observer, moments[chosen], moments[chosen], chosen, labels, steps))
        LOG.debug(
            "steps of tasks exit graph=%s sequences=%d seconds=%.4f",
            facts.name,
            len(result),
            time.perf_counter() - started,
        )
        return result


class Changes:
    @timed("view.changes.serves")
    def serves(self, registry):
        return not registry.extras["queries"]

    def __call__(self, facts, registry):
        started = time.perf_counter()
        if len(facts.query_time):
            raise ValueError(
                "graph %s holds tasks; a view whose steps are the times of change shows none" % facts.name
            )
        times = [np.asarray(facts.label_time, np.int64)]
        for group in GROUPS:
            times += [
                np.asarray(facts.arrays[group + "_" + name], np.int64)
                for name in ("since", "until", "known_time", "replaced_time")
            ]
        times = np.concatenate(times)
        moments = np.unique(times[(times > ALWAYS) & (times < NEVER)])
        labels = np.arange(len(facts.label_time))
        steps = at(facts.name, moments, np.asarray(facts.label_time))
        result = [sequence(-1, moments, moments, np.zeros(0, np.int64), labels, steps)]
        LOG.debug(
            "steps of changes exit graph=%s steps=%d seconds=%.4f",
            facts.name,
            len(moments),
            time.perf_counter() - started,
        )
        return result


STEPS.register("tasks", Tasks())
STEPS.register("changes", Changes())


@timed("view.offered")
def offered(registry):
    return tuple(name for name in STEPS.names() if STEPS.named(name).serves(registry))


def sequences(facts, registry, steps):
    started = time.perf_counter()
    chooser = STEPS.named(steps)
    if not chooser.serves(registry):
        raise ValueError("the choice of steps %s does not serve the graphs of this bank" % steps)
    result = chooser(facts, registry)
    placed = np.concatenate([held["labels"] for held in result] + [np.zeros(0, np.int64)])
    if not np.array_equal(np.sort(placed), np.arange(len(facts.label_time))):
        raise ValueError(
            "the choice of steps %s does not give every label of graph %s one step" % (steps, facts.name)
        )
    LOG.debug(
        "sequences exit graph=%s steps=%s sequences=%d seconds=%.4f",
        facts.name,
        steps,
        len(result),
        time.perf_counter() - started,
    )
    return result


def shown(facts, group, held, observer):
    started = time.perf_counter()
    stamps = facts.stamps(group)
    permitted = clock.allowed(
        len(stamps["since"]),
        facts.arrays[group + "_seen_row"],
        facts.arrays[group + "_seen_observer"],
        observer,
    )
    begun, ended = clock.visible(
        stamps["since"],
        stamps["until"],
        stamps["known"],
        stamps["replaced"],
        permitted,
        held["moments"],
        held["looking"],
    )
    counts = ended - begun
    row = np.repeat(np.arange(len(counts)), counts)
    step = np.repeat(begun, counts) + spans.counted(counts)
    LOG.debug(
        "shown exit graph=%s group=%s facts=%d rows=%d seconds=%.4f",
        facts.name,
        group,
        len(counts),
        len(row),
        time.perf_counter() - started,
    )
    return row, step, stamps["since"][row]


def view(facts, registry, held, several):
    started = time.perf_counter()
    schema, observer, steps = registry.schema, held["observer"], len(held["moments"])
    if steps < 1:
        raise ValueError("the view of graph %s shows no step" % facts.name)
    choose, place = SEVERAL.named(several), ranks(facts.keys)
    row, step, since = shown(facts, "obs", held, observer)
    entity = np.asarray(facts.obs_entity, np.int64)[row]
    keys = clock.keys(
        schema.ties,
        dict(
            seq=np.asarray(facts.obs_seq)[row],
            code=np.asarray(facts.kinds)[entity],
            source=place[entity],
            target=place[entity],
        ),
    )
    order = np.lexsort(tuple(keys[::-1]) + (since, entity, step))
    first = np.r_[True, (step[order][1:] != step[order][:-1]) | (entity[order][1:] != entity[order][:-1])]
    starts = np.flatnonzero(first[: len(order)])
    chosen = order[choose(starts, np.r_[starts[1:], len(order)][: len(starts)])]
    observed = spans.observation_spans(
        entity[chosen].astype(np.int32),
        step[chosen].astype(np.int32),
        np.asarray(facts.obs_values)[row[chosen]],
    )
    observed["obs_values"] = observed.pop("obs_x")
    row, step, since = shown(facts, "fact", held, observer)
    source, target = np.asarray(facts.fact_source)[row], np.asarray(facts.fact_target)[row]
    code, present = np.asarray(facts.fact_code, np.int64)[row], np.asarray(facts.fact_present)[row]
    keys = clock.keys(
        schema.ties,
        dict(seq=np.asarray(facts.fact_seq)[row], code=code, source=place[source], target=place[target]),
    )
    order = np.lexsort((since,) + tuple(keys[::-1]) + (~present, step))
    queries, labels = held["queries"], held["labels"]
    grouped = schema.queries is not None and schema.queries["grouping"] is not None
    graph = Graph(
        facts.name if observer < 0 else "%s seen by observer %d" % (facts.name, observer),
        facts.keys,
        steps,
        kinds=np.asarray(facts.kinds),
        entity_token=np.asarray(facts.entity_token),
        period=np.asarray(facts.query_grouping)[queries].astype(np.int32)
        if grouped
        else np.arange(steps, dtype=np.int32),
        obs_x=encoded(registry, observed["obs_values"]),
        **observed,
        **spans.fact_spans(
            step[order].astype(np.int32),
            source[order],
            target[order],
            code[order],
            present[order],
            np.asarray(facts.fact_values)[row[order]],
        ),
        label_entity=np.asarray(facts.label_entity)[labels],
        label_step=held["label_step"],
        label_row=np.asarray(facts.label_row)[labels],
        label_id=np.asarray(facts.label_id)[labels],
        query_key=np.asarray(facts.query_key)[queries],
        query_category=np.asarray(facts.query_category)[queries],
        query_references=np.asarray(facts.query_references)[queries],
        query_values=np.asarray(facts.query_values)[queries],
        query_x=np.asarray(facts.query_x)[queries],
        query_signals=np.asarray(facts.query_signals)[queries],
        graph_values=np.asarray(facts.graph_values),
    )
    verify(graph, registry)
    LOG.info(
        "view exit graph=%s observer=%d steps=%d observations=%d facts=%d seconds=%.3f",
        facts.name,
        observer,
        steps,
        len(chosen),
        len(order),
        time.perf_counter() - started,
    )
    return graph


def validate(facts, registry):
    started = time.perf_counter()
    names = offered(registry)
    if not names:
        raise ValueError("no choice of steps serves the graphs of this bank")
    result = [
        view(facts, registry, held, SEVERAL.names()[0]) for held in sequences(facts, registry, names[0])
    ]
    LOG.debug(
        "validate exit graph=%s views=%d seconds=%.4f", facts.name, len(result), time.perf_counter() - started
    )
    return result
