import functools
import heapq
import logging
import pickle
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

import numba
import numpy as np

from graphspec import registries
from graphspec.infra.files import Arrays
from graphspec.infra.processes.processes import guarded, spawned, worker_threads
from graphspec.infra.timing import measure
from graphspec.operations import pipeline
from graphspec.operations.state import EMPTY
from graphspec.representations import base as representations
from graphspec.representations.registry import chosen as representing
from graphspec.views.graph import Dense, Graph, verify_dense

from . import transform
from .segments import Plan, features, opened, paired, unified
from .stitch import BUFFERED, Appender

LOG = logging.getLogger(__name__)
PIPE = {}
RANK = dict(stitch=0, expand=1)


class Stage:
    def __init__(self, name, step, phase, part, carried, source, gather, last):
        self.name, self.step, self.phase, self.part, self.carried = name, step, phase, part, carried
        self.source, self.gather, self.last = source, gather, last
        self.inputs = (source,) + (gather or ())


def staging(steps, registry):
    started = time.perf_counter()
    stages, current, pending = [Stage("open", None, None, None, True, "expand", None, False)], "open", None
    for place, step in enumerate(steps):
        phases = step.operation.phases
        for index, phase in enumerate(phases):
            last, parts = index == len(phases) - 1, step.parts(index, registry)
            carried = phase.carries is not None
            if parts is not None:
                names = tuple("%d.%d.%d" % (place, index, number) for number in range(len(parts)))
                stages += [
                    Stage(name, place, index, part, carried, current, None, False)
                    for name, part in zip(names, parts)
                ]
                pending = names
                continue
            name = "%d.%d" % (place, index)
            stages.append(Stage(name, place, index, None, carried, current, pending, last))
            current, pending = name, None
    stages.append(Stage("represent", None, None, None, True, current, None, False))
    LOG.debug("staging exit stages=%d seconds=%.4f", len(stages), time.perf_counter() - started)
    return stages


def attach(jobs, registry, stated, spec, groups, seed, limits, partitions, threads):
    started = time.perf_counter()
    numba.set_num_threads(threads)
    steps = pipeline.resolved(registry, stated)
    PIPE.clear()
    PIPE.update(
        jobs=[
            dict(plan=plan, paths=paths, spill=Path(spill), destination=destination, states={})
            for plan, paths, spill, destination in jobs
        ],
        registry=registry,
        steps=steps,
        stages={stage.name: stage for stage in staging(steps, registry)},
        initial=transform.initial(registry, steps),
        spec=spec,
        chosen=representing(groups),
        seed=seed,
        limits=limits,
        partitions=partitions,
    )
    LOG.debug("attach exit jobs=%d seconds=%.4f", len(jobs), time.perf_counter() - started)


def staged(function):
    @functools.wraps(function)
    def run(job, *arguments):
        started = time.perf_counter()
        result = function(job, *arguments) or {}
        seconds = time.perf_counter() - started
        LOG.info("%s exit job=%d index=%s seconds=%.3f", function.__name__, job, arguments[0], seconds)
        return dict(result, seconds=seconds)

    return run


def first(stage):
    if stage.name == "open":
        return EMPTY
    if stage.name == "represent":
        return representations.initial(PIPE["registry"], PIPE["spec"], PIPE["chosen"])
    step = PIPE["steps"][stage.step]
    held = PIPE["initial"][step.name][step.operation.phases[stage.phase].name]
    return held if stage.part is None else held[stage.part]


def state_of(job, stage, index):
    if index == 0:
        return first(stage)
    version, state = PIPE["jobs"][job]["states"].get(stage.name, (0, None))
    if version != index:
        version, state = peek(job, 0, "state-" + stage.name)
    if version != index:
        raise RuntimeError("segment stage %s received state %d for segment %d" % (stage.name, version, index))
    return state


def keep(job, stage, index, state, floating=True):
    PIPE["jobs"][job]["states"][stage.name] = (index + 1, state)
    if floating:
        park(job, 0, "state-" + stage.name, (index + 1, state))


def folder(job, index, stage):
    return PIPE["jobs"][job]["spill"] / ("%s_%05d.pickle" % (stage, index))


def park(job, index, stage, value):
    with open(folder(job, index, stage), "wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def peek(job, index, stage):
    with open(folder(job, index, stage), "rb") as stream:
        return pickle.load(stream)


def taken(job, index, stage):
    value = peek(job, index, stage)
    folder(job, index, stage).unlink()
    return value


def packed(graph, carry, held):
    return dict(graph=(graph.name, graph.steps, graph.arrays), carry=carry, held=held)


def unpacked(job, value):
    name, steps, arrays = value["graph"]
    carry = value["carry"]
    carry.entities = PIPE["jobs"][job]["plan"].entities
    return Dense(name, [], steps, **arrays), carry, value["held"]


@staged
def expand_task(job, index):
    member = Graph.load(PIPE["jobs"][job]["paths"][index], mapped=True).dense()
    park(job, index, "expand", (member.name, member.steps, member.arrays))


@staged
def open_task(job, index):
    plan, stage = PIPE["jobs"][job]["plan"], PIPE["stages"]["open"]
    name, steps, arrays = taken(job, index, "expand")
    graph = Dense(name, [], steps, **arrays)
    carry, dense, seen, _, _ = opened(
        plan, index, graph, 0, steps, None, plan.entities, (0, 0), state_of(job, stage, index)
    )
    park(job, index, "open", packed(dense, carry, None))
    keep(job, stage, index, seen)


@staged
def phase_task(job, index, name):
    stage, registry, limits = PIPE["stages"][name], PIPE["registry"], PIPE["limits"]
    step = PIPE["steps"][stage.step]
    if stage.part is not None:
        graph, carry, held = unpacked(job, peek(job, index, stage.source))
    else:
        graph, carry, held = unpacked(job, taken(job, index, stage.source))
    if stage.gather is not None:
        held = {PIPE["stages"][part].part: taken(job, index, part) for part in stage.gather}
    state = state_of(job, stage, index) if stage.carried else None
    glob = carry.glob
    result, state = step.phase(
        stage.phase, graph, registry, PIPE["seed"], limits, carry, held, state, stage.part, PIPE["partitions"]
    )
    if stage.part is not None:
        if carry.glob is not glob:
            raise ValueError("a phase of the operation %s run in parts widened its segment" % step.operation.name)
        park(job, index, name, result)
    elif stage.last:
        transform.checked(step, graph, result, registry)
        if stage.step == len(PIPE["steps"]) - 1:
            with measure("verify"):
                verify_dense(result, registry, carry.known)
        park(job, index, name, packed(result, carry, None))
    else:
        park(job, index, name, packed(graph, carry, result))
    if stage.carried:
        keep(job, stage, index, state)


@staged
def represent_task(job, index):
    work, stage = PIPE["jobs"][job], PIPE["stages"]["represent"]
    graph, carry, _ = unpacked(job, taken(job, index, stage.source))
    if not PIPE["steps"]:
        with measure("verify"):
            verify_dense(graph, PIPE["registry"], carry.known)
    piece, state = representations.represented(
        carry,
        graph,
        PIPE["registry"],
        PIPE["spec"],
        PIPE["chosen"],
        PIPE["seed"],
        PIPE["limits"],
        state_of(job, stage, index),
    )
    keep(job, stage, index, state, False)
    if "appender" not in work:
        work["appender"] = Appender(work["destination"], BUFFERED)
    work["appender"].add(piece)
    if index == len(work["paths"]) - 1:
        appender = work.pop("appender")
        appender.add(representations.ended(PIPE["chosen"], state))
        appender.close()


@staged
def stitch_task(job, index):
    work = PIPE["jobs"][job]
    plan, registry, spec, chosen = work["plan"], PIPE["registry"], PIPE["spec"], PIPE["chosen"]
    tables = [held.astype(np.float16, copy=False) for held in plan.second_tables()[0]] + [
        np.asarray(features(Graph.load(path, mapped=True))).astype(np.float16, copy=False) for path in work["paths"]
    ]
    arrays = representations.stitched(
        Arrays(work["destination"]),
        np.concatenate(tables),
        plan.entities.kinds,
        plan.entities.tokens,
        registry,
        spec,
        chosen,
        PIPE["limits"],
        int(plan.steps[-1]),
        work["destination"],
    )
    representations.validate(arrays, plan.name, registry, spec, chosen)


TASKS = dict(expand=expand_task, open=open_task, represent=represent_task, stitch=stitch_task)


def variants(jobs, registry, steps, spec, groups, seed, workers, limits, partitions):
    started = time.perf_counter()
    plans = [
        Plan(job["members"], paired(steps, job["beside"]), unified(job["members"], registry, spec, seed), 0)
        for job in jobs
    ]
    stages = staging(steps, registry)
    named = {stage.name: stage for stage in stages}
    sizes = [len(job["members"]) for job in jobs]
    initargs = (
        [
            (plan, [str(path) for path in job["paths"]], str(job["spill"]), str(job["destination"]))
            for plan, job in zip(plans, jobs)
        ],
        registry,
        [step.entry() for step in steps],
        spec,
        groups,
        seed,
        limits,
        partitions,
        max(1, numba.config.NUMBA_NUM_THREADS // workers),
    )
    hosts, dedicated = max(1, workers // 8), workers > 2
    order = {stage.name: place for place, stage in enumerate(stages)}
    ordered = {(job, stage.name): 0 for job in range(len(jobs)) for stage in stages}
    last = {}
    done = {(job, name): set() for job in range(len(jobs)) for name in tuple(RANK) + tuple(named)}
    expanded, active = [0] * len(jobs), {}
    waiting, running, pending, busy, stitched = [], {}, {}, {}, 0
    lookahead = 2 * workers

    def ready(job, name):
        stage, index = named[name], ordered[job, name]
        return (
            index < sizes[job]
            and not (stage.carried and active.get((job, name)))
            and all(index in done[job, held] for held in stage.inputs)
        )

    def chosen(worker):
        reserved = dedicated and worker < hosts
        found = [
            (ordered[key], -order[key[1]], last.get(key) != worker, key)
            for key in ordered
            if ready(*key) and (key[0] % hosts == worker if key[1] == "represent" else not reserved)
        ]
        if found:
            job, name = min(found)[3]
            index = ordered[job, name]
            ordered[job, name] += 1
            active[job, name] = active.get((job, name), 0) + 1
            last[job, name] = worker
            return job, name, (index,) if name in TASKS else (index, name)
        if waiting and not reserved:
            _, _, job, name, arguments = heapq.heappop(waiting)
            return job, name, arguments
        return None

    executors = [
        ProcessPoolExecutor(1, **spawned(attach, initargs, registries.carried, registries.restored))
        for _ in range(workers)
    ]
    try:
        with worker_threads(initargs[-1]):
            while stitched < len(jobs):
                for job in range(len(jobs)):
                    while expanded[job] < sizes[job] and expanded[job] - len(done[job, "represent"]) < lookahead:
                        heapq.heappush(waiting, (expanded[job], RANK["expand"], job, "expand", (expanded[job],)))
                        expanded[job] += 1
                for worker in range(workers):
                    if worker not in pending:
                        task = chosen(worker)
                        if task is not None:
                            job, name, arguments = task
                            function = TASKS[name] if name in TASKS else phase_task
                            running[executors[worker].submit(guarded, function, job, *arguments)] = worker
                            pending[worker] = task
                if not running:
                    raise RuntimeError("parallel segment build stalled")
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    job, name, arguments = pending.pop(running.pop(future))
                    result = future.result()
                    busy[name] = busy.get(name, 0.0) + result["seconds"]
                    if name == "stitch":
                        stitched += 1
                        continue
                    if (job, name) in active:
                        active[job, name] -= 1
                    index = arguments[0]
                    done[job, name].add(index)
                    if name == "represent" and index == sizes[job] - 1:
                        LOG.info("segments represented job=%d seconds=%.3f", job, time.perf_counter() - started)
                        heapq.heappush(waiting, (-1, RANK["stitch"], job, "stitch", (index,)))
    finally:
        with ThreadPoolExecutor(workers) as closing:
            list(closing.map(lambda executor: executor.shutdown(cancel_futures=True), executors))
    LOG.info(
        "segments parallel build exit seconds=%.3f groups=%d segments=%d workers=%d stages=%d busy=%s",
        time.perf_counter() - started,
        len(jobs),
        sum(sizes),
        workers,
        len(stages),
        ",".join("%s:%.1f" % item for item in sorted(busy.items())),
    )
