import argparse
import logging
import time
from contextlib import closing
from pathlib import Path

import numpy as np

from graphspec.infra.files import read_json, write_json
from graphspec.infra.log import configure, emit
from graphspec.infra.postgres import connect as postgres
from graphspec.infra.postgres import rows
from graphspec.staged import Staged

from .domain import SERIES, SIGNALS, TARGET

LOG = logging.getLogger(__name__)
OWNED = (
    "SELECT count(*) FROM (SELECT DISTINCT ON (state) country, present FROM state_history "
    "WHERE run_id = %(run)s AND seq <= %(seq)s ORDER BY state, seq DESC) s WHERE present AND country = %(player)s"
)
MEASURES = (
    (
        "gdp",
        "SELECT ln(gdp) FROM (SELECT DISTINCT ON (country) gdp, present FROM country_economy_report "
        "WHERE run_id = %(run)s AND country = %(player)s AND seq <= %(seq)s ORDER BY country, seq DESC) s "
        "WHERE present AND gdp > 0",
    ),
    (
        "population",
        "SELECT ln(total) FROM (SELECT lower_strata + middle_strata + upper_strata AS total FROM player_population "
        "WHERE run_id = %(run)s AND seq <= %(seq)s ORDER BY seq DESC LIMIT 1) s WHERE total > 0",
    ),
    (
        "sol",
        "SELECT average_standard_of_living_by_population FROM (SELECT DISTINCT ON (country) "
        "average_standard_of_living_by_population, present FROM country_standard_of_living_report "
        "WHERE run_id = %(run)s AND country = %(player)s AND seq <= %(seq)s ORDER BY country, seq DESC) s WHERE present",
    ),
)
PLAYER = "SELECT player_country FROM run WHERE run_id = %(run)s"


def check_graph(tables, index):
    problems = []
    queries = tables["queries"]
    steps = queries.column("step").to_numpy()
    if not np.array_equal(steps, np.arange(len(steps))):
        problems.append("graph %d query steps are not 0..n-1" % index)
    periods = queries.column("period").to_numpy()
    if np.any(np.diff(periods) < 0):
        problems.append("graph %d periods decrease" % index)
    observations = tables["observations"]
    entity = observations.column("entity").to_numpy()
    obs_step = observations.column("step").to_numpy()
    obs_seq = observations.column("seq").to_numpy()
    if len({*zip(obs_step.tolist(), obs_seq.tolist())}) != len(observations):
        problems.append("graph %d repeats (step, seq) among observations" % index)
    if len({*zip(entity.tolist(), obs_step.tolist())}) != len(observations):
        problems.append("graph %d observes an entity twice in one step" % index)
    x = np.asarray(observations.column("x").combine_chunks().flatten().to_numpy(zero_copy_only=False))
    if not np.isfinite(x).all():
        problems.append("graph %d has non-finite observation features" % index)
    observed_at = {}
    for node, step in zip(entity.tolist(), obs_step.tolist()):
        observed_at.setdefault(node, set()).add(step)
    first_seen = {node: min(steps) for node, steps in observed_at.items()}
    present_keys, absent_keys, ever = set(), set(), {}
    for relation, edges in tables["edges"].items():
        source = edges.column("source").to_numpy()
        target = edges.column("target").to_numpy()
        step = edges.column("step").to_numpy()
        seq = edges.column("seq").to_numpy()
        present = edges.column("present").to_numpy()
        for name in edges.column_names[5:]:
            values = edges.column(name).to_numpy()
            if not np.isfinite(values[present]).all():
                problems.append("graph %d %s has non-finite %s on present facts" % (index, relation, name))
        for s, t, st, sq, p in zip(
            source.tolist(), target.tolist(), step.tolist(), seq.tolist(), present.tolist()
        ):
            keys = present_keys if p else absent_keys
            if (st, sq) in keys:
                problems.append(
                    "graph %d repeats (step, seq) %r among %s facts"
                    % (index, (st, sq), "present" if p else "absent")
                )
            keys.add((st, sq))
            if p:
                if st not in observed_at.get(s, ()) or st not in observed_at.get(t, ()):
                    problems.append(
                        "graph %d %s fact at step %d lacks an endpoint observation" % (index, relation, st)
                    )
                ever.setdefault((relation, s, t), []).append(st)
            else:
                if first_seen.get(s, st + 1) > st or first_seen.get(t, st + 1) > st:
                    problems.append(
                        "graph %d %s removal at step %d precedes its endpoints" % (index, relation, st)
                    )
                if (relation, s, t) not in ever or st - 1 not in ever[(relation, s, t)]:
                    problems.append(
                        "graph %d %s removal at step %d had no fact at step %d"
                        % (index, relation, st, st - 1)
                    )
                if st in ever.get((relation, s, t), ()):
                    problems.append("graph %d %s removal and fact share step %d" % (index, relation, st))
    for role in ("actor", "subject", "context"):
        chosen = queries.column(role).to_numpy()
        for node, st in zip(chosen.tolist(), steps.tolist()):
            if st not in observed_at.get(node, ()):
                problems.append("graph %d query %s at step %d is unobserved" % (index, role, st))
    check_blocks(tables, index, problems)
    targets = tables["targets"]
    if targets.num_rows == 0:
        problems.append("graph %d has no labels" % index)
    label_steps = targets.column("step").to_numpy()
    if np.any(label_steps < 0) or np.any(label_steps >= len(steps)):
        problems.append("graph %d labels name missing queries" % index)
    y = targets.column("y").to_numpy()
    if not np.isfinite(y).all():
        problems.append("graph %d has non-finite labels" % index)
    return problems


def check_blocks(tables, index, problems):
    nodes = tables["nodes"]
    kind_of = dict(zip(nodes.column("entity").to_pylist(), nodes.column("kind").to_pylist()))
    key_of = dict(zip(nodes.column("entity").to_pylist(), nodes.column("key").to_pylist()))
    queries = tables["queries"]
    steps = queries.column("step").to_pylist()
    keys = queries.column("key").to_pylist()
    if any(later <= earlier for earlier, later in zip(keys, keys[1:])):
        problems.append("graph %d query keys do not increase with the step" % index)
    block_of = dict(zip(steps, queries.column("subject").to_pylist()))
    for st, key in zip(steps, keys):
        if key_of[block_of[st]] != "block:%d" % key:
            problems.append(
                "graph %d query at step %d has subject %s, not its block" % (index, st, key_of[block_of[st]])
            )
    for role in ("actor", "context"):
        chosen = set(queries.column(role).to_pylist())
        if len(chosen) != 1 or kind_of[chosen.pop()] != "country":
            problems.append("graph %d query %s is not one country" % (index, role))
    edges = tables["edges"]["in_block"]
    joined = set()
    for s, t, st, p in zip(
        edges.column("source").to_pylist(),
        edges.column("target").to_pylist(),
        edges.column("step").to_pylist(),
        edges.column("present").to_pylist(),
    ):
        if p and block_of[st] != t:
            problems.append("graph %d in_block fact at step %d joins another step's block" % (index, st))
        if p:
            joined.add((s, st))
    observations = tables["observations"]
    for node, st in zip(observations.column("entity").to_pylist(), observations.column("step").to_pylist()):
        if kind_of[node] == "action" and (node, st) not in joined:
            problems.append("graph %d action %s at step %d is not in its block" % (index, key_of[node], st))


def check_labels(staged, problems):
    (unit,) = staged.by_table[TARGET]
    table = unit.read()
    tau = table.column("tau").to_numpy()
    if np.any(tau <= 0) or not np.isfinite(tau).all():
        problems.append("label tau values must be positive elapsed days")
    if not {*table.column("split").to_pylist()} <= {"train", "validation", "spread"}:
        problems.append("label splits must be train, validation or spread")
    if not {*table.column("series").to_pylist()} <= set(SERIES):
        problems.append("label series must be one of " + ", ".join(SERIES))


def compare(connection, staged, reader, build, samples, seed):
    rng = np.random.default_rng(seed)
    runs = {item["index"]: item["run"] for item in build["graphs"]}
    mismatches, checked = [], 0
    for index, run in sorted(runs.items()):
        player = rows(connection, PLAYER, dict(run=run))[0][0]
        tables = reader.graph(index)
        queries = tables["queries"]
        edges = tables["edges"]["owns"]
        actor = queries.column("actor").to_numpy()
        keys = queries.column("key").to_numpy()
        signal = np.asarray(
            queries.column("signal").combine_chunks().flatten().to_numpy(zero_copy_only=False)
        ).reshape(-1, len(SIGNALS))
        observed = np.asarray(
            queries.column("signal_observed").combine_chunks().flatten().to_numpy(zero_copy_only=False)
        ).reshape(-1, len(SIGNALS))
        for step in rng.choice(len(keys), min(samples, len(keys)), replace=False).tolist():
            seq = int(keys[step])
            expected = int(rows(connection, OWNED, dict(run=run, seq=seq, player=player))[0][0])
            picked = (
                (edges.column("step").to_numpy() == step)
                & edges.column("present").to_numpy()
                & (edges.column("source").to_numpy() == actor[step])
            )
            found = int(picked.sum())
            checked += 1
            if expected != found:
                mismatches.append(
                    dict(graph=index, step=step, seq=seq, check="owns", expected=expected, found=found)
                )
            for code, (name, statement) in enumerate(MEASURES):
                found = rows(connection, statement, dict(run=run, seq=seq, player=player))
                value, seen = float(signal[step, code]), float(observed[step, code])
                checked += 1
                expected = float(found[0][0]) if found and found[0][0] is not None else None
                wrong = seen != 0.0 if expected is None else seen != 1.0 or abs(value - expected) > 1e-4
                if wrong:
                    mismatches.append(
                        dict(graph=index, step=step, seq=seq, check=name, expected=expected, found=value)
                    )
    return mismatches, checked


def verify(task_path, host, port, database, user, samples, seed):
    started = time.perf_counter()
    task = read_json(task_path)
    root = Path(task_path).parent
    build = read_json(root / (task["name"] + ".build.json"))
    staged = Staged(root / task["staging"])
    reader = staged.reader(task["target"])
    problems = []
    for index in sorted(staged.names):
        problems.extend(check_graph(reader.graph(index), index))
    check_labels(staged, problems)
    mismatches, checked = [], 0
    if database:
        with closing(postgres(host, port, database, user, 60000)) as connection:
            mismatches, checked = compare(connection, staged, reader, build, samples, seed)
    staged.close()
    result = dict(
        graphs=len(staged.names),
        problems=problems,
        checks=checked,
        mismatches=mismatches,
        seconds=time.perf_counter() - started,
    )
    write_json(root / (task["name"] + ".verify.json"), result)
    emit(
        "verify_exit",
        graphs=len(staged.names),
        problems=len(problems),
        checks=checked,
        mismatches=len(mismatches),
        seconds=result["seconds"],
    )
    if problems or mismatches:
        raise SystemExit(
            "%d contract problems and %d corpus mismatches; see %s"
            % (len(problems), len(mismatches), root / (task["name"] + ".verify.json"))
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55433)
    parser.add_argument("--database", default="vic3")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    configure()
    verify(args.task, args.host, args.port, args.database, args.user, args.samples, args.seed)


if __name__ == "__main__":
    main()
