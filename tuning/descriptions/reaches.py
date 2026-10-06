import logging
import time
from pathlib import Path

from graphspec.descriptions import describe as truth
from graphspec.descriptions import effect
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json, sha256_json
from graphspec.infra.timing import timed
from graphspec.operations import parameters as specs
from tuning.bank.bank import Bank, Deriver
from tuning.families.heads import fielded
from tuning.statement import agreed, state

from . import reach as reaching

LOG = logging.getLogger(__name__)
REFUSALS = (KeyError, TypeError, ValueError)
STATED = ("model", "pipeline", "graph")
FILES = ("config.json", "statement.json")
PLANNED = ("bank_stored_sha256", "trials")
SHOWN = 10
READING = (
    "variant: the first %d characters of the key of the variant of the trial, trials of one key and one realisation "
    "being described once; realisation: the realisation of options the model of the trial states; per kind of task "
    "that decides: its options in its decisions holding a point, then per field of the content of its options, the "
    "options whose field is connected to a reference of their task at its step over the options holding the field, "
    "the fewest and the most hops of those connected, those in which it is apart, held and joined to no reference, "
    "those in which it is absent, not held at the step, and the entries of the pipeline after which the field of "
    "those not connected stops being connected, each with its options, no entry where it is connected in no state, "
    "the view before the first entry included; unread: a field of a sort the realisation does not read; held by "
    "none: a field no option holds; nothing left at the steps of tasks: the entries of the pipeline that leave no "
    "node of a kind or no edge of a relation at the steps of the tasks of a kind of task where one was held before "
    "them, each with the graphs of the view in which it does over the graphs of the view; not described: in place of "
    "the columns of a trial, what its statement refuses of its configuration, what the trial lacks, or what stopped "
    "the derivation of its variant, with no variant and no realisation where its statement names none; the settings "
    "under which a field comes back are in the reach of one trial, which the verb describe --reach prints for the "
    "bank and the config.json and statement.json of the trial" % SHOWN
)


def named(bank, derivers, config, statement):
    started = time.perf_counter()
    stated = state(statement)
    agreed(config, stated)
    lacking = [name for name in STATED if name not in config]
    if lacking:
        raise KeyError("the configuration states no " + " and no ".join(lacking))
    held = (stated.device, stated.derivation_seed)
    if held not in derivers:
        derivers[held] = Deriver(bank, *held, RESIDENT)
    deriver = derivers[held]
    name = reaching.realising(deriver, config)
    key = specs.digest(deriver.identity(config))
    LOG.debug("reaches named exit variant=%s realisation=%s seconds=%.4f", key[:SHOWN], name, time.perf_counter() - started)
    return deriver, name, key


def severed(reached, group, states, cuts):
    started = time.perf_counter()
    told = effect.told(states, reached.loaded(group)[1], reached.registry, reached.steps)
    for graph in told:
        for held in graph["effects"]:
            for found in held["cuts"]:
                cut = (held["entry"], held["operation"], found["sort"], found["name"], found["task"])
                cuts[cut] = cuts.get(cut, 0) + 1
    LOG.debug("reaches severed exit group=%d graphs=%d seconds=%.4f", group, len(told), time.perf_counter() - started)
    return len(told)


def tallied(reached, decision, task):
    started = time.perf_counter()
    options, last = reached.optioned(decision), len(reached.steps)
    task["decisions"] += 1
    task["options"] += len(options)
    for option in options:
        for name, held in reached.read(None, last, decision, option).items():
            cell = task["fields"][name]
            cell["held"] += 1
            cell[held["state"]] += 1
            if held["state"] == reaching.CONNECTED:
                cell["hops"].append(held["hops"])
            else:
                place = reached.stopped(decision, option, name)
                cell["cut"][place] = cell["cut"].get(place, 0) + 1
    LOG.debug("reaches tallied exit decision=%d options=%d seconds=%.4f", decision["key"], len(options), time.perf_counter() - started)


@timed("reaches.counted")
def counted(cell, steps):
    cut = sorted(cell["cut"].items(), key=lambda pair: len(steps) if pair[0] is None else pair[0])
    return dict(
        held=cell["held"],
        connected=cell[reaching.CONNECTED],
        apart=cell[reaching.APART],
        absent=cell[reaching.ABSENT],
        least=min(cell["hops"]) if cell["hops"] else None,
        most=max(cell["hops"]) if cell["hops"] else None,
        cut=[
            dict(
                entry=None if place is None else steps[place].name,
                operation=None if place is None else steps[place].operation.name,
                options=options,
            )
            for place, options in cut
        ],
    )


def summed(deriver, config, name):
    started = time.perf_counter()
    reached = reaching.Reached(deriver, config, name)
    steps, schema = reached.steps, deriver.registry.schema
    tasks = {
        task: dict(
            decisions=0,
            options=0,
            fields={
                field["name"]: {"held": 0, reaching.CONNECTED: 0, reaching.APART: 0, reaching.ABSENT: 0, "hops": [], "cut": {}}
                if field["sort"] in reached.entry.reads
                else None
                for field in fielded(schema, task)
            },
        )
        for task in schema.decided()
    }
    cuts, graphs = {}, 0
    for group in deriver.graphs(config):
        states = [reached.derived(None, count, group) for count in range(len(steps) + 1)]
        graphs += severed(reached, group, states, cuts)
        for decision in reached.decided(states[-1], group):
            tallied(reached, decision, tasks[reached.registry.tasks[decision["kind"]]])
            reached.decision_done()
        reached.group_done()
    places = {step.name: place for place, step in enumerate(steps)}
    result = dict(
        graphs=graphs,
        tasks={
            task: dict(
                held,
                fields={field: None if cell is None else counted(cell, steps) for field, cell in held["fields"].items()},
            )
            for task, held in tasks.items()
        },
        cuts=[
            dict(entry=entry, operation=operation, sort=sort, name=held, task=task, graphs=count)
            for (entry, operation, sort, held, task), count in sorted(
                cuts.items(), key=lambda pair: (places[pair[0][0]], pair[0][2:])
            )
        ],
    )
    LOG.info("reaches summed exit realisation=%s graphs=%d seconds=%.3f", name, graphs, time.perf_counter() - started)
    return result


def rowed(deriver, config, key, name, fields):
    started = time.perf_counter()
    try:
        held, reason = summed(deriver, config, name), None
    except ValueError as error:
        held, reason = dict(graphs=None, tasks=None, cuts=None), str(error)
    result = dict(
        version=truth.VERSION,
        variant=key,
        stored_sha256=deriver.source,
        realisation=name,
        fields=fields,
        reason=reason,
        **held,
    )
    LOG.info("reaches rowed exit variant=%s described=%s seconds=%.3f", key[:SHOWN], reason is None, time.perf_counter() - started)
    return result


def recorded(bank, manifest, fields, stated):
    started = time.perf_counter()
    rows = []
    for trial, config, found, reason in stated:
        row = dict(trial=trial, variant=None, realisation=None, reason=reason, graphs=None, tasks=None, cuts=None)
        if found is not None:
            deriver, name, key = found
            held = Path(bank) / "descriptions" / "reach_rows" / reaching.stemmed(key, name)
            kept = truth.kept(held, lambda: rowed(deriver, config, key, name, fields), row_text)
            row.update({part: kept[part] for part in row if part != "trial"})
        rows.append(row)
    result = dict(version=truth.VERSION, stored_sha256=manifest["stored_sha256"], fields=fields, trials=rows)
    LOG.info("reaches recorded exit trials=%d seconds=%.3f", len(rows), time.perf_counter() - started)
    return result


def tabled(bank, manifest, trials, folder):
    started = time.perf_counter()
    schema = Bank(bank).registry.schema
    if not schema.decided():
        raise ValueError("the bank %s holds no kind of task that decides; reach describes the options of decisions" % bank)
    derivers, stated = {}, []
    for trial, config, statement, reason in trials:
        found = None
        if reason is None:
            try:
                found = named(bank, derivers, config, statement)
            except REFUSALS as error:
                reason = str(error.args[0])
        stated.append((trial, config, found, reason))
    key = sha256_json(
        dict(
            stored=manifest["stored_sha256"],
            version=truth.VERSION,
            trials=[
                dict(trial=trial, row=None if found is None else reaching.stemmed(found[2], found[1]), reason=reason)
                for trial, _, found, reason in stated
            ],
        )
    )
    fields = {task: [field["name"] for field in fielded(schema, task)] for task in schema.decided()}
    held = Path(folder) / "descriptions" / "reaches" / key
    result = truth.kept(held, lambda: recorded(bank, manifest, fields, stated), text)
    LOG.info("reaches tabled exit trials=%d seconds=%.3f", len(stated), time.perf_counter() - started)
    return result


def over(bank, configs, statement):
    started = time.perf_counter()
    manifest = truth.manifested(bank)
    if type(configs) is not list:
        raise TypeError("reach over a set of variants reads a list of the configurations of fits, found %r" % (configs,))
    state(statement)
    trials = [(place, config, statement, None) for place, config in enumerate(configs)]
    result = tabled(bank, manifest, trials, bank)
    LOG.info("reaches over exit configurations=%d seconds=%.3f", len(configs), time.perf_counter() - started)
    return result


def search(bank, folder):
    started = time.perf_counter()
    manifest = truth.manifested(bank)
    found = Path(folder) / "plan.json"
    if not found.is_file():
        raise FileNotFoundError(
            "%s holds no plan.json; reach over a search reads the folder of a search, which holds its plan and its trials" % folder
        )
    plan = read_json(found)
    lacking = [name for name in PLANNED if name not in plan]
    if lacking:
        raise KeyError("the plan.json of %s lacks %s" % (folder, " and ".join(lacking)))
    if plan["bank_stored_sha256"] != manifest["stored_sha256"]:
        raise ValueError(
            "the search %s searches the bank whose build stored %s, and the bank %s stored %s"
            % (folder, plan["bank_stored_sha256"], bank, manifest["stored_sha256"])
        )
    trials = []
    for item in plan["trials"]:
        trial = Path(folder) / item["directory"]
        lacking = [name for name in FILES if not (trial / name).is_file()]
        if lacking:
            trials.append((item["directory"], None, None, "the trial holds no " + " and no ".join(lacking)))
        else:
            trials.append((item["directory"], read_json(trial / FILES[0]), read_json(trial / FILES[1]), None))
    result = tabled(bank, manifest, trials, folder)
    LOG.info("reaches search exit trials=%d seconds=%.3f", len(trials), time.perf_counter() - started)
    return result


@timed("reaches.heads")
def heads(fields):
    held = [
        head
        for task, names in fields.items()
        for head in ["%s: options in decisions" % task] + ["%s: %s" % (task, name) for name in names]
    ]
    return ["variant", "realisation"] + held + ["nothing left at the steps of tasks"]


@timed("reaches.field_cell")
def field_cell(cell):
    if cell is None:
        return "unread"
    if not cell["held"]:
        return "held by none"
    parts = ["%d/%d" % (cell["connected"], cell["held"])]
    if cell["connected"]:
        parts.append("hops %d" % cell["least"] if cell["least"] == cell["most"] else "hops %d to %d" % (cell["least"], cell["most"]))
    parts += ["%s %d" % (name, cell[name]) for name in (reaching.APART, reaching.ABSENT) if cell[name]]
    if cell["cut"]:
        parts.append(
            "cut by "
            + " and ".join("%s %d" % ("no entry" if held["entry"] is None else held["entry"], held["options"]) for held in cell["cut"])
        )
    return ", ".join(parts)


@timed("reaches.cut_cell")
def cut_cell(cuts, graphs):
    entries = {}
    for held in cuts:
        entries.setdefault(held["entry"], []).append(
            "%s %s at %s %d/%d" % (held["sort"], held["name"], held["task"], held["graphs"], graphs)
        )
    return "; ".join("%s: %s" % (entry, ", ".join(found)) for entry, found in entries.items()) or "none"


def cells(row, fields):
    started = time.perf_counter()
    result = ["-" if row["variant"] is None else row["variant"][:SHOWN], "-" if row["realisation"] is None else row["realisation"]]
    if row["reason"] is None:
        for task, names in fields.items():
            held = row["tasks"][task]
            result.append("%d in %d" % (held["options"], held["decisions"]))
            result += [field_cell(held["fields"][name]) for name in names]
        result.append(cut_cell(row["cuts"], row["graphs"]))
    else:
        result.append("not described: " + row["reason"])
    LOG.debug("reaches cells exit cells=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("reaches.lined")
def lined(header, rows):
    held = [header] + rows
    aligned = [row if len(row) == len(header) else row[:-1] for row in held]
    widths = [max(len(row[at]) for row in aligned if at < len(row)) for at in range(len(header))]
    return ["    " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in held]


def row_text(record):
    started = time.perf_counter()
    lines = [
        "reach of the options of the variant %s of the bank stored as %s in one row, described at version %d"
        % (record["variant"], record["stored_sha256"], record["version"]),
        READING,
    ]
    lines += lined(heads(record["fields"]), [cells(record, record["fields"])])
    LOG.debug("reaches row text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return "\n".join(lines) + "\n"


def text(record):
    started = time.perf_counter()
    rows, fields = record["trials"], record["fields"]
    described = [row for row in rows if row["reason"] is None]
    lines = [
        "reach over %d trials on the bank stored as %s, described at version %d"
        % (len(rows), record["stored_sha256"], record["version"]),
        "described %d, of %d variants and realisations; not described %d"
        % (len(described), len({(row["variant"], row["realisation"]) for row in described}), len(rows) - len(described)),
        "one row per trial, in the order of the trials; " + READING,
    ]
    lines += lined(["trial"] + heads(fields), [[str(row["trial"])] + cells(row, fields) for row in rows])
    LOG.debug("reaches text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return "\n".join(lines) + "\n"
