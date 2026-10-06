import logging
import time
from pathlib import Path

from graphspec.budgets import Refused
from graphspec.descriptions import describe as truth
from graphspec.descriptions import effect
from graphspec.infra.files import read_json, sha256_json
from graphspec.infra.timing import timed
from tuning.bank.bank import Bank
from tuning.families.heads import fielded
from tuning.statement import state

from . import reach as reaching
from .restated import STOPS, lined

LOG = logging.getLogger(__name__)
FILES = ("config.json", "statement.json")
PLANNED = ("bank_stored_sha256", "trials")
STOPPED = "the derivation of its variant stopped with %s: %s"
SHOWN = 10
READING = (
    "variant: the first %d characters of the key of the variant of the trial, trials of one key and one realisation "
    "being described once; realisation: the realisation of options the model of the trial states; a fit, in the "
    "table of several trials: refused where the check of the routes from the options to the prediction, which a fit "
    "of the trial runs before it starts, refuses its configuration, for the reason in the last column, and what the "
    "row says of its options is then what such a fit would have been given had it not been refused; per kind of task "
    "that decides, options held/stated in decisions described/shown: the options the batches of a fit of the trial "
    "hold, by the rule that packs them, over the options stated by the decisions described, those of which an option "
    "holds a point, over the decisions the graphs of the view show; an option a fit leaves out is put before no "
    "model, and after left out come the entries of the pipeline after which a fit stops holding such options, each "
    "with its options, no entry where a fit holds them in no state, the view before the first entry included; then "
    "per field of the content of its options, put/held/stated: the options to which the field is put before the "
    "model, of those a fit holds, of those stating the field; beside the variant every field of an option a fit holds "
    "is put before the model, a kind, a parameter and an element an action creates with the option itself and a node "
    "or an edge as the target the variant holds; under an edit a field is put before the model only as the facts the "
    "edit makes of it, where the option's own fork holds one, and after not put come the entries after which it stops "
    "being put before the model, each with its options; for a role, joined: the targets the graph the family reads, "
    "the variant or under an edit the option's own fork, joins to a reference of their task at its step, over the "
    "targets of the role in the options a fit holds, with the fewest and the most hops, which a model does not need "
    "to read a target beside the variant; and after no path the entries after which the targets that graph holds and "
    "joins to no reference stop being joined, each with its targets; unread: a field of a sort the realisation does "
    "not read; stated by none: a field no option states; nothing left at the steps of tasks: the entries of the "
    "pipeline that leave no node of a kind or no edge of a relation at the steps of the tasks of a kind of task where "
    "one was held before them, each with the graphs of the view in which it does over the graphs of the view, and, "
    "as doing so alone, each later entry that leaves none where the pipeline is derived without the entries named "
    "before it for the same, or with one parameter of such an entry re-stated where the pipeline is refused without "
    "it; not "
    "described: in place of the columns of a trial, the file the trial lacks, what the checks of its statement and of "
    "its configuration refuse, with no variant and no realisation, or the error of the library's own that stopped the "
    "derivation of its variant, after the name of its class; any other error stops the table and is no reason of a "
    "trial; the settings under which an option, a field or a target comes back are in the reach of one trial, "
    "which the verb describe --reach prints for the bank and the config.json and statement.json of the trial" % SHOWN
)


def severed(reached, group, cuts):
    started = time.perf_counter()
    told = reached.effects(group)
    for graph in told:
        for held in graph["effects"]:
            for found in held["cuts"]:
                without = tuple((lifted["entry"], lifted["how"]) for lifted in found["without"])
                cut = (held["entry"], held["operation"], found["sort"], found["name"], found["task"], without)
                cuts[cut] = cuts.get(cut, 0) + 1
    LOG.debug("reaches severed exit group=%d graphs=%d seconds=%.4f", group, len(told), time.perf_counter() - started)
    return len(told)


@timed("reaches.stopping")
def stopping(reached, decision, option, site, found):
    place = reached.stopped(None, decision, option, site)
    found[place] = found.get(place, 0) + 1


def tallied(reached, decision, task):
    started = time.perf_counter()
    options, last = reached.optioned(decision), len(reached.steps)
    task["described"] += 1
    task["options"] += len(options)
    for option in options:
        held = reached.read(None, last, decision, option)
        task["held"] += held["held"]
        if not held["held"]:
            stopping(reached, decision, option, (reaching.HELD,), task["left"])
        for name, field in held["fields"].items():
            cell = task["fields"][name]
            cell["stated"] += 1
            if field is None:
                continue
            cell["held"] += 1
            cell["put"] += field["put"]
            if not field["put"]:
                stopping(reached, decision, option, (reaching.PUT, name, None), cell["unput"])
            for place, target in enumerate(field["targets"]) if "targets" in field else ():
                cell["targets"] += 1
                if target["state"] == reaching.JOINED:
                    cell["hops"].append(target["hops"])
                elif target["state"] == reaching.UNJOINED:
                    stopping(reached, decision, option, (reaching.JOINED, name, place), cell["parted"])
    LOG.debug("reaches tallied exit decision=%d options=%d seconds=%.4f", decision["key"], len(options), time.perf_counter() - started)


@timed("reaches.cutting")
def cutting(found, steps, part):
    return [
        {
            "entry": None if place is None else steps[place].name,
            "operation": None if place is None else steps[place].operation.name,
            part: count,
        }
        for place, count in sorted(found.items(), key=lambda pair: len(steps) if pair[0] is None else pair[0])
    ]


@timed("reaches.counted")
def counted(cell, steps):
    aimed = cell["role"]
    return dict(
        stated=cell["stated"],
        held=cell["held"],
        put=cell["put"],
        unput=cutting(cell["unput"], steps, "options"),
        targets=cell["targets"] if aimed else None,
        joined=len(cell["hops"]) if aimed else None,
        least=min(cell["hops"]) if cell["hops"] else None,
        most=max(cell["hops"]) if cell["hops"] else None,
        parted=cutting(cell["parted"], steps, "targets") if aimed else None,
    )


def summed(deriver, config, name):
    started = time.perf_counter()
    reached = reaching.Reached(deriver, config, name)
    steps, schema = reached.steps, deriver.registry.schema
    tasks = {
        task: dict(
            decisions=0,
            described=0,
            options=0,
            held=0,
            left={},
            fields={
                field["name"]: dict(role=field["sort"] == "targets", stated=0, held=0, put=0, unput={}, targets=0, hops=[], parted={})
                if field["sort"] in reached.entry.reads
                else None
                for field in fielded(schema, task)
            },
        )
        for task in schema.decided()
    }
    cuts, graphs = {}, 0
    for group in deriver.graphs(config):
        graphs += severed(reached, group, cuts)
        trunk = reached.derived(None, len(steps), group)
        for shown in reached.counted(trunk, group).values():
            for task, count in shown.items():
                tasks[task]["decisions"] += count
        for decision in reached.decided(trunk, group):
            tallied(reached, decision, tasks[reached.registry.tasks[decision["kind"]]])
            reached.decision_done()
        reached.group_done()
    places = {step.name: place for place, step in enumerate(steps)}
    result = dict(
        graphs=graphs,
        tasks={
            task: dict(
                held,
                left=cutting(held["left"], steps, "options"),
                fields={field: None if cell is None else counted(cell, steps) for field, cell in held["fields"].items()},
            )
            for task, held in tasks.items()
        },
        cuts=[
            dict(
                entry=entry,
                operation=operation,
                sort=sort,
                name=held,
                task=task,
                without=[dict(entry=lifted, how=how) for lifted, how in without],
                graphs=count,
            )
            for (entry, operation, sort, held, task, without), count in sorted(
                cuts.items(), key=lambda pair: (places[pair[0][0]], pair[0][2:])
            )
        ],
    )
    LOG.info("reaches summed exit realisation=%s graphs=%d seconds=%.3f", name, graphs, time.perf_counter() - started)
    return result


def rowed(deriver, config, key, name, fields):
    started = time.perf_counter()
    result = dict(
        version=truth.VERSION,
        variant=key,
        stored_sha256=deriver.source,
        realisation=name,
        fields=fields,
        **summed(deriver, config, name),
    )
    LOG.info("reaches rowed exit variant=%s seconds=%.3f", key[:SHOWN], time.perf_counter() - started)
    return result


def recorded(bank, manifest, fields, stated):
    started = time.perf_counter()
    rows = []
    for trial, config, found, reason in stated:
        row = dict(trial=trial, variant=None, realisation=None, refused=None, reason=reason, graphs=None, tasks=None, cuts=None)
        if found is not None:
            deriver, name, key, refused = found
            row.update(variant=key, realisation=name, refused=refused)
            held = Path(bank) / "descriptions" / "reach_rows" / reaching.stemmed(key, name, None)
            try:
                kept = truth.kept(held, lambda: rowed(deriver, config, key, name, fields), row_text)
                row.update({part: kept[part] for part in ("graphs", "tasks", "cuts")})
            except STOPS as error:
                row.update(reason=STOPPED % (type(error).__name__, lined(error)))
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
                found = reaching.named(bank, derivers, config, statement)
            except Refused as error:
                reason = lined(error)
        stated.append((trial, config, found, reason))
    key = sha256_json(
        dict(
            stored=manifest["stored_sha256"],
            version=truth.VERSION,
            trials=[
                dict(
                    trial=trial,
                    row=None if found is None else reaching.stemmed(found[2], found[1], None),
                    refused=None if found is None else found[3],
                    reason=reason,
                )
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
        raise ValueError("the plan.json of %s lacks %s" % (folder, " and ".join(lacking)))
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
        for head in ["%s: options held/stated in decisions described/shown" % task]
        + ["%s: %s, put/held/stated" % (task, name) for name in names]
    ]
    return ["variant", "realisation"] + held + ["nothing left at the steps of tasks"]


@timed("reaches.cut_by")
def cut_by(found, part):
    return "cut by " + " and ".join("%s %d" % ("no entry" if held["entry"] is None else held["entry"], held[part]) for held in found)


@timed("reaches.task_cell")
def task_cell(held):
    result = "%d/%d in %d/%d" % (held["held"], held["options"], held["described"], held["decisions"])
    return result + (", left out %s" % cut_by(held["left"], "options") if held["left"] else "")


@timed("reaches.field_cell")
def field_cell(cell):
    if cell is None:
        return "unread"
    if not cell["stated"]:
        return "stated by none"
    parts = ["%d/%d/%d" % (cell["put"], cell["held"], cell["stated"])]
    if cell["unput"]:
        parts.append("not put %s" % cut_by(cell["unput"], "options"))
    if cell["targets"]:
        parts.append("joined %d/%d" % (cell["joined"], cell["targets"]))
    if cell["least"] is not None:
        parts.append("hops %d" % cell["least"] if cell["least"] == cell["most"] else "hops %d to %d" % (cell["least"], cell["most"]))
    if cell["parted"]:
        parts.append("no path %s" % cut_by(cell["parted"], "targets"))
    return ", ".join(parts)


@timed("reaches.cut_cell")
def cut_cell(cuts, graphs):
    entries = {}
    for held in cuts:
        alone = " alone %s" % effect.lifted(held["without"]) if held["without"] else ""
        entries.setdefault(held["entry"], []).append(
            "%s %s at %s %d/%d%s" % (held["sort"], held["name"], held["task"], held["graphs"], graphs, alone)
        )
    return "; ".join("%s: %s" % (entry, ", ".join(found)) for entry, found in entries.items()) or "none"


@timed("reaches.fitted")
def fitted(row, fields):
    if row["reason"] is not None:
        named = ["-" if row[part] is None else row[part] for part in ("variant", "realisation")]
        return [named[0][:SHOWN], named[1], "not described: " + row["reason"]]
    held = cells(row, fields)
    if row["refused"] is None:
        return held[:2] + ["not refused"] + held[2:]
    return held[:2] + ["refused"] + held[2:] + [row["refused"]]


def cells(row, fields):
    started = time.perf_counter()
    result = [row["variant"][:SHOWN], row["realisation"]]
    for task, names in fields.items():
        held = row["tasks"][task]
        result.append(task_cell(held))
        result += [field_cell(held["fields"][name]) for name in names]
    result.append(cut_cell(row["cuts"], row["graphs"]))
    LOG.debug("reaches cells exit cells=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("reaches.tabled_lines")
def tabled_lines(header, rows):
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
    lines += tabled_lines(heads(record["fields"]), [cells(record, record["fields"])])
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
    header = heads(fields)
    header = ["trial"] + header[:2] + ["a fit"] + header[2:] + ["a fit is refused because"]
    lines += tabled_lines(header, [[str(row["trial"])] + fitted(row, fields) for row in rows])
    LOG.debug("reaches text exit lines=%d seconds=%.4f", len(lines), time.perf_counter() - started)
    return "\n".join(lines) + "\n"
