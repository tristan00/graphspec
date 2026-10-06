import logging
import time

from graphspec import registries
from graphspec.budgets import Refused
from graphspec.infra.timing import timed
from graphspec.operations import parameters as specs
from graphspec.operations.parameters import GROUPS, among, choice, equals, flag, narrow, number, real
from graphspec.operations.pipeline import resolved
from graphspec.representations.registry import REPRESENTATIONS as REGISTERED

from .heads import DEPTH, DEPTHS, NONE, REALISATION, REALISATIONS, fielded, reading, realisation, refused

LOG = logging.getLogger(__name__)
MODEL, GRAPH = "the model of the configuration", "the graph of the configuration"
UNREAD = (
    "the realisation of options %s puts options into the graph of the family of its model, and the family %s does not "
    "read them there: %s"
)
SPACES = {}
DERIVES = "representations/%s"
VIEWS = (
    "snapshot",
    "mixed",
    "event_graph",
    "memory",
    "projection",
    "sequence",
    "tree",
    "walks",
    "affinity",
    "signal",
)
OPTIONED = VIEWS
PASSING = "it passes no message over a graph"
STRUCTURE = dict(
    snapshot=None,
    mixed=None,
    event_graph=None,
    tree=None,
    memory=(
        "its rules update both ends of every event alike, so an option put into its graph would change the state "
        "of its actor and targets, which only a realisation stating that messages pass back may do"
    ),
    projection="it propagates fixed random projections along events, and " + PASSING,
    sequence="it reads the sampled partners of a node as a sequence of tokens, and " + PASSING,
    walks="it reads sampled walks from a node, and " + PASSING,
    affinity="it reads counts of tasks per node and reads no edge",
    signal="it reads the signals of tasks and the features of members and reads no edge",
)
EDITS = ("facts", "relations", "values")
DIRECTION = "relations/direction"
NEEDS = dict(targets="facts", kind="relations", parameters="values")
EDITED = dict(
    mixed=(),
    event_graph=("facts", "relations"),
    memory=("facts", "relations"),
    projection=("facts", "relations"),
    sequence=("facts", "relations"),
    tree=("facts", "relations"),
    walks=("facts", "relations"),
    affinity=(),
    signal=(),
)
SPATIALLY = dict(
    none=(),
    mean=("facts",),
    normalised=("facts",),
    relational=EDITS,
    attention=("facts",),
    chebyshev=("facts",),
    relational_attention=("facts", "relations"),
)
PARTS = dict(facts="the facts an edit adds", relations="their relations", values="their values")
MADE = dict(targets="the end of a fact", kind="the relation of a fact", parameters="a value of a fact")
UNRELATED = "mutual_rnn"
SHOWN = dict(event_graph="events/kinds", sequence="neighbors/kinds", tree="neighbors/kinds", walks="neighbors/kinds")
RELATION = "relation"
VARIED = ("snapshot/spatial", "memory/update")
SPATIAL = ("none", "mean", "normalised", "relational", "attention", "chebyshev", "relational_attention")
ENCODINGS = ("log_cosine", "cosine", "log_linear")
EMBEDDING = ("snapshot", "mixed", "event_graph", "memory", "projection", "sequence", "tree", "walks")
TEMPORAL = (
    "graph_sequence",
    "entity_gru",
    "entity_tcn",
    "entity_attention",
    "graph_cell",
    "evolving",
    "coupling_edges",
)
MESSAGES = ("gated", "attention")
UPDATES = ("mailbox_gru", "mutual_rnn", "intensity", "antisymmetric")
OPERATORS = ("mixer", "transformer", "ssm")
STRUCTURES = ("none", "occurrence", "overlap")
REPRESENTATIONS = dict(
    lifted="coupling", dag="events", tree="neighbors", history="neighbors", walks="walks", mixed="mixed"
)
TRAINING = (
    number("epochs", 1, 12, 4),
    real("clip", 0.5, 2.0, 1.0),
    real("lr", 0.0001, 0.005, 0.001, log=True),
    real("weight_decay", 1e-7, 1e-2, 1e-5, log=True),
)


def views(registry):
    started = time.perf_counter()
    extras = registry.extras
    excluded = set()
    if not extras["signals"]:
        excluded.add("signal")
    if not extras["events"]:
        excluded.add("affinity")
    result = tuple(view for view in VIEWS if view not in excluded)
    LOG.debug("views exit offered=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


@timed("model_spec.view")
def view(*names):
    return among("view", names)


@timed("model_spec.attends")
def attends(spec):
    return spec.get("snapshot/sequence") == "attention" or spec.get("snapshot/temporal") == "entity_attention"


def memory_state(spec):
    return spec.get("view") == "memory" and spec["memory/update"] != "mutual_rnn"


@timed("model_spec.reads_time")
def reads_time(spec):
    if spec["view"] == "memory":
        return spec["memory/update"] != "antisymmetric" or spec["memory/embedding"] == "attention"
    return (
        spec["view"] in ("event_graph", "projection", "sequence", "tree", "walks")
        or spec.get("snapshot/temporal") == "coupling_edges"
    )


@timed("model_spec.embeds_keys")
def embeds_keys(spec):
    return spec["view"] in EMBEDDING


@timed("model_spec.reads_dropout")
def reads_dropout(spec):
    if spec["view"] == "snapshot":
        return (
            spec["snapshot/spatial"] != "none"
            or attends(spec)
            or spec["snapshot/temporal"] == "coupling_edges"
        )
    return spec["view"] in ("mixed", "event_graph") or spec.get("sequence/operator") == "transformer"


def space(registry):
    started = time.perf_counter()
    key = (
        registries.GENERATION,
        views(registry),
        registry.extras["events"],
        reading(registry, "time/encoding"),
        bool(registry.schema.decided()),
        OPTIONED,
    )
    if key not in SPACES:
        SPACES[key] = built(*key[1:])
    LOG.debug("model space exit parameters=%d seconds=%.4f", len(SPACES[key]), time.perf_counter() - started)
    return SPACES[key]


@timed("model_spec.optioned")
def optioned(spec, families):
    return spec["view"] in families


@timed("model_spec.realised")
def realised(spec, schema):
    if not schema.decided():
        return None
    if not isinstance(spec, dict) or "view" not in spec:
        raise Refused("%s states no view, the family of its model, one of %s" % (MODEL, ", ".join(VIEWS)))
    if not optioned(spec, OPTIONED):
        return None
    if REALISATION not in spec:
        raise Refused(
            "the model of the family %s states no %s, which a model of a family that reads options states on a bank "
            "that decides: %s or a realisation registered in tuning.REALISATIONS" % (spec["view"], REALISATION, NONE)
        )
    name = spec[REALISATION]
    if name != NONE and name not in REALISATIONS.names():
        raise Refused(
            "the model of the family %s states %s %s, and the realisations of options registered in "
            "tuning.REALISATIONS are %s" % (spec["view"], REALISATION, name, ", ".join(REALISATIONS.names()))
        )
    return None if name == NONE else name


def deep(spec):
    return REALISATION in spec and spec[REALISATION] != NONE and REALISATIONS.named(spec[REALISATION]).structure


@timed("model_spec.depth")
def depth(spec):
    if DEPTH not in spec:
        raise ValueError(
            "the model of the family %s states the realisation of options %s, which puts options into the graph of its "
            "family, and states no %s, the number of rounds of messages from the state into each option, one of %s"
            % (spec["view"], spec[REALISATION], DEPTH, ", ".join(map(str, DEPTHS)))
        )
    return spec[DEPTH]


@timed("model_spec.edited")
def edited(spec, schema):
    name = realised(spec, schema)
    return name if name is not None and realisation(name).back else None


def sending(registry, entry, held):
    started = time.perf_counter()
    name = held["relation"]
    stated = entry.relations[name] if name in entry.relations else registry.schema.edges[name]
    result = stated["directed"] and held["actor"] == "source"
    LOG.debug("model sending exit relation=%s sending=%s seconds=%.6f", name, result, time.perf_counter() - started)
    return result


def mapped(registry):
    started = time.perf_counter()
    result = set()
    for name in REALISATIONS.names():
        entry = realisation(name)
        if entry.back and refused(name, registry) is None:
            result |= {
                held["relation"]
                for actions in entry.mapping.values()
                for roles in actions.values()
                for held in roles.values()
                if held["relation"] in registry.schema.edges
            }
    LOG.debug("model mapped exit relations=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def turning(registry, spec, parameters):
    started = time.perf_counter()
    name, named, result = edited(spec, registry.schema), {parameter.name: parameter for parameter in parameters}, {}
    entry = None if name is None else realisation(name)
    sends = entry is not None and any(
        sending(registry, entry, held)
        for actions in entry.mapping.values()
        for roles in actions.values()
        for held in roles.values()
    )
    if sends and DIRECTION in named:
        result[DIRECTION] = [option for option in named[DIRECTION].options if option != "directed"]
    if entry is not None and spec["view"] in SHOWN and SHOWN[spec["view"]] in named:
        result[SHOWN[spec["view"]]] = [option for option in named[SHOWN[spec["view"]]].options if showing(option)]
    LOG.debug("model turning exit settings=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def turned(registry, spec, parameters):
    started = time.perf_counter()
    kept = turning(registry, spec, parameters)
    result = narrow(parameters, kept) if kept else parameters
    LOG.debug("model turned exit settings=%d seconds=%.4f", len(kept), time.perf_counter() - started)
    return result


@timed("model_spec.variants")
def variants(named):
    result = []
    for family in named["view"].options:
        found = [{"view": family}]
        for name in VARIED:
            if named[name].active(found[0]):
                found = [dict(spec, **{name: option}) for spec in found for option in named[name].options]
        result += found
    return result


@timed("model_spec.turnable")
def turnable(registry, parameters, settings):
    named = {parameter.name: parameter for parameter in parameters}
    for name in named[REALISATION].options if REALISATION in named else ():
        if name == NONE or not realisation(name).back:
            continue
        for spec in variants(named):
            if not named[REALISATION].active(spec) or not depths(spec, name, registry.schema, named[DEPTH].options):
                continue
            for setting, kept in turning(registry, dict(spec, **{REALISATION: name}), settings).items():
                if not kept:
                    (held,) = [parameter.options for parameter in settings if parameter.name == setting]
                    raise ValueError(
                        "the bounds leave %s the values %s alone, and the family %s of a model the bounds leave the "
                        "realisation of options %s reads the facts its edit adds under none of them"
                        % (setting, ", ".join(held), spec["view"], name)
                    )


@timed("model_spec.unstructured")
def unstructured(spec):
    if spec["view"] == "snapshot" and spec["snapshot/spatial"] == "none":
        return "its spatial layer is none, and it reads options put into its graph through its spatial layers"
    return STRUCTURE[spec["view"]]


@timed("model_spec.relationless")
def relationless(spec):
    return spec["view"] == "memory" and spec["memory/update"] == UNRELATED


@timed("model_spec.edits")
def edits(spec):
    if spec["view"] == "snapshot":
        return SPATIALLY[spec["snapshot/spatial"]]
    return () if relationless(spec) else EDITED[spec["view"]]


@timed("model_spec.unedited")
def unedited(spec):
    held = edits(spec)
    if held == EDITS:
        return None
    lacking = " nor ".join(PARTS[part] for part in EDITS if part not in held)
    if relationless(spec):
        return "its update %s reads observations and tasks and no relation, so neither %s" % (UNRELATED, lacking)
    if not held:
        return "it reads no edge, so neither %s" % lacking
    return "it reads %s, and not %s" % (" and ".join(PARTS[part] for part in held), lacking)


@timed("model_spec.showing")
def showing(value):
    return RELATION in value.split("+")


def wanting(spec, schema):
    started = time.perf_counter()
    held, result = edits(spec), None
    if not held:
        result = unedited(spec)
    elif held != EDITS:
        lacking = [
            (task, field) for task in schema.decided() for field in fielded(schema, task) if NEEDS[field["sort"]] not in held
        ]
        if lacking:
            task, field = lacking[0]
            result = "an edit makes the field %s of the kind of task %s %s, and %s" % (
                field["name"],
                task,
                MADE[field["sort"]],
                unedited(spec),
            )
    LOG.debug("model wanting exit wanting=%s seconds=%.6f", result is not None, time.perf_counter() - started)
    return result


@timed("model_spec.ignored")
def ignored(spec, name, schema):
    if name == NONE:
        return None
    entry = realisation(name)
    if entry.structure:
        reason = unstructured(spec)
        if reason is None or not depth(spec):
            return None
        return "%s; it reads options put there at the %s 0 alone, where none of its layers passes a message" % (reason, DEPTH)
    return wanting(spec, schema) if entry.back else None


def depths(spec, name, schema, offered):
    started = time.perf_counter()
    stated = dict(spec, **{REALISATION: name})
    if deep(stated):
        result = tuple(value for value in offered if ignored(dict(stated, **{DEPTH: value}), name, schema) is None)
    else:
        result = offered if ignored(stated, name, schema) is None else ()
    LOG.debug("model depths exit name=%s depths=%d seconds=%.6f", name, len(result), time.perf_counter() - started)
    return result


def unreadable(spec, name, schema, offered):
    started = time.perf_counter()
    stated = dict(spec, **{REALISATION: name})
    result = ignored(dict(stated, **{DEPTH: offered[0]}) if deep(stated) else stated, name, schema)
    LOG.debug("model unreadable exit name=%s seconds=%.6f", name, time.perf_counter() - started)
    return result


def fitting(parameter, spec, schema, parameters):
    started = time.perf_counter()
    (offered,) = [held.options for held in parameters if held.name == DEPTH]
    if parameter.name == DEPTH:
        name, offered = spec[REALISATION], parameter.options
        kept = depths(spec, name, schema, offered)
        stops = "the model of the family %s reads the realisation of options %s at none of the values of %s left to it, %s"
        stops %= (spec["view"], name, DEPTH, ", ".join(map(str, offered)))
    else:
        name = parameter.options[0]
        kept = tuple(held for held in parameter.options if depths(spec, held, schema, offered))
        stops = "the model of the family %s reads none of the realisations of options left to it, %s" % (
            spec["view"],
            ", ".join(parameter.options),
        )
    if not kept:
        raise ValueError("%s: %s" % (stops, unreadable(spec, name, schema, offered)))
    result = parameter if kept == parameter.options else narrow((parameter,), {parameter.name: list(kept)})[0]
    LOG.debug("model fitting exit parameter=%s kept=%d seconds=%.6f", parameter.name, len(kept), time.perf_counter() - started)
    return result


def drawn(parameters, choose, schema):
    started = time.perf_counter()
    spec = {}
    for parameter in parameters:
        if parameter.active(spec):
            held = fitting(parameter, spec, schema, parameters) if parameter.name in (REALISATION, DEPTH) else parameter
            spec[parameter.name] = choose(held)
    LOG.debug("model drawn exit parameters=%d seconds=%.6f", len(spec), time.perf_counter() - started)
    return spec


@timed("model_spec.placed")
def placed(parameters, fixed, schema):
    named = {parameter.name: parameter for parameter in parameters}
    if REALISATION not in named:
        return
    held, offered = named[REALISATION].options, named[DEPTH].options
    choices = [(name,) for name in held] if "model/" + REALISATION in fixed else [held]
    levels = [(value,) for value in offered] if "model/" + DEPTH in fixed else [offered]
    unread, names, values = {}, [], []
    for spec in variants(named):
        if not named[REALISATION].active(spec):
            continue
        for group in choices:
            for level in levels:
                if any(depths(spec, name, schema, level) for name in group):
                    continue
                unread.setdefault(spec["view"], unreadable(spec, group[0], schema, level))
                names += [name for name in group if name not in names]
                structural = any(realisation(name).structure for name in group)
                values += [value for value in level if structural and value not in values]
    if not unread:
        return
    families, reasons = ", ".join(unread), "; ".join("%s: %s" % pair for pair in unread.items())
    left = "the realisations of options %s alone%s, each putting options into the graph of its family" % (
        ", ".join(names),
        " at the %s %s" % (DEPTH, ", ".join(map(str, sorted(values)))) if values else "",
    )
    if not fixed:
        raise ValueError(
            "the bounds leave the model %s, and leave it the families %s, which read none of them there: %s"
            % (left, families, reasons)
        )
    raise ValueError(
        "the random parameters %s fix one value for every model of a proposal, and with the bounds they may leave the "
        "model %s, and the families %s, which read none of them there: %s" % (", ".join(fixed), left, families, reasons)
    )


def realisable(registry, parameters):
    started = time.perf_counter()
    held = [parameter for parameter in parameters if parameter.name == REALISATION]
    result = parameters
    if held:
        offered = [name for name in held[0].options if name == NONE or refused(name, registry) is None]
        result = narrow(parameters, {REALISATION: offered})
    LOG.debug("model realisable exit realisations=%d seconds=%.4f", len(held), time.perf_counter() - started)
    return result


def built(offered, events, encoded, decides, families):
    started = time.perf_counter()
    updates = UPDATES if events else tuple(update for update in UPDATES if update != "mutual_rnn")
    parameters = [
        choice("view", offered, offered[0]),
        number("hidden", 1, 128, 32),
        choice("snapshot/spatial", SPATIAL, "relational", view("snapshot")),
        number(
            "snapshot/layers",
            1,
            3,
            2,
            lambda spec: spec.get("view") == "snapshot" and spec["snapshot/spatial"] != "none",
        ),
        number("snapshot/chebyshev_order", 1, 4, 2, equals("snapshot/spatial", "chebyshev")),
        choice("snapshot/temporal", TEMPORAL, "entity_gru", view("snapshot")),
        choice(
            "snapshot/sequence",
            ("gru", "lstm", "attention"),
            "gru",
            equals("snapshot/temporal", "graph_sequence"),
        ),
        number(
            "snapshot/temporal_layers",
            1,
            4,
            2,
            among("snapshot/temporal", ("entity_tcn", "graph_cell", "evolving", "coupling_edges")),
        ),
        number("snapshot/diffusion_hops", 0, 4, 0, equals("snapshot/temporal", "graph_cell")),
        number("snapshot/heads", 1, 4, 2, attends),
        choice("snapshot/message", MESSAGES, "gated", equals("snapshot/temporal", "coupling_edges")),
        choice("mixed/aggregation", ("mean", "attention"), "mean", view("mixed")),
        choice("mixed/update", ("gru", "residual"), "gru", view("mixed")),
        number("mixed/layers", 1, 3, 2, view("mixed")),
        choice("event_graph/message", MESSAGES, "gated", view("event_graph")),
        number("event_graph/layers", 1, 3, 2, view("event_graph")),
        choice("memory/update", updates, "mailbox_gru", view("memory")),
        choice("memory/embedding", ("state", "attention"), "attention", memory_state),
        flag("memory/pair_terms", False, memory_state),
        number(
            "memory/neighbors",
            4,
            64,
            8,
            lambda spec: spec.get("memory/embedding") == "attention" or spec.get("memory/pair_terms") is True,
        ),
        number("memory/integration_steps", 1, 16, 2, equals("memory/update", "antisymmetric")),
        real("memory/damping", 0.001, 1.0, 0.1, equals("memory/update", "antisymmetric"), log=True),
        number("projection/hops", 1, 4, 2, view("projection")),
        number("projection/width", 16, 256, 16, view("projection")),
        real("projection/decay", 1e-7, 1.0, 0.05, view("projection"), log=True),
        number("projection/neighbors", 4, 64, 8, view("projection")),
        choice("sequence/operator", OPERATORS, "transformer", view("sequence")),
        number("sequence/neighbors", 4, 64, 8, view("sequence")),
        number("sequence/layers", 1, 3, 2, view("sequence")),
        choice("sequence/structure", STRUCTURES, "none", view("sequence")),
        number("sequence/patch", 1, 16, 2, equals("sequence/operator", "transformer")),
        number("sequence/heads", 1, 4, 2, equals("sequence/operator", "transformer")),
        number("sequence/state", 4, 64, 4, equals("sequence/operator", "ssm")),
        number("tree/depth", 1, 2, 2, view("tree")),
        number("tree/neighbors", 4, 20, 8, view("tree")),
        number("walks/count", 1, 32, 4, view("walks")),
        number("walks/hops", 1, 4, 3, view("walks")),
        number("affinity/buffer", 1, 32, 4, view("affinity")),
        number("signal/topk", 1, 20, 3, view("signal")),
        number("signal/hops", 1, 4, 1, view("signal")),
        number("signal/layers", 1, 3, 2, view("signal")),
        real("dropout", 0.0, 0.5, 0.0, reads_dropout),
        choice(
            "time/encoding",
            ENCODINGS,
            "log_cosine",
            lambda spec: encoded or reads_time(spec),
        ),
        flag("keys/embedding", False, embeds_keys),
        number("keys/table", 4096, 65536, 16384, lambda spec: spec.get("keys/embedding") is True),
    ]
    parameters += [flag(DERIVES % name, False) for name in unread()]
    if decides:
        parameters.append(
            choice(REALISATION, (NONE,) + REALISATIONS.names(), None, lambda spec: optioned(spec, families))
        )
        parameters.append(choice(DEPTH, DEPTHS, None, deep))
    LOG.debug(
        "model space built exit seconds=%.4f parameters=%d views=%d",
        time.perf_counter() - started,
        len(parameters),
        len(offered),
    )
    return tuple(parameters)


def blocks(spec):
    kind = spec["view"]
    if kind == "snapshot":
        return frozenset(
            ("snapshot", "lifted") if spec["snapshot/temporal"] == "coupling_edges" else ("snapshot",)
        )
    if kind == "mixed":
        return frozenset(("mixed",))
    if kind == "event_graph":
        return frozenset(("events", "dag"))
    if kind == "memory":
        result = {"events"}
        if spec.get("memory/embedding") == "attention" or spec.get("memory/pair_terms"):
            result.update(("history", "history_targets"))
        if spec["memory/update"] in ("intensity", "antisymmetric"):
            result.add("snapshot")
        return frozenset(result)
    if kind == "projection":
        return frozenset(("events", "history", "history_targets"))
    if kind == "sequence":
        result = {"history", "history_targets", "period_history"}
        if spec["sequence/structure"] == "overlap":
            result.add("structure")
        return frozenset(result)
    if kind == "tree":
        return frozenset(("tree", "period_history"))
    if kind == "walks":
        return frozenset(("walks",))
    return {"affinity": frozenset(("events", "affinity")), "signal": frozenset(("signal",))}[kind]


@timed("model_spec.windowed")
def windowed(spec):
    return spec["view"] == "affinity"


@timed("model_spec.shared_frames")
def shared_frames(spec):
    return spec["view"] == "snapshot" and spec["snapshot/temporal"] != "coupling_edges"


@timed("model_spec.temporal")
def temporal(spec, modes):
    return spec["view"] == "snapshot" and spec["snapshot/temporal"] in modes


@timed("model_spec.row_local")
def row_local(spec):
    return spec["view"] in ("tree", "walks", "sequence", "mixed", "event_graph")


@timed("model_spec.groups")
def groups(spec):
    wanted = {REPRESENTATIONS[block] for block in blocks(spec) if block in REPRESENTATIONS}
    return tuple(group for group in GROUPS if group in wanted)


@timed("model_spec.unread")
def unread():
    return tuple(name for name in REGISTERED.names() if name not in GROUPS)


@timed("model_spec.derived")
def derived(spec):
    extra = unread()
    lacking = [DERIVES % name for name in extra if DERIVES % name not in spec]
    if lacking:
        raise KeyError(
            "the model states no %s; a representation registered outside the groups the families read is "
            "derived where the model states it" % ", ".join(lacking)
        )
    return groups(spec) + tuple(name for name in extra if spec[DERIVES % name])


@timed("model_spec.neighbors")
def neighbors(spec):
    return spec[spec["view"] + "/neighbors"]


@timed("model_spec.describe")
def describe(spec):
    parts = [spec["view"]]
    for name, value in spec.items():
        if name != "view":
            parts.append(
                "%s %s" % (name.split("/", 1)[-1], "%.3g" % value if isinstance(value, float) else value)
            )
    return " ".join(parts)


@timed("model_spec.configured")
def configured(registry, values):
    return specs.configured(space(registry), values)


def configuration(registry, values, pipeline, graph, seed, splits):
    started = time.perf_counter()
    spec = configured(registry, values)
    resolved(registry, pipeline)
    graph = specs.canonical(specs.settings(registry, groups(spec)), graph, GRAPH)
    result = dict(
        seed=seed, model=spec, pipeline=pipeline, graph=graph, splits=splits, **specs.default(TRAINING)
    )
    LOG.debug("configuration exit entries=%d seconds=%.4f", len(pipeline), time.perf_counter() - started)
    return result


def samples(registry):
    started = time.perf_counter()
    parameters = realisable(registry, space(registry))
    switches = [
        parameter
        for parameter in parameters
        if parameter.kind in ("choice", "bool")
        and isinstance(parameter.options[0], (str, bool))
        and parameter.name != "view"
    ]
    result, seen, schema = [], set(), registry.schema

    def add(values):
        found = []
        for spec in starts(values):
            if REALISATION in spec and ignored(spec, spec[REALISATION], schema) is not None:
                continue
            found.append(spec)
            if tuple(spec.items()) not in seen:
                seen.add(tuple(spec.items()))
                result.append(spec)
        return found

    def starts(values):
        spec = {}
        for parameter in parameters:
            if parameter.active(spec):
                if parameter.default is None and parameter.name not in values:
                    held = fitting(parameter, spec, schema, parameters) if parameter.name in (REALISATION, DEPTH) else parameter
                    return [found for option in held.options for found in starts(dict(values, **{parameter.name: option}))]
                spec[parameter.name] = parameter.given(values)
        return [spec]

    for name in views(registry):
        for base in add(dict(view=name)):
            first = [
                spec
                for parameter in switches
                if parameter.active(base)
                for option in parameter.options
                for spec in add(dict(base, **{parameter.name: option}))
            ]
            for spec in first:
                for parameter in switches:
                    if parameter.name in spec:
                        for option in parameter.options:
                            add(dict(spec, **{parameter.name: option}))
    LOG.debug("model samples exit seconds=%.4f samples=%d", time.perf_counter() - started, len(result))
    return result
