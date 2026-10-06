import logging
import time

import numpy as np

from graphspec.infra.timing import timed
from graphspec.registries import Names

from .fields import fields, named
from .references import SORTS
from .targets import TARGETS, listed, real

LOG = logging.getLogger(__name__)
GENERATORS = Names("generator of options", True)
RULES = Names("rule of options", True)
ROLES = ("node", "edge", "new")
TOGETHER = ("set", "sequence")
RECORDED = "recorded"
WHOLE = -1
ACTION = ("roles", "parameters")
ROLE = ("sort", "kinds")
RULE = ("name", "parameters")
STATED = ("options", "concurrent", "next", "ended", "discount")
OPTION = ("key", "actions", "together", "available", "chosen", "order", "shown", "logged", "origin", "outcome")
ACTED = ("kind", "actor", "targets", "parameters")
DRAWN = ("generator", "seed", "drawn")
ACTING = ("id", "graph", "kind", "actor")
PARTS = ("state", "decisions", "chosen", "rule")
GENERATED = ("name", "seed", "repeats", "parameters")
SEEDED = 2**63 - 1


@timed("decisions.ruled")
def ruled(stated, schema, where):
    if stated is None:
        return None
    where = "the rule of the options of the target of " + where
    stated = fields(stated, RULE, where)
    entry = RULES.named(stated["name"])
    return dict(name=stated["name"], parameters=entry.checked(stated["parameters"], schema, where))


def generator(name):
    started = time.perf_counter()
    entry = GENERATORS.named(name)
    if (
        not all(hasattr(entry, part) for part in ("reads", "probable", "checked"))
        or not isinstance(entry.reads, tuple)
        or any(part not in PARTS for part in entry.reads)
        or type(entry.probable) is not bool
        or not callable(entry)
        or not callable(entry.checked)
    ):
        raise ValueError(
            "the generator of options %s states reads, a tuple of what it reads among %s; probable, whether it "
            "states the probability of each draw, true or false; checked, which checks its parameters; and draws "
            "when called" % (name, ", ".join(PARTS))
        )
    LOG.debug("decisions generator exit name=%s seconds=%.6f", name, time.perf_counter() - started)
    return entry


def generators(stated, schema, target, references, where):
    started = time.perf_counter()
    if not isinstance(stated, list):
        raise ValueError("the target of %s states its generators of options as a list, found %r" % (where, stated))
    result = []
    for place, held in enumerate(stated):
        inner = "the generator %d of the options of the target of %s" % (place, where)
        held = fields(held, GENERATED, inner)
        entry = generator(held["name"])
        if type(held["seed"]) is not int or not 0 <= held["seed"] <= SEEDED:
            raise ValueError("%s states its seed as a whole number from 0 to %d, found %r" % (inner, SEEDED, held["seed"]))
        if type(held["repeats"]) is not bool:
            raise ValueError(
                "%s states whether an option it draws may equal an option already present, true or false, found %r"
                % (inner, held["repeats"])
            )
        if "rule" in entry.reads and target["rule"] is None:
            raise ValueError("%s, %s, draws from the rule of its kind, which states none" % (inner, held["name"]))
        parameters = entry.checked(held["parameters"], schema, target, references, inner)
        result.append(dict(name=held["name"], seed=held["seed"], repeats=held["repeats"], parameters=parameters))
    about = target["about"]
    if result and about is not None and (
        SORTS.named(references[about]["sort"]).edges or set(references[about]["kinds"]) & set(schema.sets)
    ):
        raise ValueError(
            "the target of %s states generators of options and is about %s, whose facts a split's rule keeps from "
            "its tasks; a generator reads, when the bank is built, what the clock shows" % (where, about)
        )
    LOG.debug("decisions generators exit generators=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def acted(name, entry, schema, where):
    started = time.perf_counter()
    where = "the kind of action %s of %s" % (named(name, "a kind of action of " + where), where)
    entry = fields(entry, ACTION, where)
    roles = entry["roles"]
    if not isinstance(roles, dict):
        raise ValueError("%s states its roles as a mapping from name to sort and kinds, found %r" % (where, roles))
    held = {}
    for role, statement in roles.items():
        inner = "the role %s of %s" % (named(role, "a role of " + where), where)
        statement = fields(statement, ROLE, inner)
        sort, kinds = statement["sort"], statement["kinds"]
        if sort not in ROLES:
            raise ValueError("%s is of one of the sorts %s, found %r" % (inner, ", ".join(ROLES), sort))
        allowed = schema.nodes if sort == "new" else schema.entities()
        if (
            not isinstance(kinds, list)
            or not kinds
            or any(type(kind) is not str or kind not in allowed for kind in kinds)
            or len(set(kinds)) != len(kinds)
            or (sort == "new" and len(kinds) != 1)
        ):
            raise ValueError(
                "%s names %s, found %r"
                % (
                    inner,
                    "the one kind of node of the element it creates"
                    if sort == "new"
                    else "kinds of node or of set edge that are stated, each once",
                    kinds,
                )
            )
        held[role] = dict(sort=sort, kinds=list(kinds))
    columns = entry["parameters"]
    if (
        not isinstance(columns, list)
        or any(type(column) is not str or column not in schema.columns or column in ACTING for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError(
            "%s states its parameters as a list of distinct columns of the schema, none named %s, found %r"
            % (where, ", ".join(ACTING), columns)
        )
    LOG.debug("decisions acted exit kind=%s roles=%d seconds=%.4f", name, len(held), time.perf_counter() - started)
    return dict(roles=held, parameters=list(columns))


@timed("decisions.probable")
def probable(value):
    return value is None or real(value)


@timed("decisions.outcome")
def outcome(stated, series, where):
    if stated is None:
        return None
    if not isinstance(stated, dict) or set(stated) != set(series) or not all(real(value) for value in stated.values()):
        raise ValueError(
            "%s states an outcome as a finite number for every series of its kind, %s, found %r"
            % (where, ", ".join(series), stated)
        )
    return stated


def option(stated, series, where):
    started = time.perf_counter()
    stated = fields(stated, OPTION, where)
    named(stated["key"], "the key of " + where)
    if not isinstance(stated["actions"], list):
        raise ValueError("%s states its actions as a list, found %r" % (where, stated["actions"]))
    for place, action in enumerate(stated["actions"]):
        inner = "the action %d of %s" % (place, where)
        action = fields(action, ACTED, inner)
        if type(action["kind"]) is not str or type(action["actor"]) is not str:
            raise ValueError("%s names its kind and the key of its actor" % inner)
        targets, parameters = action["targets"], action["parameters"]
        if not isinstance(targets, dict) or any(
            type(role) is not str
            or not (
                type(target) is str
                or isinstance(target, list) and len(target) == 3 and all(type(part) is str for part in target)
            )
            for role, target in targets.items()
        ):
            raise ValueError(
                "%s states its targets as a mapping from role to the key of a node or of an element it creates, or "
                "to the key of the first node, the relation and the key of the second node of an edge, found %r"
                % (inner, targets)
            )
        if not isinstance(parameters, dict) or any(type(name) is not str for name in parameters):
            raise ValueError("%s states its parameters as a mapping from column to value, found %r" % (inner, parameters))
    if stated["together"] not in TOGETHER:
        raise ValueError("%s states its actions as one of %s, found %r" % (where, ", ".join(TOGETHER), stated["together"]))
    if type(stated["available"]) is not bool or type(stated["chosen"]) is not bool:
        raise ValueError("%s states whether it was available and whether it was chosen, true or false" % where)
    if stated["order"] is not None and type(stated["order"]) is not int:
        raise ValueError("%s states its order of choice as a whole number or none, found %r" % (where, stated["order"]))
    if not probable(stated["shown"]) or not probable(stated["logged"]):
        raise ValueError("%s states the probabilities under which it was shown and chosen as numbers or none" % where)
    origin = stated["origin"]
    if origin != RECORDED:
        origin = fields(origin, DRAWN, "the origin of " + where)
        if type(origin["generator"]) is not str or type(origin["seed"]) is not int or not probable(origin["drawn"]):
            raise ValueError(
                "%s states its origin as %s, or as the name of the generator that drew it, its seed, a whole number, "
                "and the probability of the draw, a number or none, found %r" % (where, RECORDED, origin)
            )
    held = stated["outcome"]
    if isinstance(held, list):
        if len(held) != len(stated["actions"]):
            raise ValueError("%s states an outcome per action, one for each of its actions" % where)
        if all(part is None for part in held):
            raise ValueError(
                "%s, of the key %s, states as its outcome per action %r, the outcome of none of its actions; an option "
                "with no outcome states none" % (where, stated["key"], held)
            )
        for place, part in enumerate(held):
            outcome(part, series, "the action %d of %s" % (place, where))
    else:
        outcome(held, series, where)
    LOG.debug("decisions option exit actions=%d seconds=%.4f", len(stated["actions"]), time.perf_counter() - started)
    return stated


def decision(stated, target, where):
    started = time.perf_counter()
    stated = fields(stated, STATED, where)
    if not isinstance(stated["options"], list):
        raise ValueError("%s states its options as a list, found %r" % (where, stated["options"]))
    for place, found in enumerate(stated["options"]):
        option(found, target["series"], "the option %d of %s" % (place, where))
    if not isinstance(stated["concurrent"], list) or any(type(key) is not int for key in stated["concurrent"]):
        raise ValueError(
            "%s states the decisions of other actors taken at the same time as a list of keys of tasks, found %r"
            % (where, stated["concurrent"])
        )
    following = stated["next"]
    if not isinstance(following, dict) or any(
        type(actor) is not str or type(key) is not int for actor, key in following.items()
    ):
        raise ValueError(
            "%s states the next decision of each of its actors as a mapping from the key of the actor to the key of "
            "the task, found %r" % (where, following)
        )
    if stated["ended"] is not None and type(stated["ended"]) is not bool:
        raise ValueError("%s states whether its episode ends as true, false or none, found %r" % (where, stated["ended"]))
    if not probable(stated["discount"]):
        raise ValueError("%s states its discount as a number or none, found %r" % (where, stated["discount"]))
    LOG.debug("decisions decision exit options=%d seconds=%.4f", len(stated["options"]), time.perf_counter() - started)
    return stated


class Decision:
    version = 3
    scaled = True
    matures = None
    names = None
    parameters = ("actors", "actions", "nothing", "rule", "generators")
    columns = dict(series="STRING", option="INT32", part="INT32", chosen="BOOL")
    inputs = ()
    outcomes = ("option", "part", "chosen")
    times = ()

    @timed("decisions.decision.checked")
    def checked(self, stated, schema, references, where):
        actors = stated["actors"]
        sort = SORTS.named(references[actors]["sort"]) if type(actors) is str and actors in references else None
        if sort is None or not sort.nodes or sort.edges:
            raise ValueError(
                "the target of %s names as its actors one of its references, of a sort that holds nodes only, "
                "found %r" % (where, actors)
            )
        actions = stated["actions"]
        if not isinstance(actions, dict) or not actions:
            raise ValueError(
                "the target of %s states its kinds of action as a mapping from name to roles and parameters, at "
                "least one, found %r" % (where, actions)
            )
        found = {name: acted(name, entry, schema, "the target of " + where) for name, entry in actions.items()}
        nothing = stated["nothing"]
        if nothing is not None:
            named(nothing, "the option that does nothing of the target of " + where)
        result = dict(actors=actors, actions=found, nothing=nothing, rule=ruled(stated["rule"], schema, where))
        result["generators"] = generators(
            stated["generators"], schema, dict(result, about=stated["about"]), references, where
        )
        return result

    @timed("decisions.decision.series")
    def series(self, stated, where):
        return listed(stated, where)

    @timed("decisions.decision.written")
    def written(self, series):
        return list(series)

    def points(self, rows, target, where):
        started = time.perf_counter()
        series = target["series"]
        owner, names = np.asarray(rows["task"], np.int64), list(rows["series"])
        strangers = sorted({repr(name) for name in names if name not in series})
        if strangers:
            raise ValueError(
                "a point of %s names the series %s, which its kind does not state" % (where, ", ".join(strangers))
            )
        if (
            any(type(value) is not int or value < 0 for value in rows["option"])
            or any(value is not None and (type(value) is not int or value < 0) for value in rows["part"])
            or any(type(value) is not bool for value in rows["chosen"])
        ):
            raise ValueError(
                "the points of %s state the place of their option, a whole number from 0; the place of their action, "
                "a whole number from 0, or none for the option as a whole; and whether their option was chosen, true "
                "or false" % where
            )
        held = np.asarray(rows["option"], np.int64)
        part = np.asarray([WHOLE if value is None else value for value in rows["part"]], np.int64)
        code = np.asarray([series.index(name) for name in names], np.int64)
        order = np.lexsort((code, part, held, owner))
        keys = np.stack((owner, held, part, code), 1)[order]
        if np.any((np.diff(keys, axis=0) == 0).all(1)):
            at = int(order[np.flatnonzero((np.diff(keys, axis=0) == 0).all(1))[0]])
            raise ValueError(
                "graph %s has two points of %s with one task, one option, one action and one series"
                % (rows["graph"][at], where)
            )
        grouped = np.unique(keys[:, :3], axis=0, return_counts=True)[1]
        options, mixed = np.unique(keys[:, :2], axis=0, return_inverse=True)
        whole = np.bincount(mixed.reshape(-1), keys[:, 2] == WHOLE, len(options))
        parted = np.bincount(mixed.reshape(-1), keys[:, 2] != WHOLE, len(options))
        if np.any(grouped != len(series)) or np.any((whole > 0) & (parted > 0)):
            raise ValueError(
                "the points of %s state the outcome of an option, or of each of its actions, as a number for every "
                "series of its kind, %s" % (where, ", ".join(series))
            )
        LOG.debug("decision points exit points=%d seconds=%.4f", len(owner), time.perf_counter() - started)
        return dict(series=names, option=held.tolist(), part=part.tolist(), chosen=list(rows["chosen"]))

    def staged(self, stated, target, where):
        started = time.perf_counter()
        rows = []
        for place, found in enumerate(decision(stated, target, where)["options"]):
            held = found["outcome"]
            parts = list(enumerate(held)) if isinstance(held, list) else [(None, held)]
            rows += [
                dict(option=place, part=part, chosen=found["chosen"], series=name, y=float(values[name]))
                for part, values in parts
                if values is not None
                for name in target["series"]
            ]
        LOG.debug("decision staged exit points=%d seconds=%.4f", len(rows), time.perf_counter() - started)
        return rows

    def read(self, rows):
        started = time.perf_counter()
        result = {}
        for row in rows:
            part = None if row["part"] in (None, WHOLE) else row["part"]
            result.setdefault((row["option"], part), {})[row["series"]] = row["y"]
        LOG.debug("decision read exit points=%d seconds=%.4f", len(rows), time.perf_counter() - started)
        return result

    @timed("decisions.decision.predicted")
    def predicted(self, values, task, fields):
        return values


@timed("decisions.deciding")
def deciding(target):
    return target is not None and isinstance(TARGETS.named(target["kind"]), Decision)


def roled(schema):
    started = time.perf_counter()
    actions = [
        list(entry["target"]["actions"].values()) if deciding(entry["target"]) else []
        for entry in schema.tasks["kinds"].values()
    ]
    width = max([len(found) for found in actions] + [1])
    depth = max([len(action["roles"]) for found in actions for action in found] + [1])
    result = np.full((len(actions), width, depth), -1, np.int8)
    for kind, found in enumerate(actions):
        for place, action in enumerate(found):
            for role, entry in enumerate(action["roles"].values()):
                result[kind, place, role] = ROLES.index(entry["sort"])
    LOG.debug("roled exit kinds=%d seconds=%.4f", len(actions), time.perf_counter() - started)
    return result


def counted(schema, rows):
    started = time.perf_counter()
    kinds = schema.decided()
    result = np.asarray([row["chosen"] if row["kind"] in kinds else True for row in rows], bool).reshape(-1)
    LOG.debug("counted exit points=%d counted=%d seconds=%.4f", len(rows), int(result.sum()), time.perf_counter() - started)
    return result


def drawn(schema):
    started = time.perf_counter()
    ruled = [
        (kind, entry["target"]["rule"]["name"])
        for kind, entry in schema.tasks["kinds"].items()
        if deciding(entry["target"])
        and entry["target"]["rule"] is not None
        and not any("rule" in GENERATORS.named(held["name"]).reads for held in entry["target"]["generators"])
    ]
    if ruled:
        raise ValueError(
            "the kind of task %s states its options by the rule %s; the options of a rule are drawn by a generator "
            "of options before a fit reads them, and no generator has drawn them" % ruled[0]
        )
    LOG.debug("drawn exit kinds=%d seconds=%.4f", len(schema.tasks["kinds"]), time.perf_counter() - started)


@timed("decisions.generating")
def generating(schema):
    return any(schema.tasks["kinds"][kind]["target"]["generators"] for kind in schema.decided())


TARGETS.register("decision", Decision())
