import logging
import time

from .declarations import EdgeType
from .registry import Registry

LOG = logging.getLogger(__name__)
TASK_KEYS = frozenset(("name", "database", "staging", "target", "schema"))
MARKERS = ("non_relation", "turn_context")
HOPS = (2, 3)
COUPLING = "temporal"
UNTYPED = "untyped"
BOUNDARIES = ("boundary:first", "boundary:last", "boundary:both")
RESERVED = ("diffusion", "rewire", UNTYPED, COUPLING)


def projection(relation, side, first, second):
    return "project:%s:%s:%s:%s" % (relation, side, first, second)


def completion(first, second):
    return "complete:%s:%s" % (first, second)


def hop(distance):
    return "hop:%d" % distance


def target_level(source, task):
    unknown = sorted(set(task) - TASK_KEYS)
    if unknown:
        raise ValueError("unknown task keys: " + ", ".join(unknown))
    found = source.target(task["target"])
    if found != "node" and not source.queries:
        raise ValueError(
            "%s-level targets are predicted at queries, so the source needs a query table" % found
        )
    return found


def registry(source, task, series, encoding):
    started = time.perf_counter()
    level = target_level(source, task)
    if (level == "path") != bool(series) or list(series) != sorted(set(series)):
        raise ValueError("path targets need their sorted distinct series names, and other targets none")
    schema = encoding.schema
    nodes = tuple(schema.nodes)
    relations = [
        dict(
            name=name,
            table=table,
            pairs=entry["endpoints"],
            attributes=entry["columns"],
            symmetric=not entry["directed"],
        )
        for table, entry in schema.edges.items()
        for name in encoding.relations[table]
    ]
    clash = sorted(
        relation["name"]
        for relation in relations
        if type(relation["name"]) is not str
        or not relation["name"]
        or ":" in relation["name"]
        or relation["name"] in RESERVED
    )
    if clash:
        raise ValueError("names collide with derived relation names: " + ", ".join(map(str, clash)))
    edges = [
        EdgeType(
            relation["name"],
            ("base", relation["table"]),
            relation["pairs"],
            relation["attributes"],
            symmetric=relation["symmetric"],
        )
        for relation in relations
    ]
    for relation in relations:
        for side, position in (("sources", 0), ("targets", 1)):
            counterparts = {}
            for pair in relation["pairs"]:
                counterparts.setdefault(pair[position], set()).add(pair[1 - position])
            members = tuple(counterparts)
            for i, first in enumerate(members):
                for second in members[i:]:
                    if counterparts[first] & counterparts[second]:
                        edges.append(
                            EdgeType(
                                projection(relation["name"], side, first, second),
                                ("project", relation["name"], side, first, second),
                                ((first, second),),
                                symmetric=True,
                            )
                        )
    for i, first in enumerate(nodes):
        for second in nodes[i:]:
            edges.append(
                EdgeType(
                    completion(first, second), ("complete", first, second), ((first, second),), symmetric=True
                )
            )
    everything = tuple((first, second) for first in nodes for second in nodes)
    edges.extend(
        EdgeType(hop(distance), ("hop", str(distance)), everything, symmetric=True) for distance in HOPS
    )
    edges.append(EdgeType("diffusion", ("diffusion",), everything))
    edges.append(EdgeType("rewire", ("rewire",), everything, symmetric=True))
    edges.append(EdgeType(UNTYPED, ("untyped",), everything))
    edges.append(EdgeType(COUPLING, ("coupling",), everything, symmetric=True))
    edges.extend(
        EdgeType(name, ("boundary", name.split(":")[1]), everything, symmetric=True) for name in BOUNDARIES
    )
    labelled = set(source.destinations(task["target"] + "_of")) if level == "node" else set()
    extras = dict(
        level=level,
        queries=source.queries,
        signals=0 if level == "node" else source.signals,
        base=[relation["name"] for relation in relations],
        protected=[kind for kind in nodes if kind in labelled],
    )
    if level == "path":
        extras["series"] = list(series)
    result = Registry(task["name"], encoding, edges, MARKERS, extras)
    LOG.info(
        "vocabulary exit seconds=%.3f nodes=%d relations=%d level=%s",
        time.perf_counter() - started,
        len(nodes),
        result.relations,
        level,
    )
    return result
