import logging
import time

from .declarations import EdgeType
from .registry import Registry

LOG = logging.getLogger(__name__)
TASK_KEYS = frozenset(("name", "database", "staging", "target"))
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


def target_level(schema, task):
    unknown = sorted(set(task) - TASK_KEYS)
    if unknown:
        raise ValueError("unknown task keys: " + ", ".join(unknown))
    found = schema.target(task["target"])
    if found != "node" and not schema.queries:
        raise ValueError(
            "%s-level targets are predicted at queries, so the database needs a query table" % found
        )
    return found


def registry(schema, task, series):
    started = time.perf_counter()
    level = target_level(schema, task)
    if (level == "path") != bool(series) or list(series) != sorted(set(series)):
        raise ValueError("path targets need their sorted distinct series names, and other targets none")
    if schema.features < 2 or schema.features % 2:
        raise ValueError("feature vectors need an even width: values, then presence")
    clash = sorted(
        [
            relation["name"]
            for relation in schema.relations
            if not relation["name"] or ":" in relation["name"] or relation["name"] in RESERVED
        ]
        + [kind for kind in schema.nodes if not kind or ":" in kind or kind == "step"]
    )
    if clash:
        raise ValueError("names collide with derived relation names: " + ", ".join(clash))
    nodes = schema.nodes
    edges = [
        EdgeType(relation["name"], ("base",), relation["pairs"], relation["attributes"])
        for relation in schema.relations
    ]
    for relation in schema.relations:
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
    labelled = set(schema.destinations(task["target"] + "_of")) if level == "node" else set()
    extras = dict(
        level=level,
        queries=schema.queries,
        signals=0 if level == "node" else schema.signals,
        base=[relation["name"] for relation in schema.relations],
        protected=[kind for kind in nodes if kind in labelled],
    )
    if level == "path":
        extras["series"] = list(series)
    result = Registry(task["name"], nodes, edges, schema.features // 2, MARKERS, extras)
    LOG.info(
        "vocabulary exit seconds=%.3f nodes=%d relations=%d level=%s",
        time.perf_counter() - started,
        len(nodes),
        result.relations,
        level,
    )
    return result
