EDGE = "edge_"
ROLES = ("actor", "subject", "context")
FACTS = ("seq", "step", "present")
RESERVED = ("graph", "observation", "observed", "query") + tuple("query_" + role for role in ROLES)
LEVELS = ("node", "query", "graph", "path")


def feature_width(kind):
    if not (kind.startswith("FLOAT[") and kind.endswith("]")):
        raise ValueError("expected a FLOAT[n] column, found " + kind)
    width = int(kind[6:-1])
    if width < 1:
        raise ValueError("feature column width must be positive")
    return width


class TableSchema:
    def __init__(self, kinds):
        self.kinds = kinds
        required = (("graph", "NODE"), ("observation", "NODE"), ("observed", "REL"))
        queries = (("query", "NODE"),) + tuple(("query_" + role, "REL") for role in ROLES)
        self.queries = any(name in kinds for name, _ in queries)
        for name, kind in required + (queries if self.queries else ()):
            if kinds.get(name) != kind:
                raise ValueError("graph source requires a %s table named %s" % (kind.lower(), name))
        observed = self.pairs("observed")
        if any(target != "observation" for _, target in observed):
            raise ValueError("observed relationships must point to observation")
        sources = {source for source, _ in observed}
        self.nodes = tuple(name for name in kinds if name in sources)
        if not self.nodes or any(kinds[node] != "NODE" for node in self.nodes):
            raise ValueError("observed relationships require declared node tables")
        self.relations = tuple(
            self.relation(name) for name, kind in kinds.items() if kind == "REL" and name.startswith(EDGE)
        )
        self.features = feature_width(self.columns("observation")["x"])
        self.signals = 0
        if self.queries:
            for role in ROLES:
                if any(
                    source != "query" or target not in self.nodes
                    for source, target in self.pairs("query_" + role)
                ):
                    raise ValueError("query_%s must connect queries to observed node types" % role)
            typed = self.columns("query")
            if feature_width(typed["x"]) != self.features:
                raise ValueError("query.x and observation.x must have the same width")
            if ("signal" in typed) != ("signal_observed" in typed):
                raise ValueError("query signals require matching signal and signal_observed columns")
            if "signal" in typed:
                self.signals = feature_width(typed["signal"])
                if feature_width(typed["signal_observed"]) != self.signals:
                    raise ValueError("query signal widths must match")

    def relation(self, name):
        pairs, typed = self.pairs(name), self.columns(name)
        missing = sorted(set(FACTS) - set(typed))
        if missing:
            raise ValueError("%s lacks the columns %s" % (name, ", ".join(missing)))
        if typed["seq"] != "INT32" or typed["step"] != "INT32" or typed["present"] not in ("BOOL", "BOOLEAN"):
            raise ValueError("%s requires seq INT32, step INT32, and boolean present columns" % name)
        attributes = tuple(field for field in typed if field not in FACTS)
        if any(typed[field] != "FLOAT" for field in attributes):
            raise ValueError("attributes of %s must be FLOAT columns" % name)
        if not {node for pair in pairs for node in pair} <= set(self.nodes):
            raise ValueError("%s connects tables that are not observed node types" % name)
        return dict(name=name[len(EDGE) :], pairs=pairs, attributes=attributes)

    def destinations(self, table):
        return tuple(dict.fromkeys(target for _, target in self.pairs(table)))

    def target(self, name):
        if self.kinds.get(name) != "NODE" or self.kinds.get(name + "_of") != "REL":
            raise ValueError("graph source requires target table %s with a %s_of relationship" % (name, name))
        if any(source != name for source, _ in self.pairs(name + "_of")):
            raise ValueError("%s_of relationships must start at %s" % (name, name))
        destinations = set(self.destinations(name + "_of"))
        typed = self.columns(name)
        if destinations == {"query"}:
            if "tau" not in typed:
                return "query"
            if typed["tau"] not in ("FLOAT", "DOUBLE"):
                raise ValueError("%s.tau must be a FLOAT or DOUBLE column, found %s" % (name, typed["tau"]))
            return "path"
        if "tau" in typed:
            raise ValueError("%s carries a tau column, which only targets on queries may have" % name)
        if destinations == {"graph"}:
            return "graph"
        if not destinations or not destinations <= set(self.nodes):
            raise ValueError("%s_of must point to query, graph, or observed node types" % name)
        if typed.get("step") != "INT32":
            raise ValueError("%s labels node types and requires a step INT32 column" % name)
        return "node"
