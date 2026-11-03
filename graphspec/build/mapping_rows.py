from collections.abc import Mapping as Held

from graphspec.infra.timing import timed

from .expression_types import EXTENDED_TIME
from .mappings import TIMES
from .table_types import Context, broadcast, dependencies, take

NUMBERS = ("number", "ordered", "duration")
TYPED = dict(number="number", ordered="number", duration="number", flag="flag", category="category")
FORMED = ("always", "state", "event", "change", "signal")


class Reader:
    @timed("mapping_rows.Reader")
    def __init__(self, session, table, name, plan):
        self.frame = session.get(table)
        self.session, self.name, self.plan = session, name, plan
        self.context = Context(name, plan, session.unit, session.retained())

    @property
    @timed("mapping_rows.Reader.count")
    def count(self):
        return self.frame.table.num_rows

    @timed("mapping_rows.Reader.row")
    def row(self, place):
        return self.frame.positions[place].as_py() if self.frame.positions is not None else place

    @timed("mapping_rows.Reader.where")
    def where(self, text, field="where"):
        if text is None:
            return self
        values, kind = self.values(text, field)
        if kind.name not in ("flag", "missing"):
            raise ValueError("declaration %s field %s gives a flag" % (self.name, field))
        return self.taken([place for place, value in enumerate(values) if value is True])

    @timed("mapping_rows.Reader.taken")
    def taken(self, places):
        result = object.__new__(Reader)
        result.session, result.name, result.plan, result.context = self.session, self.name, self.plan, self.context
        result.frame = take(self.frame, places)
        return result

    @timed("mapping_rows.Reader.values")
    def values(self, text, field, outcome=False):
        if not outcome:
            marks = dependencies(self.frame, text)[1]
            if marks:
                raise ValueError("declaration %s field %s reads an outcome, known from %s; a mapping reads outcomes "
                                 "only in a target and in the outcome of a decision"
                                 % (self.name, field, ", ".join(mark[1] for mark in marks)))
        value = self.context.evaluate(self.frame, text, field)
        if value.kind.arrow == EXTENDED_TIME:
            raise ValueError("declaration %s field %s gives the time never" % (self.name, field))
        return broadcast(value, self.count).to_pylist(), value.kind

    @timed("mapping_rows.Reader.texts")
    def texts(self, stated, field, missing="stop", outcome=False):
        parts = [self.values(text, field, outcome)[0]
                 for text in (stated if isinstance(stated, (list, tuple)) else [stated])]
        result = []
        for place in range(self.count):
            found = [spelled(values[place]) for values in parts]
            if None in found:
                if missing == "stop":
                    raise ValueError("declaration %s field %s row %d reads a missing value as a key"
                                     % (self.name, field, self.row(place)))
                result.append(None)
            else:
                result.append(tuple(found) if isinstance(stated, (list, tuple)) else found[0])
        return result

    @timed("mapping_rows.Reader.listed")
    def listed(self, text, field, outcome=False):
        values, kind = self.values(text, field, outcome)
        return [None if value is None else [spelled(item) for item in (value if kind.name == "list" else [value])]
                for value in values]

    @timed("mapping_rows.Reader.moments")
    def moments(self, text, field, missing="stop", outcome=False):
        values, kind = self.values(text, field, outcome)
        result = []
        for place, value in enumerate(values):
            if value is None:
                if missing == "stop":
                    raise ValueError("declaration %s field %s row %d reads a missing value as a time"
                                     % (self.name, field, self.row(place)))
                result.append(None)
            elif kind.name == "time" or kind.name == "number" and float(value).is_integer():
                result.append(int(value))
            else:
                raise ValueError("declaration %s field %s row %d gives %r, not a whole time"
                                 % (self.name, field, self.row(place), value))
        return result

    @timed("mapping_rows.Reader.features")
    def features(self, stated, field="features", outcome=False):
        columns, values = {}, {}
        for column, encoder in (stated or {}).items():
            found, kind = self.values(column if column.isidentifier() else "column(%r)" % column,
                                      "%s.%s" % (field, column), outcome)
            if kind.name in ("vector", "point"):
                for place in range(kind.width):
                    name = "%s.%d" % (column, place)
                    columns[name] = dict(type="number", encoder=encoder)
                    values[name] = [None if value is None else float(value[place]) for value in found]
            elif kind.name in TYPED:
                columns[column] = dict(type=TYPED[kind.name], encoder=encoder)
                values[column] = [None if value is None else float(value) if kind.name in NUMBERS else value
                                  for value in found]
            else:
                raise ValueError("declaration %s field %s.%s is of the type %s, which is never a feature"
                                 % (self.name, field, column, kind.name))
        return columns, [{name: values[name][place] for name in values} for place in range(self.count)]

    @timed("mapping_rows.Reader.grouped")
    def grouped(self, keys):
        found = {}
        for place, key in enumerate(keys):
            if key is not None:
                found.setdefault(key, []).append(place)
        return found


@timed("mapping_rows.spelled")
def spelled(value):
    if value is None:
        return None
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    if isinstance(value, (list, tuple, dict)):
        raise ValueError("a key is one value, found %r" % (value,))
    return str(value)


@timed("mapping_rows.joined")
def joined(key):
    return "|".join(key) if isinstance(key, tuple) else key


@timed("mapping_rows.when")
def when(reader, fields, timed_model, clocks):
    count = reader.count
    holds = fields.get("holds")
    result = dict(form=["state"] * count, since=[None] * count, until=[None] * count, known=[None] * count,
                  replaced=[None] * count, seen=[None] * count, seq=[0] * count)
    if fields.get("seen") is not None:
        values, _ = reader.values(fields["seen"], "seen")
        result["seen"] = [None if value is None else sorted(value) if isinstance(value, list) else [str(value)]
                          for value in values]
    if not timed_model:
        for name in TIMES:
            if name in fields and name != "seen":
                raise ValueError("declaration %s states %s, and no mapping of the script states a time"
                                 % (reader.name, name))
        return result
    for name in ("holds", "known"):
        if name not in fields:
            raise ValueError("declaration %s lacks %s, which every mapping of facts states where the graphs "
                             "have time" % (reader.name, name))
    if holds != "always":
        holds = dict(holds)
        form = holds.pop("name")
        if form not in FORMED:
            raise ValueError("declaration %s field holds names the form %s, which mappings do not state; snapshot "
                             "needs a rule of graphspec.BETWEEN" % (reader.name, form))
        result.update(times(reader, form, holds))
    if fields["known"] is not None:
        result["known"] = reader.moments(fields["known"], "known")
    if fields.get("replaced") is not None:
        result["replaced"] = reader.moments(fields["replaced"], "replaced", missing="omit")
    if (fields["known"] is not None or fields.get("replaced") is not None) and not clocks:
        raise ValueError("declaration %s states known or replaced, which needs the model clocks" % reader.name)
    ties = fields.get("ties")
    if isinstance(ties, Held) and ties["name"] == "sequence":
        if "seq" not in ties:
            raise ValueError("declaration %s field ties states the rule sequence and lacks seq" % reader.name)
        result["seq"] = reader.moments(ties["seq"], "ties.seq")
    return result



@timed("mapping_rows.times")
def times(reader, form, holds):
    count = reader.count
    stated = dict(state=("since", "until"), event=("at",), change=("at",), signal=("index", "start", "stride"))[form]
    if sorted(holds) != sorted(stated):
        raise ValueError("declaration %s field holds of the form %s states exactly %s"
                         % (reader.name, form, ", ".join(stated)))
    if form == "state":
        since = [None] * count if holds["since"] is None else reader.moments(holds["since"], "holds.since")
        until = [None] * count if holds["until"] is None else reader.moments(holds["until"], "holds.until", "omit")
        return dict(since=since, until=until)
    if form == "signal":
        start, index, stride = holds["start"], holds["index"], holds["stride"]
        since = reader.moments("(%s) + (%s) * (%s)" % (start, index, stride), "holds")
        until = reader.moments("(%s) + ((%s) + 1) * (%s)" % (start, index, stride), "holds")
        return dict(since=since, until=until)
    at = reader.moments(holds["at"], "holds.at")
    return dict(form=[form] * count, since=at, until=[moment + 1 for moment in at] if form == "event" else [None] * count)


@timed("mapping_rows.following")
def following(groups, moments, where):
    order = sorted(range(len(moments)), key=lambda place: (repr(groups[place]), moments[place]))
    until = [None] * len(moments)
    for before, after in zip(order, order[1:]):
        if groups[before] == groups[after]:
            if moments[before] == moments[after]:
                raise ValueError("%s states two changes of one thing at the time %d" % (where, moments[after]))
            until[before] = moments[after]
    return until
