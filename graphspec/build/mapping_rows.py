from collections.abc import Mapping as Held

from graphspec.infra.timing import timed

from .expression_types import EXTENDED_TIME, Kind
from .mapping_known import BEFORE, NEVER
from .mappings import TIMES
from .presence import BETWEEN
from .table_types import Context, broadcast, references, ruled, take

NUMBERS = ("number", "ordered", "duration")
TYPED = dict(number="number", ordered="number", duration="number", flag="flag", category="category")
FORMED = ("always", "state", "event", "change", "signal")
EARLY = "a fact chooses, dates and carries nothing by what is known after it"


class Reader:
    windows = {}
    asking = False

    @timed("mapping_rows.Reader")
    def __init__(self, session, table, name, plan):
        self.frame = session.get(table)
        self.session, self.name, self.plan, self.knowing = session, name, plan, session.knowing
        self.context = Context(name, plan, session.unit, session.retained())

    @timed("mapping_rows.Reader.framed")
    def framed(self, frame):
        result = object.__new__(Reader)
        result.session, result.name, result.plan, result.context = self.session, self.name, self.plan, self.context
        result.knowing, result.asking, result.windows, result.frame = self.knowing, self.asking, self.windows, frame
        return result

    @timed("mapping_rows.Reader.asked")
    def asked(self):
        result = self.framed(self.frame)
        result.asking = True
        return result

    @timed("mapping_rows.Reader.refused")
    def refused(self, text, field):
        if self.asking and field != "at":
            self.knowing.ended(self, text, field)
        found = self.knowing.outcomes(self, text)
        if found:
            raise ValueError("declaration %s field %s reads the outcome column %s; a mapping reads an outcome only in "
                             "a target, the outcome of a decision, and the where, times and features of a fact known "
                             "no earlier than it" % (self.name, field, ", ".join(found)))

    @timed("mapping_rows.Reader.windowed")
    def windowed(self, windows):
        result = self.framed(self.frame)
        result.windows = dict(windows)
        return result

    @timed("mapping_rows.Reader.scoped")
    def scoped(self, text, field, outcome, scopes):
        unknown = sorted(scopes - set(self.windows))
        if unknown:
            raise ValueError("declaration %s field %s reads %s rows, which it may not read there; later rows are read "
                             "by the value and known of a target, earlier rows by eligible"
                             % (self.name, field, ", ".join(unknown)))
        frames = {name: self.windows[name].frame for name in scopes}
        if not outcome:
            self.refused(text, field)
        elif self.asking:
            self.knowing.ended(self, text, field)
        values, kind = [], Kind("missing")
        for place in range(self.count):
            namespace = {name: take(frames[name], self.windows[name].places[place]) for name in scopes}
            value = self.context.evaluate(take(self.frame, [place]), text, field, namespace=namespace)
            values.append(broadcast(value, 1).to_pylist()[0])
            kind = value.kind if kind.name == "missing" else kind
        return values, kind

    @property
    @timed("mapping_rows.Reader.count")
    def count(self):
        return self.frame.table.num_rows

    @timed("mapping_rows.Reader.row")
    def row(self, place):
        return self.frame.positions[place].as_py() if self.frame.positions is not None else place

    @timed("mapping_rows.Reader.where")
    def where(self, text, field="where", outcome=False):
        if text is None:
            return self
        values, kind = self.values(text, field, outcome)
        if kind.name not in ("flag", "missing"):
            raise ValueError("declaration %s field %s gives a flag" % (self.name, field))
        return self.taken([place for place, value in enumerate(values) if value is True])

    @timed("mapping_rows.Reader.taken")
    def taken(self, places):
        return self.framed(take(self.frame, places))

    @timed("mapping_rows.Reader.values")
    def values(self, text, field, outcome=False):
        scopes = {scope for scope, _ in references(text)} - {None}
        if scopes:
            return self.scoped(text, field, outcome, scopes)
        if not outcome:
            self.refused(text, field)
        elif self.asking:
            self.knowing.ended(self, text, field)
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
            text = column if column.isidentifier() else "column(%r)" % column
            self.knowing.ended(self, text, "%s.%s" % (field, column))
            if outcome and column in self.frame.kinds and self.knowing.never(self.frame, column):
                raise ValueError("declaration %s field %s.%s reads the outcome column %s, which is never known and so "
                                 "never a feature" % (self.name, field, column,
                                                      ", ".join(self.knowing.named(self.frame, column))))
            found, kind = self.values(text, "%s.%s" % (field, column), outcome)
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

    @timed("mapping_rows.Reader.lately")
    def lately(self, stated, columns):
        found = {}
        for column in stated or {}:
            known = self.knowing.column(self.frame, column) if column in self.frame.kinds else None
            if known is not None:
                origins = self.knowing.named(self.frame, column)
                for name in columns:
                    head, _, tail = name.rpartition(".")
                    if name == column or head == column and tail.isdigit():
                        found[name] = (column, known, origins)
        return found

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


@timed("mapping_rows.sighted")
def sighted(reader, fields, graphs, timed_model):
    holds = fields.get("holds")
    if not timed_model or not isinstance(holds, Held) or holds.get("name") != "snapshot":
        return reader, graphs, None, None
    where = "declaration %s field holds" % reader.name
    if "between" not in holds:
        raise ValueError("%s.between: the form snapshot states the rule for the times between two sightings, "
                         "registered in graphspec.BETWEEN" % where)
    if sorted(holds) != ["at", "between", "name"]:
        raise ValueError("%s of the form snapshot states exactly at and between" % where)
    name, parameters = ruled(holds["between"], where + ".between")
    try:
        pieces, aged = BETWEEN.named(name)(reader.moments(holds["at"], "holds.at"), graphs, parameters, reader.name)
    except (KeyError, ValueError) as error:
        raise ValueError("%s.between: %s" % (where, error.args[0])) from error
    places = [piece[0] for piece in pieces]
    taken = reader.taken(places)
    times = dict(since=[piece[1] for piece in pieces], until=[piece[2] for piece in pieces])
    ages = None if aged is None else (aged[0], aged[1], [piece[3] for piece in pieces])
    return taken, [graphs[place] for place in places], times, ages


@timed("mapping_rows.aged")
def aged(ages, columns, values, where):
    if ages is None:
        return
    name, encoder, found = ages
    if name in columns:
        raise ValueError("%s states the feature %s, which the rule carried also names" % (where, name))
    columns[name] = dict(type="number", encoder=encoder)
    for held, age in zip(values, found):
        held[name] = float(age)


@timed("mapping_rows.when")
def when(reader, fields, timed_model, clocks, sightings):
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
    stated = dict(known=fields["known"], replaced=fields.get("replaced"))
    if sightings is not None:
        result.update(sightings)
    elif holds != "always":
        holds = dict(holds)
        form = holds.pop("name")
        if form not in FORMED:
            raise ValueError("declaration %s field holds names the form %s, which mappings do not state"
                             % (reader.name, form))
        result.update(times(reader, form, holds))
        stated.update({"holds." + name: text for name, text in holds.items() if name != "until"})
    if fields["known"] is not None:
        result["known"] = reader.moments(fields["known"], "known", outcome=True)
    if fields.get("replaced") is not None:
        result["replaced"] = reader.moments(fields["replaced"], "replaced", missing="omit", outcome=True)
    if (fields["known"] is not None or fields.get("replaced") is not None) and not clocks:
        raise ValueError("declaration %s states known or replaced, which needs the model clocks" % reader.name)
    ties = fields.get("ties")
    if isinstance(ties, Held) and ties["name"] == "sequence":
        if "seq" not in ties:
            raise ValueError("declaration %s field ties states the rule sequence and lacks seq" % reader.name)
        result["seq"] = reader.moments(ties["seq"], "ties.seq")
    known = learned(result)
    for field, text in stated.items():
        if text is not None:
            reader.knowing.timely(reader, str(text), field, known, "its fact is known", EARLY)
    return result


@timed("mapping_rows.learned")
def learned(stated):
    return [(BEFORE if since is None else since) if known is None else known
            for since, known in zip(stated["since"], stated["known"])]


@timed("mapping_rows.knowable")
def knowable(reader, fields, timed_model):
    count = reader.count
    holds = fields.get("holds")
    if not timed_model or holds is None:
        return [BEFORE] * count
    since = [BEFORE] * count
    if isinstance(holds, Held):
        form = holds.get("name")
        text = holds.get("since") if form == "state" else holds.get("at") if form in ("event", "change", "snapshot") \
            else "(%s) + (%s) * (%s)" % (holds.get("start"), holds.get("index"), holds.get("stride")) \
            if form == "signal" else None
        if text is not None:
            since = [NEVER if moment is None else moment
                     for moment in reader.moments(str(text), "holds", "omit", True)]
    if fields.get("known") is None:
        return since
    return [NEVER if moment is None else moment for moment in reader.moments(fields["known"], "known", "omit", True)]



@timed("mapping_rows.times")
def times(reader, form, holds):
    count = reader.count
    stated = dict(state=("since", "until"), event=("at",), change=("at",), signal=("index", "start", "stride"))[form]
    if sorted(holds) != sorted(stated):
        raise ValueError("declaration %s field holds of the form %s states exactly %s"
                         % (reader.name, form, ", ".join(stated)))
    if form == "state":
        since = [None] * count if holds["since"] is None else reader.moments(holds["since"], "holds.since",
                                                                             outcome=True)
        until = [None] * count if holds["until"] is None else reader.moments(holds["until"], "holds.until", "omit",
                                                                             True)
        return dict(since=since, until=until)
    if form == "signal":
        start, index, stride = holds["start"], holds["index"], holds["stride"]
        since = reader.moments("(%s) + (%s) * (%s)" % (start, index, stride), "holds", outcome=True)
        until = reader.moments("(%s) + ((%s) + 1) * (%s)" % (start, index, stride), "holds", outcome=True)
        return dict(since=since, until=until)
    at = reader.moments(holds["at"], "holds.at", outcome=True)
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
