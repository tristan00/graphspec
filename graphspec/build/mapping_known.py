import ast
import logging
import math
import time

from graphspec.infra.timing import timed

from .table_types import Context, broadcast

LOG = logging.getLogger(__name__)
NEVER = math.inf
BEFORE = -math.inf
LAST = 2 ** 63


@timed("mapping_known.shown")
def shown(moment):
    return "never" if moment == NEVER else "before every time" if moment == BEFORE else str(moment)


@timed("mapping_known.read")
def read(text):
    tree = ast.parse(text, mode="eval")
    plain, scoped, met, waiting = [], [], [], [tree.body]
    while waiting:
        node = waiting.pop()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "met" and len(node.args) == 2 and not node.keywords:
                met.append(node)
                continue
            if node.func.id == "column" and len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
                plain.append(node.args[0].value)
                continue
            waiting.extend([*node.args, *(keyword.value for keyword in node.keywords)])
            continue
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            scoped.append((node.value.id, node.attr))
            continue
        if isinstance(node, ast.Name):
            if node.id not in ("missing", "never"):
                plain.append(node.id)
            continue
        waiting.extend(ast.iter_child_nodes(node))
    return list(dict.fromkeys(plain)), list(dict.fromkeys(scoped)), met


@timed("mapping_known.latest")
def latest(columns, count):
    found = [values for values in columns if values is not None]
    if not found:
        return None
    return [max(values[place] for values in found) for place in range(count)]


class Knowing:
    @timed("mapping_known.Knowing")
    def __init__(self, session, plan):
        self.session, self.plan, self.held, self.ending = session, plan, {}, frozenset()

    @timed("mapping_known.Knowing.frames")
    def frames(self):
        return {self.session.names[identity]: frame for identity, frame in self.session.frames.items()}

    @timed("mapping_known.Knowing.marked")
    def marked(self, mark):
        if mark in self.held:
            return self.held[mark]
        name, text = mark
        frame = self.frames()[name]
        if text == "never":
            self.held[mark] = [NEVER] * frame.table.num_rows
            return self.held[mark]
        context = Context(name, self.plan, self.session.unit, self.session.retained())
        value = context.evaluate(frame, text, "outcome")
        if value.kind.name not in ("time", "number", "missing"):
            raise ValueError("declaration %s states an outcome known from %s, which gives no time" % (name, text))
        found = []
        for place, moment in enumerate(broadcast(value, frame.table.num_rows).to_pylist()):
            if moment is None or moment >= LAST:
                found.append(NEVER)
            elif float(moment).is_integer():
                found.append(int(moment))
            else:
                raise ValueError("declaration %s row %d states an outcome known from %r, not a whole time"
                                 % (name, place, moment))
        self.held[mark] = found
        return found

    @timed("mapping_known.Knowing.rows")
    def rows(self, frame, mark):
        values = self.marked(mark)
        parents = frame.provenance.get(mark[0])
        if parents is None:
            raise ValueError("the outcome of %s is read through a table that keeps none of its rows" % mark[0])
        return [max((values[place] for place in group), default=BEFORE) for group in parents.to_pylist()]

    @timed("mapping_known.Knowing.column")
    def column(self, frame, name):
        marks = frame.outcomes.get(name, ())
        if not marks:
            return None
        return latest([self.rows(frame, mark) for mark in marks], frame.table.num_rows)

    @timed("mapping_known.Knowing.named")
    def named(self, frame, name):
        frames, found = self.frames(), []
        for mark in frame.outcomes.get(name, ()):
            carriers = [column for column, marks in frames[mark[0]].outcomes.items() if mark in marks]
            origins = [column for column in carriers if (mark[0], column) in frame.lineage.get(name, ())]
            found += ["%s.%s" % (mark[0], column) for column in origins or carriers]
        return list(dict.fromkeys(found))

    @timed("mapping_known.Knowing.never")
    def never(self, frame, name):
        return any(mark[1] == "never" for mark in frame.outcomes.get(name, ()))

    @timed("mapping_known.Knowing.outcomes")
    def outcomes(self, reader, text):
        plain, scoped, met = read(text)
        found = [name for column in plain if column in reader.frame.outcomes for name in self.named(reader.frame, column)]
        for scope, column in scoped:
            window = reader.windows.get(scope)
            if window is not None and column in window.frame.outcomes:
                found += self.named(window.frame, column)
        for node in met:
            found += [name for part in node.args for name in self.outcomes(reader, ast.unparse(part))]
        return list(dict.fromkeys(found))

    @timed("mapping_known.Knowing.ended")
    def ended(self, reader, text, field):
        for column in read(text)[0]:
            origins = set(reader.frame.lineage.get(column, ())) & self.ending
            if origins and column not in reader.frame.outcomes:
                raise ValueError("declaration %s field %s reads %s, the column until of a state reads (%s); the end "
                                 "of a state is read only as an outcome, stated with the time it is known from"
                                 % (reader.name, field, column, ", ".join(sorted("%s.%s" % origin for origin in origins))))

    @timed("mapping_known.Knowing.expression")
    def expression(self, reader, text, field):
        started = time.perf_counter()
        plain, scoped, met = read(text)
        count = reader.count
        columns = [self.column(reader.frame, column) for column in plain if column in reader.frame.kinds]
        for scope, column in scoped:
            window = reader.windows.get(scope)
            if window is None or column not in window.frame.kinds:
                continue
            values = self.column(window.frame, column)
            if values is not None:
                columns.append([max((values[place] for place in window.places[row]), default=BEFORE)
                                for row in range(count)])
        for node in met:
            happened = self.expression(reader, ast.unparse(node.args[0]), field)
            deadline = reader.moments(ast.unparse(node.args[1]), field, "omit", True)
            late = self.expression(reader, ast.unparse(node.args[1]), field)
            if happened is None:
                columns.append(late)
                continue
            columns.append([min(happened[place], max(NEVER if deadline[place] is None else deadline[place],
                                                     BEFORE if late is None else late[place]))
                            for place in range(count)])
        result = latest(columns, count)
        LOG.debug("expression exit declaration=%s field=%s rows=%d seconds=%.6f", reader.name, field, count,
                  time.perf_counter() - started)
        return result

    @timed("mapping_known.Knowing.timely")
    def timely(self, reader, text, field, known, what, where):
        found = self.expression(reader, text, field)
        if found is None:
            return
        for place, (moment, fact) in enumerate(zip(found, known)):
            if moment > fact:
                raise ValueError("declaration %s field %s reads the outcome column %s, known from %s in row %d, later "
                                 "than %s, from %s; %s"
                                 % (reader.name, field, ", ".join(self.outcomes(reader, text)), shown(moment),
                                    reader.row(place), what, shown(fact), where))
