import logging
from collections import namedtuple

from graphspec.infra.timing import timed

from .readers import Source
from .tables import Table

LOG = logging.getLogger(__name__)
Error = namedtuple("Error", ("declaration", "places", "message"))


@timed("invalid.placed")
def placed(value):
    if isinstance(value, Source):
        return [str(value.place)]
    if isinstance(value, Table):
        return list(dict.fromkeys(place for parent in value.inputs for place in placed(parent)))
    return []


@timed("invalid.text")
def text(errors, unchecked):
    held = {}
    for error in errors:
        held.setdefault(error.declaration, []).append(error)
    lines = ["the build holds %d %s" % (len(errors), "error" if len(errors) == 1 else "errors")]
    for declaration, found in held.items():
        places = list(dict.fromkeys(place for error in found for place in error.places))
        lines.append("%s%s" % ("the script" if declaration is None else "declaration " + declaration,
                               ", reading " + ", ".join(places) if places else ""))
        lines += ["  " + error.message for error in found]
    if unchecked:
        lines.append("not checked, as they read a declaration with an error: " + ", ".join(unchecked))
    return "\n".join(lines)


class Invalid(ValueError):
    @timed("invalid.Invalid")
    def __init__(self, errors, unchecked=()):
        super().__init__(tuple(errors), tuple(unchecked))
        self.errors, self.unchecked = self.args

    @timed("invalid.Invalid.str")
    def __str__(self):
        return text(self.errors, self.unchecked)


class Errors:
    @timed("invalid.Errors")
    def __init__(self, order):
        self.found, self.unchecked, self.failed, self.order = [], [], set(), list(order)

    @timed("invalid.Errors.rank")
    def rank(self, declaration):
        if declaration is None:
            return -1
        return self.order.index(declaration) if declaration in self.order else len(self.order)

    @timed("invalid.Errors.ranked")
    def ranked(self, error):
        return self.rank(error.declaration)

    @timed("invalid.Errors.add")
    def add(self, declaration, value, places, error):
        self.failed.add(id(value))
        self.found.append(Error(declaration, tuple(places), str(error)))

    @timed("invalid.Errors.skip")
    def skip(self, declaration, value):
        self.failed.add(id(value))
        if declaration not in self.unchecked and all(error.declaration != declaration for error in self.found):
            self.unchecked.append(declaration)

    @timed("invalid.Errors.checked")
    def checked(self, label, value, needed, places, work, *arguments):
        if id(value) in self.failed:
            return
        if any(id(found) in self.failed for found in needed):
            self.skip(label, value)
            return
        try:
            work(*arguments)
        except ValueError as error:
            self.add(label, value, places, error)

    @timed("invalid.Errors.read")
    def read(self, session, value, label, labels):
        inputs = value.inputs if isinstance(value, Table) else ()
        for parent in inputs:
            self.read(session, parent, labels.get(id(parent), label), labels)
        self.checked(label, value, inputs, placed(value), session.get, value)

    @timed("invalid.Errors.raised")
    def raised(self):
        if self.found:
            LOG.info("invalid build errors=%d unchecked=%d", len(self.found), len(self.unchecked))
            raise Invalid(sorted(self.found, key=self.ranked), sorted(self.unchecked, key=self.rank))
