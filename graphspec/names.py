import logging
import time

LOG = logging.getLogger(__name__)


class Names:
    def __init__(self, label):
        self.label, self.held = label, {}

    def register(self, name, entry):
        started = time.perf_counter()
        if type(name) is not str or not name:
            raise ValueError("a %s is registered under a name, found %r" % (self.label, name))
        if name in self.held:
            raise ValueError("the %s %s is registered twice" % (self.label, name))
        self.held[name] = entry
        LOG.debug(
            "register exit label=%s name=%s seconds=%.4f", self.label, name, time.perf_counter() - started
        )
        return entry

    def named(self, name):
        if type(name) is not str or name not in self.held:
            raise KeyError("no %s is registered under the name %r" % (self.label, name))
        return self.held[name]

    def names(self):
        return tuple(self.held)
