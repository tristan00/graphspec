import logging
import time

from graphspec.registries import Names

LOG = logging.getLogger(__name__)
FAMILIES = ("structure", "sample", "encode")


class Representation:
    def __init__(self, name, family, carries, horizon, piece, ended, stitched, check):
        started = time.perf_counter()
        if type(name) is not str or not name or "/" in name:
            raise ValueError("a representation states its name as a word without a slash, found %r" % (name,))
        if family not in FAMILIES:
            raise ValueError(
                "the representation %s states its family as one of %s, found %r"
                % (name, ", ".join(FAMILIES), family)
            )
        stated = dict(
            carries=callable(carries),
            horizon=callable(horizon),
            piece=callable(piece),
            ended=ended is None or callable(ended),
            stitched=stitched is None or callable(stitched),
            check=callable(check),
        )
        wrong = [label for label, held in stated.items() if not held]
        if wrong:
            raise ValueError(
                "the representation %s states %s as the constructor of a representation asks: a function from a "
                "registry and settings to the state it carries into its first segment; a function from settings "
                "to the steps and the groups of periods of history it reads; a function of a segment and its "
                "state to the arrays of the segment; none, or a function of its state to the rows it adds after "
                "the last segment; none, or a function of the arrays of every segment, the registry, settings "
                "and limits to arrays over them all; a function that checks its arrays"
                % (name, ", ".join(wrong))
            )
        self.name, self.family, self.carries, self.horizon, self.piece = name, family, carries, horizon, piece
        self.ended, self.stitched, self.check = ended, stitched, check
        LOG.debug("representation exit name=%s family=%s seconds=%.4f", name, family, time.perf_counter() - started)


class Representations(Names):
    def register(self, name, entry):
        started = time.perf_counter()
        if not isinstance(entry, Representation) or entry.name != name:
            raise ValueError(
                "a representation is registered under its own name; %r is registered as %s"
                % (getattr(entry, "name", entry), name)
            )
        result = super().register(name, entry)
        LOG.debug("representations register exit name=%s seconds=%.4f", name, time.perf_counter() - started)
        return result


REPRESENTATIONS = Representations("representation")


def chosen(names):
    started = time.perf_counter()
    result = tuple(REPRESENTATIONS.named(name) for name in names)
    LOG.debug("chosen exit representations=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result


def families():
    started = time.perf_counter()
    result = {
        family: tuple(name for name in REPRESENTATIONS.names() if REPRESENTATIONS.named(name).family == family)
        for family in FAMILIES
    }
    LOG.debug("families exit families=%d seconds=%.4f", len(result), time.perf_counter() - started)
    return result
