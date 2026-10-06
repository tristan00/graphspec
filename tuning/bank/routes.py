import logging
import time

from graphspec.budgets import Refused
from graphspec.operations.pipeline import emptied, unrelated
from tuning.families.heads import fielded, realisation, refused
from tuning.families.model_spec import DIRECTION, SHOWN, UNREAD, ignored, realised, sending, showing

LOG = logging.getLogger(__name__)


def cut(bank, entry, task, field, removed, unkept, settings, family):
    started = time.perf_counter()
    directed = DIRECTION in settings and settings[DIRECTION] == "directed"
    shown = SHOWN[family] if family in SHOWN else None
    kinds, relations, held, result = field["named"], field["relations"], None, None
    if entry.back:
        schema = bank.registry.schema
        held = entry.mapping[task][field["action"]][field["part"]]
        relations = (held["relation"],) if held["relation"] in schema.edges else ()
        kinds += tuple(schema.tasks["references"][schema.tasks["kinds"][task]["target"]["actors"]]["kinds"])
    elif field["role"] == "new":
        kinds = ()
    lost = [kind for kind in kinds if kind in removed]
    gone = [relation for relation in relations if relation in unkept]
    if lost:
        result = "the pipeline removes the kind of node %s, which it may name, the entry %s removing it" % (
            lost[0],
            removed[lost[0]],
        )
    elif gone:
        step, parameter, value = unkept[gone[0]]
        result = "the pipeline removes the relation %s, which it may name, by %s %s of the entry %s" % (
            gone[0],
            parameter,
            value,
            step,
        )
    elif held is not None and directed and sending(bank.registry, entry, held):
        result = (
            "an edit makes it facts of the relation %s, which has a direction, from the actor, and the setting %s "
            "directed reads a relation from its source to its target alone, so the actor reads none of them"
            % (held["relation"], DIRECTION)
        )
    elif held is not None and shown is not None and not showing(settings[shown]):
        result = "an edit makes it facts of the relation %s, and the setting %s %s shows the family %s no relation" % (
            held["relation"],
            shown,
            settings[shown],
            family,
        )
    LOG.debug("routes cut exit field=%s cut=%s seconds=%.4f", field["name"], result is not None, time.perf_counter() - started)
    return result


def routed(bank, cfg):
    started = time.perf_counter()
    _, steps, settings, _ = bank.stated(cfg)
    name = realised(cfg["model"], bank.registry.schema)
    if name is None:
        LOG.debug("routes routed exit realisation=none seconds=%.4f", time.perf_counter() - started)
        return
    entry, spec, schema = realisation(name), cfg["model"], bank.registry.schema
    reason = ignored(spec, name, schema)
    if reason is not None:
        raise Refused(UNREAD % (name, spec["view"], reason))
    reason = refused(name, bank.registry)
    if reason is not None:
        raise Refused(reason)
    removed, unkept = emptied(bank.registry, steps), unrelated(bank.registry, steps)
    for task in schema.decided():
        for field in fielded(schema, task):
            where = "the field %s of the kind of task %s has no route to the prediction under the realisation of options %s"
            where %= (field["name"], task, name)
            reason = cut(bank, entry, task, field, removed, unkept, settings, spec["view"]) if field["sort"] == "targets" else None
            if reason is not None:
                raise Refused("%s: %s" % (where, reason))
    LOG.info("routes routed exit realisation=%s seconds=%.4f", name, time.perf_counter() - started)
