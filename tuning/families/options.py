import logging
import time

import torch
from torch import nn

from graphspec.infra.timing import timed

from .heads import REALISATIONS
from .layers import padded

LOG = logging.getLogger(__name__)
FIELDS = ("kind", "targets", "parameters")
STRUCTURE = ("sequence", "decision", "action_option", "action_place", "role_action", "role_code")


def realisation(name):
    started = time.perf_counter()
    entry = REALISATIONS.named(name)
    if (
        not all(hasattr(entry, part) for part in ("reads", "together", "back", "module"))
        or not isinstance(entry.reads, tuple)
        or not entry.reads
        or any(field not in FIELDS for field in entry.reads)
        or len(set(entry.reads)) != len(entry.reads)
        or type(entry.together) is not bool
        or type(entry.back) is not bool
        or not callable(entry.module)
    ):
        raise ValueError(
            "the realisation of options %s states reads, a tuple of the fields of an option it reads among %s; "
            "together, whether the score of an option reads the other options of its decision, true or false; "
            "back, whether messages pass from options back to the state, true or false; and module, which builds "
            "it" % (name, ", ".join(FIELDS))
        )
    LOG.debug("options realisation exit name=%s seconds=%.4f", name, time.perf_counter() - started)
    return entry


def coded(schema):
    started = time.perf_counter()
    actions, roles = [], []
    for task in schema.decided():
        for action, entry in schema.tasks["kinds"][task]["target"]["actions"].items():
            actions.append((task, action))
            roles += [(task, action, role, held["sort"], tuple(held["kinds"])) for role, held in entry["roles"].items()]
    LOG.debug("options coded exit actions=%d roles=%d seconds=%.4f", len(actions), len(roles), time.perf_counter() - started)
    return actions, roles


def slots(encoding, columns, name):
    started = time.perf_counter()
    sizes = [encoding.sizes[column] for column in columns]
    values = sum(size[0] for size in sizes)
    place = columns.index(name)
    start, flag = sum(size[0] for size in sizes[:place]), values + sum(size[1] for size in sizes[:place])
    result = list(range(start, start + sizes[place][0])) + list(range(flag, flag + sizes[place][1]))
    LOG.debug("options slots exit column=%s slots=%d seconds=%.6f", name, len(result), time.perf_counter() - started)
    return result


def fielded(schema, task):
    started = time.perf_counter()
    actions, roles = coded(schema)
    result = [dict(name="kind", sort="kind", code=None)]
    for code, (owner, action, role, sort, kinds) in enumerate(roles):
        if owner == task:
            result.append(
                dict(name="target %s of %s" % (role, action), sort="targets", code=code, role=sort, kinds=kinds)
            )
    for code, (owner, action) in enumerate(actions):
        if owner == task:
            for column in schema.tasks["kinds"][task]["target"]["actions"][action]["parameters"]:
                result.append(dict(name="parameter %s of %s" % (column, action), sort="parameters", code=code, column=column))
    LOG.debug("options fielded exit task=%s fields=%d seconds=%.4f", task, len(result), time.perf_counter() - started)
    return result


class Fields(nn.Module):
    @timed("options.Fields.__init__")
    def __init__(self, registry, hidden, nodes):
        super().__init__()
        actions, roles = coded(registry.schema)
        self.kind = nn.Embedding(len(actions), hidden)
        self.role = nn.Embedding(len(roles), hidden)
        self.relation = nn.Embedding(registry.relation_slots, hidden)
        self.node = nn.Sequential(nn.Linear(nodes, hidden), nn.SiLU())

    def forward(self, b, members):
        held = b["options"]
        ends = held["role_member"]
        node = padded(self.node(members), ends).sum(1) / (ends >= 0).sum(1, keepdim=True).clamp_min(1)
        relation = held["role_relation"]
        targets = node + self.role(held["role_code"]) + self.relation(relation.clamp_min(0)) * (relation >= 0)[:, None]
        return dict(kind=self.kind(held["action_kind"]), targets=targets, parameters=held["action_values"])


class Options(nn.Module):
    @timed("options.Options.__init__")
    def __init__(self, name, registry, hidden, nodes, spec):
        super().__init__()
        entry = realisation(name)
        self.name, self.hidden = name, hidden
        self.fields = Fields(registry, hidden, nodes)
        width = registry.encoding.width(registry.schema.parameters())
        self.realised = entry.module(hidden, width, spec, registry)

    def encoded(self, fields, b):
        held = b["options"]
        structure = {key: held[key] for key in STRUCTURE}
        options, actions = self.realised(fields, structure)
        wanted = ((len(held["sequence"]), self.hidden), (len(held["action_option"]), self.hidden))
        if (tuple(options.shape), tuple(actions.shape)) != wanted:
            raise ValueError(
                "the realisation of options %s gives encodings of the shapes %s and %s, and the batch holds %d "
                "options and %d actions, each encoded in %d values"
                % (self.name, tuple(options.shape), tuple(actions.shape), wanted[0][0], wanted[1][0], self.hidden)
            )
        return options, actions

    def forward(self, b, members):
        held = b["options"]
        options, actions = self.encoded(self.fields(b, members), b)
        action = held["point_action"]
        index = torch.where(action >= 0, action + len(options), held["point_option"])
        return padded(torch.cat((options, actions)), index)
