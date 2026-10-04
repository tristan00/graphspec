import logging
import time
from importlib import import_module

import torch
from torch import nn

from graphspec.infra.timing import timed

from .heads import ONE, coded, realisation
from .layers import mean, padded

LOG = logging.getLogger(__name__)
STRUCTURE = ("sequence", "decision", "action_option", "action_place", "role_action", "role_code")


def slots(encoding, columns, name):
    started = time.perf_counter()
    sizes = [encoding.sizes[column] for column in columns]
    values = sum(size[0] for size in sizes)
    place = columns.index(name)
    start, flag = sum(size[0] for size in sizes[:place]), values + sum(size[1] for size in sizes[:place])
    result = list(range(start, start + sizes[place][0])) + list(range(flag, flag + sizes[place][1]))
    LOG.debug("options slots exit column=%s slots=%d seconds=%.6f", name, len(result), time.perf_counter() - started)
    return result


class Fields(nn.Module):
    @timed("options.Fields.__init__")
    def __init__(self, registry, hidden, nodes, reads, actor):
        super().__init__()
        actions, roles = coded(registry.schema)
        self.reads = reads
        if "kind" in reads:
            self.kind = nn.Embedding(len(actions), hidden)
        if "targets" in reads:
            self.role = nn.Embedding(len(roles) + actor, hidden)
            self.relation = nn.Embedding(registry.relation_slots, hidden)
            self.node = nn.Sequential(nn.Linear(nodes, hidden), nn.SiLU())

    def forward(self, b, members):
        held, found = b["options"], {}
        if "kind" in self.reads:
            found["kind"] = self.kind(held["action_kind"])
        if "targets" in self.reads:
            ends, relation = held["role_member"], held["role_relation"]
            node = padded(self.node(members), ends).sum(1) / (ends >= 0).sum(1, keepdim=True).clamp_min(1)
            related = self.relation(relation.clamp_min(0)) * (relation >= 0)[:, None]
            found["targets"] = node + self.role(held["role_code"]) + related
        if "parameters" in self.reads:
            found["parameters"] = held["action_values"]
        return found


class Node(nn.Module):
    @timed("options.Node.__init__")
    def __init__(self, registry):
        super().__init__()
        self.roles = coded(registry.schema)[1]

    def forward(self, fields, structure):
        owner, options = structure["action_option"], len(structure["sequence"])
        if len(owner) != options:
            count = torch.bincount(owner, minlength=options)
            place = int(torch.nonzero(count != 1)[0])
            codes = structure["role_code"][torch.isin(structure["role_action"], torch.nonzero(owner == place).flatten())]
            named = [self.roles[code] for code in codes.tolist()]
            raise ValueError(
                "an option of the kind of task %s holds %d actions, of the kinds %s, and %s"
                % (named[0][0], int(count[place]), ", ".join(role[1] for role in named), ONE)
            )
        return fields["targets"], fields["targets"]


class Vector(nn.Module):
    @timed("options.Vector.__init__")
    def __init__(self, hidden, width):
        super().__init__()
        self.bound = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())
        self.place = nn.Linear(1, hidden, bias=False)
        self.action = nn.Sequential(nn.Linear(hidden + width, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.empty = nn.Parameter(torch.zeros(hidden))

    def forward(self, fields, structure):
        kind, owner = fields["kind"], structure["action_option"]
        targets = kind.new_zeros(kind.shape).index_add(0, structure["role_action"], self.bound(fields["targets"]))
        place = self.place(torch.log1p(structure["action_place"].float())[:, None])
        placed = place * structure["sequence"][owner][:, None]
        actions = self.action(torch.cat((kind + targets + placed, fields["parameters"]), -1))
        count = actions.new_zeros(len(structure["sequence"])).index_add(0, owner, actions.new_ones(len(owner)))
        total = actions.new_zeros((len(count), actions.shape[1])).index_add(0, owner, actions)
        options = total / count.clamp_min(1)[:, None] + self.empty * (count == 0)[:, None]
        return options, actions


@timed("options.received")
def received(spec, hidden, labels):
    from .model_spec import depth, unstructured

    reason = unstructured(spec)
    if reason is not None:
        raise ValueError("the family %s reads no option put into its graph: %s" % (spec["view"], reason))
    rounds = depth(spec)
    return import_module("." + spec["view"], __package__).Received(spec, hidden, labels, rounds) if rounds else None


class Structured(nn.Module):
    @timed("options.Structured.__init__")
    def __init__(self, hidden, width, spec, registry, element, actor):
        super().__init__()
        self.slot, self.element = len(coded(registry.schema)[1]), element
        self.place = nn.Linear(1, hidden, bias=False)
        self.action = nn.Sequential(nn.Linear(hidden + width, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.empty = nn.Parameter(torch.zeros(hidden))
        self.received = received(spec, hidden, self.slot + actor)

    def forward(self, fields, structure):
        kind, owner = fields["kind"], structure["action_option"]
        code, acting = structure["role_code"], structure["role_action"]
        place = self.place(torch.log1p(structure["action_place"].float())[:, None])
        actions = self.action(torch.cat((kind + place * structure["sequence"][owner][:, None], fields["parameters"]), -1))
        if self.element == "action":
            source, sink, sinks = torch.arange(len(code), device=code.device), acting, actions
        else:
            target = torch.nonzero(code != self.slot).flatten()
            actor = torch.nonzero(code == self.slot).flatten()
            held = torch.full((len(actions),), -1, dtype=torch.long, device=code.device).index_put((acting[actor],), actor)
            each = torch.arange(len(target), device=code.device)
            source, sink, sinks = torch.cat((target, held[acting[target]])), torch.cat((each, each)), actions[acting[target]]
        if self.received is None:
            found = sinks + mean(fields["targets"][source], sink, len(sinks))
        else:
            found = self.received(sinks, fields["targets"], (source, sink), code[source])
        acted = found if self.element == "action" else mean(found, acting[target], len(actions))
        count = acted.new_zeros(len(structure["sequence"])).index_add(0, owner, acted.new_ones(len(owner)))
        total = acted.new_zeros((len(count), acted.shape[1])).index_add(0, owner, acted)
        options = total / count.clamp_min(1)[:, None] + self.empty * (count == 0)[:, None]
        return options, acted


class Options(nn.Module):
    @timed("options.Options.__init__")
    def __init__(self, name, registry, hidden, nodes, spec):
        super().__init__()
        entry = realisation(name)
        self.name, self.hidden = name, hidden
        self.fields = Fields(registry, hidden, nodes, entry.reads, entry.actor)
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
