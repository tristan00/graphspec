import logging
import time
from weakref import WeakKeyDictionary

import numpy as np

from graphspec.infra.files import load_arrays, read_json
from graphspec.infra.hashing import hashed, token, unit
from graphspec.infra.timing import timed
from graphspec.schema.decisions import ROLES
from tuning.families.heads import coded

LOG = logging.getLogger(__name__)
TOKEN = (1 << 62) - 1
ORDER, TIE = "options/order", "options/key"
READ = (
    "task_key",
    "task_kind",
    "entity_token",
    "option_task",
    "option_sequence",
    "action_option",
    "action_kind",
    "action_actor",
    "action_values",
    "role_action",
    "role_name",
    "role_source",
    "role_target",
    "role_code",
)
HELD = WeakKeyDictionary()


@timed("bank.options.offered")
def offered(bank):
    if bank not in HELD:
        HELD[bank] = Offered(bank)
    return HELD[bank]


@timed("bank.options.spans")
def spans(owner, count):
    first = np.searchsorted(owner, np.arange(count))
    return first, np.searchsorted(owner, np.arange(count), "right") - first


@timed("bank.options.starts")
def starts(decision, count):
    first, number = np.full(count, -1, np.int64), np.zeros(count, np.int64)
    decided = np.flatnonzero(decision >= 0)
    held, place, many = np.unique(decision[decided], return_index=True, return_counts=True)
    first[held], number[held] = decided[place], many
    return first, number


def located(unique, graph, keys):
    started = time.perf_counter()
    low, high = np.searchsorted(unique[:, 0], [graph, graph + 1])
    place = low + np.searchsorted(unique[low:high, 1], keys)
    known = (place < high) & (unique[np.minimum(place, len(unique) - 1), 1] == keys)
    result = np.where(known, place, -1)
    LOG.debug("offered located exit graph=%d tasks=%d seconds=%.4f", graph, len(keys), time.perf_counter() - started)
    return result


class Offered:
    def __init__(self, bank):
        started = time.perf_counter()
        schema = bank.registry.schema
        actions, roles = coded(schema)
        kinds, columns = list(schema.tasks["kinds"]), len(schema.parameters())
        acting = np.full(len(kinds), -1, np.int64)
        for code, (task, _) in reversed(list(enumerate(actions))):
            acting[kinds.index(task)] = code
        roled = np.full(len(actions), -1, np.int64)
        for code, (task, action, *_) in reversed(list(enumerate(roles))):
            roled[actions.index((task, action))] = code
        self.sorts = np.asarray([ROLES.index(sort) for *_, sort, _ in roles], np.int64)
        owners = {graph["name"]: graph["index"] for graph in bank.manifest["graphs"]}
        asked = schema.tasks["key"]
        unique = np.unique(np.asarray([(owners[row["graph"]], row[asked]) for row in bank.rows], np.int64), axis=0)
        found, options, acted, keys = dict(option=[], action=[], role=[]), 0, 0, []
        for graph in bank.manifest["graphs"]:
            held = load_arrays(bank.truth(graph["index"]), READ)
            task = np.asarray(held["option_task"], np.int64)
            if not len(task):
                continue
            keys += read_json(bank.truth(graph["index"]) / "graph.json")["options"]
            tokens = np.asarray(held["entity_token"]).astype(np.uint64) & np.uint64(TOKEN)
            owner = np.asarray(held["action_option"], np.int64)
            kind = acting[np.asarray(held["task_kind"], np.int64)[task]][owner] + np.asarray(held["action_kind"], np.int64)
            role = np.asarray(held["role_action"], np.int64)
            target = np.asarray(held["role_target"], np.int64)
            found["option"].append(
                dict(
                    decision=located(unique, graph["index"], np.asarray(held["task_key"], np.int64)[task]),
                    sequence=np.asarray(held["option_sequence"], bool),
                    kind=np.asarray(held["task_kind"], np.int64)[task],
                    decided=hashed(np.full(len(task), token(graph["name"]), np.uint64), np.asarray(held["task_key"], np.int64)[task]),
                )
            )
            found["action"].append(
                dict(
                    option=owner + options,
                    kind=kind,
                    actor=tokens[np.asarray(held["action_actor"], np.int64)].astype(np.int64),
                    values=np.asarray(held["action_values"], np.float32).reshape(len(owner), columns),
                )
            )
            found["role"].append(
                dict(
                    action=role + acted,
                    code=roled[kind[role]] + np.asarray(held["role_name"], np.int64),
                    source=tokens[np.asarray(held["role_source"], np.int64)].astype(np.int64),
                    target=np.where(target >= 0, tokens[np.maximum(target, 0)].astype(np.int64), -1),
                    relation=np.asarray(held["role_code"], np.int64),
                )
            )
            options, acted = options + len(task), acted + len(owner)
        if not found["option"]:
            raise ValueError("the bank %s holds no option of a decision to realise" % bank.directory)
        self.option, self.action, self.role = (
            {name: np.concatenate([entry[name] for entry in found[part]]) for name in found[part][0]}
            for part in ("option", "action", "role")
        )
        stated = starts(self.option["decision"], len(unique))[0]
        self.actions = spans(self.action["option"], len(self.option["decision"]))
        self.roles = spans(self.role["action"], len(self.action["option"]))
        self.action["place"] = np.arange(len(self.action["option"])) - self.actions[0][self.action["option"]]
        self.option["content"] = self.contents()
        order = self.ordered(bank.seed, keys)
        inverse = np.empty(len(order), np.int64)
        inverse[order] = np.arange(len(order))
        self.option = {name: values[order] for name, values in self.option.items()}
        self.actions = (self.actions[0][order], self.actions[1][order])
        self.action["option"] = inverse[self.action["option"]]
        self.first, self.count = starts(self.option["decision"], len(unique))
        self.role["element"] = self.elements()
        decision = bank.task_of.astype(np.int64)
        aimed = np.isin([row["kind"] for row in bank.rows], schema.decided())
        self.decision = np.where(aimed & (self.first[decision] >= 0), decision, -1)
        place, part = (np.asarray(bank.fields[name], np.int64) for name in ("option", "part"))
        at = np.where(self.decision >= 0, stated[np.maximum(self.decision, 0)] + place, 0)
        self.row_option = np.where(self.decision >= 0, inverse[at], -1)
        self.row_action = np.where(
            (self.row_option >= 0) & (part >= 0), self.actions[0][np.maximum(self.row_option, 0)] + part, -1
        )
        LOG.info(
            "offered exit decisions=%d options=%d actions=%d roles=%d seconds=%.3f",
            int((self.count > 0).sum()),
            len(self.option["decision"]),
            len(self.action["option"]),
            len(self.role["action"]),
            time.perf_counter() - started,
        )

    def ordered(self, seed, keys):
        started = time.perf_counter()
        tie = unit(seed, TIE, np.asarray([token(key) for key in keys], np.uint64))
        result = np.lexsort((tie, unit(seed, ORDER, self.option["content"]), self.option["decision"]))
        LOG.debug("offered ordered exit options=%d seconds=%.4f", len(result), time.perf_counter() - started)
        return result

    def contents(self):
        started = time.perf_counter()
        created = self.sorts[self.role["code"]] == ROLES.index("new")
        source = np.where(created, -1, self.role["source"])
        role = hashed(self.role["code"], source, self.role["target"], self.role["relation"])
        summed = np.zeros(len(self.action["option"]), np.uint64)
        np.add.at(summed, self.role["action"], role)
        values = np.ascontiguousarray(self.action["values"]).view(np.uint32).astype(np.int64)
        action = hashed(self.action["kind"], summed, *values.T)
        self.action["content"] = action
        sequence = self.option["sequence"] & (self.actions[1] > 1)
        placed = np.where(sequence[self.action["option"]], hashed(action, self.action["place"]), action)
        total = np.zeros(len(sequence), np.uint64)
        np.add.at(total, self.action["option"], placed)
        result = hashed(sequence.astype(np.int64), total, self.actions[1]).view(np.int64)
        LOG.debug("offered contents exit options=%d seconds=%.4f", len(result), time.perf_counter() - started)
        return result

    def elements(self):
        started = time.perf_counter()
        option, content = self.action["option"], self.action["content"]
        order = np.lexsort((content, option))
        held = np.stack((option[order], content[order].view(np.int64)), 1)
        fresh = np.ones(len(order), bool)
        fresh[1:] = (held[1:] != held[:-1]).any(1)
        count = np.empty(len(order), np.int64)
        count[order] = np.arange(len(order)) - np.maximum.accumulate(np.where(fresh, np.arange(len(order)), 0))
        action = self.role["action"]
        owner = option[action]
        result = hashed(self.option["decided"][owner], self.option["content"][owner], self.role["code"], content[action], count[action])
        LOG.debug("offered elements exit roles=%d seconds=%.4f", len(result), time.perf_counter() - started)
        return result
