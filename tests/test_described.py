import ast
import logging
import time
from pathlib import Path

import pytest

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("described")
ROOT = Path(__file__).parent / "described"
LANGUAGE = "graphspec"
DESCRIBED = "DESCRIPTION"
SCRIPTED = "scripted"
RECORD = "dict"
SOURCES = ("source", "database")
OPERATIONS = (
    "select",
    "compute",
    "join",
    "group",
    "pivot",
    "unpivot",
    "expand",
    "nested",
    "cut",
    "align",
    "union",
    "order",
    "closure",
)
MAPPINGS = ("graphs", "nodes", "edges", "sets", "tasks", "decisions")
WORDS = SOURCES + OPERATIONS + MAPPINGS
TARGETED = ("tasks", "decisions")
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
PLAIN = (
    ast.Assign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Call,
    ast.keyword,
    ast.Dict,
    ast.List,
    ast.Tuple,
    ast.Attribute,
)
ROWS = {
    "01_sensors.py": (
        "Sensors joined by a kernel of road distance above a threshold; a reading every five minutes; "
        "predict the next twelve",
        10,
        dict(sources=2, declarations=6, functions=0),
    ),
    "02_interactions.py": (
        "Interactions of users with items over time; predict the next item among sampled negatives",
        7,
        dict(sources=1, declarations=4, functions=0),
    ),
    "03_transfers.py": (
        "Accounts and transfers as parallel edges with amounts; the label of a transfer arrives thirty days "
        "later and is then known to later tasks",
        8,
        dict(sources=1, declarations=4, functions=0),
    ),
    "04_papers.py": (
        "Papers citing papers; a paper sees only older papers; predict the venue",
        6,
        dict(sources=2, declarations=4, functions=0),
    ),
    "05_triples.py": (
        "Triples of head, relation and tail with thousands of relations, some about other triples; "
        "predict the tail",
        5,
        dict(sources=1, declarations=4, functions=0),
    ),
    "06_molecules.py": (
        "Many small graphs in one file with a column naming the graph, nodes identified by row number; "
        "a target per graph",
        6,
        dict(sources=3, declarations=3, functions=0),
    ),
    "07_events.py": (
        "Events that each touch several objects of several kinds, each with a role",
        8,
        dict(sources=3, declarations=4, functions=0),
    ),
    "08_visits.py": (
        "Visits in order per patient, each with sets of codes; codes under a table of parents, "
        "all ancestors joined",
        12,
        dict(sources=3, declarations=8, functions=0),
    ),
    "09_trees.py": (
        "Trees of nested records with ordered children, and two more sets of edges over the same nodes",
        9,
        dict(sources=3, declarations=6, functions=0),
    ),
    "10_coauthors.py": (
        "Papers joining all their authors as one edge; predict the next set",
        6,
        dict(sources=1, declarations=4, functions=0),
    ),
    "11_game.py": (
        "A game where several players act at once and each sees part of the state",
        12,
        dict(sources=2, declarations=4, functions=0),
    ),
    "12_clicks.py": (
        "Clicks cut into sessions by a gap of thirty minutes, with edges between consecutive clicks",
        6,
        dict(sources=1, declarations=4, functions=0),
    ),
    "13_trips.py": (
        "Trips as ordered sequences of road segments; predict the duration of a trip",
        8,
        dict(sources=2, declarations=3, functions=0),
    ),
    "14_machines.py": (
        "Machines with logs of sensors; predict the days until failure, where most had not failed "
        "when recording ended",
        8,
        dict(sources=2, declarations=3, functions=0),
    ),
    "15_drugs.py": (
        "Drugs that are each a graph of atoms; predict which pairs interact",
        8,
        dict(sources=3, declarations=4, functions=0),
    ),
    "16_figures.py": (
        "Quarterly figures that are revised later; predict with what was believed on the day",
        6,
        dict(sources=1, declarations=5, functions=0),
    ),
    "17_impressions.py": (
        "Impressions in which a policy showed items in three positions, each with the probability of being "
        "shown; the actor is the policy, the outcome is the click",
        7,
        dict(sources=1, declarations=3, functions=0),
    ),
    "18_dispatches.py": (
        "Decisions where only the chosen set of actions was logged, each action with targets in the graph "
        "and a number as parameter; the other options are drawn by a function",
        9,
        dict(sources=3, declarations=3, functions=1),
    ),
    "19_database.py": (
        "A database with declared keys and views: a target, the columns of time, and who is eligible",
        6,
        dict(sources=1, declarations=3, functions=0),
    ),
}
REASONS = {
    "18_dispatches.py": {
        "rerouted": (
            "the description states that the other options are drawn by a function; no shipped generator "
            "replaces a target of a chosen action by another node its role may name"
        ),
    },
}


def opened(tree, where):
    started = time.perf_counter()
    body = list(tree.body)
    imports = [node for node in body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert body[: len(imports)] == imports, "%s imports after its first statement" % where
    words, foreign = [], []
    for node in imports:
        assert all(alias.asname is None for alias in node.names), "%s renames what it imports" % where
        if isinstance(node, ast.ImportFrom) and node.module == LANGUAGE and node.level == 0:
            words += [alias.name for alias in node.names]
        else:
            named = [node.module or ""] if isinstance(node, ast.ImportFrom) else [alias.name for alias in node.names]
            assert not any(name.partition(".")[0] in (LANGUAGE, "tuning") for name in named), (
                "%s imports the library other than by its words" % where
            )
            foreign += [alias.name.partition(".")[0] for alias in node.names]
    assert sorted(set(words)) == sorted(words), "%s imports a word twice" % where
    assert set(words) <= set(WORDS) | {SCRIPTED}, "%s imports a name that is no word of the language" % where
    rest = body[len(imports) :]
    stated = rest[0] if rest else None
    assert (
        isinstance(stated, ast.Assign)
        and len(stated.targets) == 1
        and isinstance(stated.targets[0], ast.Name)
        and stated.targets[0].id == DESCRIBED
        and isinstance(stated.value, ast.Constant)
        and isinstance(stated.value.value, str)
    ), "%s opens with its description under the name %s" % (where, DESCRIBED)
    LOG.debug("opened exit script=%s words=%d seconds=%.6f", where, len(words), time.perf_counter() - started)
    return words, foreign, stated.value.value, rest[1:]


def refused(statements, words, foreign, where):
    started = time.perf_counter()
    found, waiting = [], list(statements)
    while waiting:
        node = waiting.pop()
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.FunctionDef):
            waiting.extend(node.decorator_list)
            continue
        waiting.extend(ast.iter_child_nodes(node))
        if isinstance(node, ast.USub):
            continue
        if isinstance(node, ast.UnaryOp):
            if not (isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant)):
                found.append((where, line, "an operator"))
        elif isinstance(node, ast.Subscript):
            if not (isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
                found.append((where, line, "a subscript that is no name"))
        elif not isinstance(node, PLAIN):
            found.append((where, line, type(node).__name__))
        elif isinstance(node, ast.Dict) and None in node.keys:
            found.append((where, line, "a mapping unpacked"))
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name not in words and name not in (RECORD, SCRIPTED):
                found.append((where, line, "a call that is neither a word of the language nor a record"))
            elif not all(key.arg for key in node.keywords) or (name == RECORD and node.args):
                found.append((where, line, "arguments unpacked, or a record not made of names"))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in foreign:
            found.append((where, line, "a name imported from outside the language"))
    LOG.debug("refused exit script=%s found=%d seconds=%.6f", where, len(found), time.perf_counter() - started)
    return found


def read(name):
    started = time.perf_counter()
    tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
    words, foreign, description, statements = opened(tree, name)
    every = list(ast.walk(tree))
    called = [node for node in every if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    defined = [node for node in statements if isinstance(node, ast.FunctionDef)]
    listed = {}
    for node in defined:
        marks = node.decorator_list
        assert len(marks) == 1 and marks[0] in called and marks[0].func.id == SCRIPTED, (
            "%s line %d is a function the language does not register" % (name, node.lineno)
        )
        reasons = [
            key.value.value
            for key in marks[0].keywords
            if key.arg == "reason" and isinstance(key.value, ast.Constant) and isinstance(key.value.value, str)
        ]
        assert len(reasons) == 1 and reasons[0], "%s line %d states no reason" % (name, node.lineno)
        listed[node.name] = reasons[0]
    language = set(WORDS) | {SCRIPTED}
    worded = set(WORDS) | set(listed)
    bound = []
    for node in statements:
        if node in defined:
            continue
        assert (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.value in called
            and node.value.func.id in worded
        ), "%s line %d is no one word of the language bound to one name" % (name, node.lineno)
        bound.append(node.targets[0].id)
    assert len(set(bound)) == len(bound) and (worded | {SCRIPTED, RECORD, DESCRIBED}).isdisjoint(bound), (
        "%s binds a name twice or binds a word of the language" % name
    )
    loaded = [node for node in every if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)]
    assert sum(node.id in language for node in loaded) == sum(node.func.id in language for node in called), (
        "%s names a word of the language without calling it" % name
    )
    assert sum(node.func.id == SCRIPTED for node in called) == len(defined), (
        "%s registers other than a top-level function of its own" % name
    )
    functions = [node for node in every if isinstance(node, FUNCTIONS)]
    assert sorted(getattr(node, "name", "lambda") for node in functions) == sorted(listed), (
        "%s holds a function it does not register with its reason" % name
    )
    calls = [node.func.id for node in called if node.func.id in worded]
    result = dict(
        description=description,
        statements=len(statements),
        counts=dict(
            sources=sum(call in SOURCES for call in calls),
            declarations=sum(call not in SOURCES for call in calls),
            functions=len(functions),
        ),
        words=sorted(set(calls) & set(WORDS)),
        imported=sorted(set(words) - {SCRIPTED}),
        functions=listed,
        refused=refused(statements, worded, foreign, name),
    )
    LOG.debug("read exit script=%s statements=%d seconds=%.6f", name, len(statements), time.perf_counter() - started)
    return result


def test_every_described_script_is_declarations_alone_counted_within_its_ceiling_with_its_functions_listed():
    started = time.perf_counter()
    assert sorted(path.name for path in ROOT.iterdir()) == sorted(ROWS)
    assert len(ROWS) == 19 and len({description for description, _, _ in ROWS.values()}) == 19
    for name, (description, ceiling, expected) in ROWS.items():
        found = read(name)
        assert found["description"] == description, name
        assert found["refused"] == [], name
        assert found["counts"] == expected, name
        assert sum(found["counts"].values()) == found["statements"] <= ceiling, name
        assert found["functions"] == REASONS.get(name, {}), name
        assert found["words"] == found["imported"], name
        assert set(found["words"]) & set(TARGETED), name
    assert sorted(REASONS) == sorted(name for name, row in ROWS.items() if row[2]["functions"])
    LOG.info("described scripts exit scripts=%d seconds=%.4f", len(ROWS), time.perf_counter() - started)
