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
NEVER = "None where never is meant"
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
        dict(sources=2, declarations=3, functions=0),
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
        dict(sources=1, declarations=4, functions=0),
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
    "20_warehouse.py": (
        "A warehouse. `orders.csv` has one row per order: its number, the customer, when it was placed, and "
        "when it shipped, which is empty until it ships. `lines.csv` has one row per line of an order: the "
        "order, the product, the quantity, the shelf it was picked from and the worker who picked it, with the "
        "time of the pick. `shelves.csv` gives each shelf its aisle and height; `workers.csv` each worker's "
        "shift and the day they started; `products.csv` each product's weight and category, some weights "
        "missing. For each order, at the moment it is placed, predict whether it will ship more than two days "
        "later; the model may see only what was known at that moment, and orders placed in the last month of "
        "the data are for scoring.",
        14,
        dict(sources=5, declarations=9, functions=0),
    ),
    "21_league.py": (
        "A league. `matches.parquet` has one row per match: its two teams, the kick-off, the final score. "
        "`lineups.parquet` has one row per player per match: the team, the position, and whether they started. "
        "`substitutions.parquet` has one row per substitution: the match, the minute, the player off and the "
        "player on. `ratings.parquet` has a rating per player per week, and a rating is sometimes revised in a "
        "later week: each row carries the week it is about and the week it was published. For each match, at "
        "kick-off, predict the difference of the final scores; a model must use the ratings as they were "
        "published by then, not as revised later. Matches are split by season, a column of `matches.parquet`.",
        15,
        dict(sources=4, declarations=10, functions=0),
    ),
    "22_dispatch.py": (
        "A dispatch centre. `calls.jsonl` has one record per call: its time, its place, its priority, the "
        "dispatcher who took it, and a list of the vehicles that were free at that moment, each with its "
        "station; the vehicle the dispatcher sent is marked in that list. `vehicles.csv` gives each vehicle its "
        "type and station; `stations.csv` each station's place. `arrivals.csv` gives, per call, the minutes "
        "until the vehicle arrived, known only when it arrives. Several calls can come at the same moment and "
        "are then taken by different dispatchers, none of whom knows what the others choose. The model is to "
        "learn, for each call, which of the free vehicles to send: every free vehicle is an option, the one "
        "sent is the one chosen, and the minutes until arrival are the outcome of the chosen option alone.",
        10,
        dict(sources=4, declarations=6, functions=0),
    ),
    "23_grid.py": (
        "A power grid. `substations.csv` has one row per substation: its name, its region and its voltage class. "
        "`lines.csv` has one row per power line between two substations: its length and capacity, the day it "
        "entered service and, for some, the day it was retired. `load.parquet` holds a reading of the load of "
        "every substation every fifteen minutes, in one wide table with one column per substation. `outages.csv` "
        "has one row per outage of a line: when it began, when it ended, and its cause, which is entered by an "
        "engineer some days after the outage ends, the day of entry being a column. For every line, at the start "
        "of every day it is in service, predict whether it will have an outage within the next seven days; the "
        "readings and the causes may be used only as they were known at that moment; the lines of two named "
        "regions are kept apart for scoring.",
        12,
        dict(sources=4, declarations=8, functions=0),
    ),
    "24_tutoring.py": (
        "A tutoring system. `sessions.jsonl` has one record per session of a student: the student, the start, "
        "and a list of steps; each step holds the exercise shown, the hints the tutor could have given at that "
        "step (a list, each with its type and its length in words), the one hint it gave, and whether the "
        "student then answered correctly, which is known when the step ends, twenty to ninety seconds later, "
        "the time of the answer being in the step. `exercises.csv` gives each exercise its topic and "
        "difficulty, and `prerequisites.csv` lists pairs of topics, one needed before the other. Giving no hint "
        "is always possible and is logged as such. The model is to learn, at each step, which hint to give: the "
        "hints the tutor could have given and giving none are the options, the one given is the one chosen, and "
        "whether the answer was correct is the outcome of the chosen option alone. The sessions of the last "
        "four weeks of the data are for scoring.",
        12,
        dict(sources=3, declarations=9, functions=0),
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
OPENING = 'DESCRIPTION = "orders"\norders = source("orders.csv", reader="csv", columns=dict(number="identifier"))\n'
NAMED = 'references=dict(what=dict(sort="node", of=order, key="number", absent="stop"))'
CASES = {
    "a kind named as a word the script does not import": (
        "from graphspec import nodes, source, tasks\n"
        + OPENING
        + 'order = nodes(orders, key="number")\n'
        + "placed = tasks(orders, %s)\n" % NAMED,
        None,
        [],
        dict(sources=1, declarations=2, functions=0),
    ),
    "a word called through the library where the script gives its name to a kind": (
        "import graphspec\nfrom graphspec import nodes, source, tasks\n"
        + OPENING
        + 'ranked = graphspec.order(orders, within=["number"], by=["number"], gap=None)\n'
        + 'order = nodes(ranked, key="number")\n'
        + "placed = tasks(ranked, %s)\n" % NAMED,
        None,
        [],
        dict(sources=1, declarations=3, functions=0),
    ),
    "a name that is a word the script imports": (
        "from graphspec import nodes, order, source\n"
        + OPENING
        + 'ranked = order(orders, within=["number"], by=["number"], gap=None)\n'
        + 'order = nodes(ranked, key="number")\n',
        "binds a name twice or binds a word it imports",
        [],
        None,
    ),
    "a word called through the library whose name the script does not bind": (
        "import graphspec\nfrom graphspec import nodes, source\n"
        + OPENING
        + 'kept = graphspec.select(orders, where="number is not missing")\n'
        + 'order = nodes(kept, key="number")\n',
        "calls through the library a word whose name it does not bind",
        [],
        None,
    ),
    "a word imported under another name": (
        "from graphspec import order as arrange, source\n" + OPENING,
        "renames what it imports",
        [],
        None,
    ),
    "a word the script calls and does not import": (
        "from graphspec import source\n" + OPENING + 'order = nodes(orders, key="number")\n',
        "is no one word of the language bound to one name",
        [],
        None,
    ),
    "a target and an outcome stated as none where never is meant": (
        "from graphspec import nodes, source, tasks\n"
        + 'DESCRIPTION = "orders"\n'
        + 'orders = source("orders.csv", reader="csv", columns=dict(number="identifier", '
        + 'late=dict(type="flag", outcome=None)))\n'
        + 'order = nodes(orders, key="number", holds="always", known=None)\n'
        + 'placed = tasks(orders, %s, target=dict(kind="class", value="late", known=None))\n' % NAMED,
        None,
        [NEVER, NEVER],
        dict(sources=1, declarations=2, functions=0),
    ),
}


def opened(tree, where):
    started = time.perf_counter()
    body = list(tree.body)
    imports = [node for node in body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert body[: len(imports)] == imports, "%s imports after its first statement" % where
    words, foreign, whole = [], [], 0
    for node in imports:
        assert all(alias.asname is None for alias in node.names), "%s renames what it imports" % where
        if isinstance(node, ast.ImportFrom) and node.module == LANGUAGE and node.level == 0:
            words += [alias.name for alias in node.names]
        elif isinstance(node, ast.Import) and [alias.name for alias in node.names] == [LANGUAGE]:
            whole += 1
        else:
            named = [node.module or ""] if isinstance(node, ast.ImportFrom) else [alias.name for alias in node.names]
            assert not any(name.partition(".")[0] in (LANGUAGE, "tuning") for name in named), (
                "%s imports the library other than by its words" % where
            )
            foreign += [alias.name.partition(".")[0] for alias in node.names]
    assert sorted(set(words)) == sorted(words) and whole <= 1, "%s imports a word twice" % where
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
    return words, bool(whole), foreign, stated.value.value, rest[1:]


def refused(statements, spoken, through, foreign, where):
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
            held = {key.arg: key.value for key in node.keywords}
            none = [key for key, value in held.items() if isinstance(value, ast.Constant) and value.value is None]
            if id(node) not in through and name not in spoken and name not in (RECORD, SCRIPTED):
                found.append((where, line, "a call that is neither a word of the language nor a record"))
            elif not all(key.arg for key in node.keywords) or (name == RECORD and node.args):
                found.append((where, line, "arguments unpacked, or a record not made of names"))
            elif "outcome" in none or ("known" in none and "value" in held):
                found.append((where, line, NEVER))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in foreign:
            found.append((where, line, "a name imported from outside the language"))
    LOG.debug("refused exit script=%s found=%d seconds=%.6f", where, len(found), time.perf_counter() - started)
    return found


def read(name, text):
    started = time.perf_counter()
    tree = ast.parse(text)
    words, whole, foreign, description, statements = opened(tree, name)
    every = list(ast.walk(tree))
    calls = [node for node in every if isinstance(node, ast.Call)]
    direct = {id(node): node.func.id for node in calls if isinstance(node.func, ast.Name)}
    through = {
        id(node): node.func.attr
        for node in calls
        if whole
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == LANGUAGE
    }
    defined = [node for node in statements if isinstance(node, ast.FunctionDef)]
    listed = {}
    for node in defined:
        marks = node.decorator_list
        assert len(marks) == 1 and direct.get(id(marks[0])) == SCRIPTED and SCRIPTED in words, (
            "%s line %d is a function the language does not register" % (name, node.lineno)
        )
        reasons = [
            key.value.value
            for key in marks[0].keywords
            if key.arg == "reason" and isinstance(key.value, ast.Constant) and isinstance(key.value.value, str)
        ]
        assert len(reasons) == 1 and reasons[0], "%s line %d states no reason" % (name, node.lineno)
        listed[node.name] = reasons[0]
    imported = set(words)
    spoken = (imported - {SCRIPTED}) | set(listed)
    bound = []
    for node in statements:
        if node in defined:
            continue
        assert (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and (direct.get(id(node.value)) in spoken or through.get(id(node.value)) in WORDS)
        ), "%s line %d is no one word of the language bound to one name" % (name, node.lineno)
        bound.append(node.targets[0].id)
    assert len(set(bound)) == len(bound) and (imported | set(listed) | {RECORD, DESCRIBED, LANGUAGE}).isdisjoint(
        bound
    ), "%s binds a name twice or binds a word it imports" % name
    assert set(through.values()) <= set(WORDS) & set(bound), (
        "%s calls through the library a word whose name it does not bind" % name
    )
    loaded = [node.id for node in every if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)]
    assert sum(found in imported for found in loaded) == sum(found in imported for found in direct.values()), (
        "%s names a word of the language without calling it" % name
    )
    assert loaded.count(LANGUAGE) == len(through), "%s names the library other than to call a word" % name
    assert sum(found == SCRIPTED for found in direct.values()) == len(defined), (
        "%s registers other than a top-level function of its own" % name
    )
    functions = [node for node in every if isinstance(node, FUNCTIONS)]
    assert sorted(getattr(node, "name", "lambda") for node in functions) == sorted(listed), (
        "%s holds a function it does not register with its reason" % name
    )
    named = [found for found in direct.values() if found in spoken]
    stated = named + list(through.values())
    result = dict(
        description=description,
        statements=len(statements),
        counts=dict(
            sources=sum(call in SOURCES for call in stated),
            declarations=sum(call not in SOURCES for call in stated),
            functions=len(functions),
        ),
        words=sorted(set(named) & set(WORDS)),
        imported=sorted(imported - {SCRIPTED}),
        functions=listed,
        refused=refused(statements, spoken, through, foreign, name),
    )
    LOG.debug("read exit script=%s statements=%d seconds=%.6f", name, len(statements), time.perf_counter() - started)
    return result


def test_every_described_script_is_declarations_alone_counted_within_its_ceiling_with_its_functions_listed():
    started = time.perf_counter()
    assert sorted(path.name for path in ROOT.iterdir()) == sorted(ROWS)
    assert len(ROWS) == 24 and len({description for description, _, _ in ROWS.values()}) == 24
    for name, (description, ceiling, expected) in ROWS.items():
        found = read(name, (ROOT / name).read_text(encoding="utf-8"))
        assert found["description"] == description, name
        assert found["refused"] == [], name
        assert found["counts"] == expected, name
        assert sum(found["counts"].values()) == found["statements"] <= ceiling, name
        assert found["functions"] == REASONS.get(name, {}), name
        assert found["words"] == found["imported"], name
        assert set(found["words"]) & set(TARGETED), name
    assert sorted(REASONS) == sorted(name for name, row in ROWS.items() if row[2]["functions"])
    LOG.info("described scripts exit scripts=%d seconds=%.4f", len(ROWS), time.perf_counter() - started)


def test_a_script_names_no_word_it_imports_calls_a_word_so_named_through_the_library_and_writes_never():
    started = time.perf_counter()
    for name, (text, message, reasons, expected) in CASES.items():
        if message is not None:
            with pytest.raises(AssertionError, match=message):
                read(name, text)
            continue
        found = read(name, text)
        assert [reason for _, _, reason in found["refused"]] == reasons, name
        assert found["counts"] == expected and sum(expected.values()) == found["statements"], name
        assert found["words"] == found["imported"], name
    LOG.info("described cases exit cases=%d seconds=%.4f", len(CASES), time.perf_counter() - started)
