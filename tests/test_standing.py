import ast
import logging
import re
import time
from pathlib import Path

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).parent.parent
CODE = ("graphspec", "tuning")
NAMED = ("graphspec", "tuning")
TIMERS = ("perf_counter", "timed", "measure")
MARKS = ("timed",)
NUMERAL = ("3", "iii", "three")
BETWEEN = r"[\W_]*"
PIECES = (
    ("chu", "rn"),
    ("total", "war"),
    ("vi", "c", NUMERAL),
    ("vict", "oria", NUMERAL),
)
COMPILERS = ("njit", "jit")
LOADERS = ("import_module", "__import__", "run_module")
LOCAL = re.compile(r"(^|[\\/])local([\\/]|$)")


def compiled(node):
    return any(
        getattr(getattr(entry, "func", entry), "id", None) in COMPILERS for entry in node.decorator_list
    )


def within(name, package):
    return name == package or name.startswith(package + ".")


def sources(root, folders):
    started = time.perf_counter()
    found = [
        (path.relative_to(root).as_posix(), path.read_text(encoding="utf-8"))
        for folder in folders
        for path in sorted((root / folder).rglob("*.py"))
    ]
    LOG.debug("sources exit modules=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def called(node):
    return node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", None)


def timing(tree):
    started = time.perf_counter()
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and called(node.func) in TIMERS:
            found = True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found = found or any(called(entry) in MARKS for entry in node.decorator_list)
    LOG.debug("timing exit found=%s seconds=%.4f", found, time.perf_counter() - started)
    return found


def untimed(root, folders):
    started = time.perf_counter()
    found = []
    for place, source in sources(root, folders):
        tree = ast.parse(source)
        defines = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not compiled(node)
            for node in ast.walk(tree)
        )
        if defines and not timing(tree):
            found.append(place)
    LOG.debug("untimed exit modules=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def spelling(pieces):
    return BETWEEN.join(
        re.escape(piece) if isinstance(piece, str) else "(?:" + "|".join(map(re.escape, piece)) + ")"
        for piece in pieces
    )


def spelled(pieces, joint, change):
    started = time.perf_counter()
    found = [""]
    for piece in pieces:
        found = [
            held + (joint if held else "") + change(choice)
            for held in found
            for choice in ([piece] if isinstance(piece, str) else piece)
        ]
    LOG.debug("spelled exit names=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def mentions(source):
    started = time.perf_counter()
    names, strings = [], []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.append((node.module, node.lineno))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.append((node.value, node.lineno))
        elif isinstance(node, ast.Call):
            function = node.func
            called = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", None)
            if called in LOADERS:
                names.extend(
                    (argument.value, node.lineno)
                    for argument in node.args
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                )
    LOG.debug("mentions exit names=%d seconds=%.4f", len(names), time.perf_counter() - started)
    return names, strings


def crossing(root, folders):
    started = time.perf_counter()
    found = []
    for place, source in sources(root, folders):
        inner = place.startswith("graphspec/")
        if "local" not in source and not (inner and "tuning" in source):
            continue
        names, strings = mentions(source)
        if inner:
            found.extend((place, line, name) for name, line in names + strings if within(name, "tuning"))
        found.extend((place, line, name) for name, line in names if within(name, "local"))
        found.extend((place, line, value) for value, line in strings if LOCAL.search(value))
    LOG.debug("crossing exit found=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def named(root, places, excluded):
    started = time.perf_counter()
    patterns = [re.compile(spelling(pieces), re.IGNORECASE) for pieces in PIECES]
    found = []
    for place in places:
        for path in [root / place] if (root / place).is_file() else sorted((root / place).rglob("*")):
            if not path.is_file() or path.suffix == ".pyc" or path == excluded:
                continue
            where = path.relative_to(root).as_posix()
            for text in (where, path.read_bytes().decode("utf-8", errors="replace")):
                found.extend(
                    (where, match.group(0)) for pattern in patterns for match in pattern.finditer(text)
                )
    LOG.debug("named exit found=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def write(root, place, source):
    started = time.perf_counter()
    path = root / place
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    LOG.debug("write exit place=%s seconds=%.4f", place, time.perf_counter() - started)


def test_every_module_that_defines_a_function_times_something(tmp_path):
    assert untimed(ROOT, CODE) == []
    write(tmp_path, "graphspec/plain.py", "def work():\n    return 1\n")
    write(tmp_path, "graphspec/constant.py", "VALUE = 1\n")
    write(
        tmp_path,
        "tuning/kernel.py",
        "from numba import njit\n\n\n@njit(cache=True)\ndef work():\n    return 1\n",
    )
    write(tmp_path, "tuning/clocked.py", "import time\n\n\ndef work():\n    return time.perf_counter()\n")
    write(
        tmp_path,
        "tuning/marked.py",
        "from graphspec.infra.timing import timed\n\n\n@timed('work')\ndef work():\n    return 1\n",
    )
    write(
        tmp_path,
        "tuning/measured.py",
        "from graphspec.infra.timing import measure\n\n\ndef work():\n"
        "    with measure('work'):\n        return 1\n",
    )
    write(tmp_path, "graphspec/remarked.py", "def work():\n    return 1  # perf_counter timed measure(\n")
    write(tmp_path, "graphspec/quoted.py", "def work():\n    return 'time.perf_counter() measure( timed'\n")
    write(
        tmp_path,
        "graphspec/dated.py",
        "from datetime import timedelta\n\n\ndef work():\n    return timedelta(seconds=1)\n",
    )
    write(
        tmp_path,
        "graphspec/imported.py",
        "from time import perf_counter\n\nfrom graphspec.infra.timing import measure, timed\n\n\n"
        "def work():\n    return perf_counter, measure, timed\n",
    )
    write(tmp_path, "tuning/measurement.py", "def work(measurement, untimed):\n    return measurement(untimed)\n")
    assert untimed(tmp_path, CODE) == [
        "graphspec/dated.py",
        "graphspec/imported.py",
        "graphspec/plain.py",
        "graphspec/quoted.py",
        "graphspec/remarked.py",
        "tuning/measurement.py",
    ]


def test_no_name_of_a_dataset_that_was_an_adapter_appears(tmp_path):
    assert named(ROOT, NAMED, Path(__file__)) == []
    written = [
        name
        for pieces in PIECES
        for joint in ("", " ", "_", "-", "  ", "_-_")
        for change in (str.lower, str.upper, str.title)
        for name in spelled(pieces, joint, change)
    ]
    assert len(written) == len(set(written)) == (2 + 2 * len(NUMERAL)) * 6 * 3
    for index, name in enumerate(written):
        write(tmp_path, "graphspec/text_%d.md" % index, "built for " + name + " alone\n")
    write(tmp_path, "tuning/" + "".join(PIECES[0]) + ".py", "VALUE = 1\n")
    write(tmp_path, "tuning/clean.py", "VALUE = 'victory over total variation in 3 civic steps'\n")
    found = named(tmp_path, NAMED, Path(__file__))
    assert len(found) >= len(written) + 1
    assert {place for place, _ in found} == {
        "graphspec/text_%d.md" % index for index in range(len(written))
    } | {"tuning/" + "".join(PIECES[0]) + ".py"}


def test_graphspec_does_not_import_tuning_and_nothing_reads_local(tmp_path):
    assert crossing(ROOT, CODE) == []
    write(tmp_path, "graphspec/direct.py", "import tuning.run\n")
    write(tmp_path, "graphspec/inside.py", "def work():\n    from tuning import run\n\n    return run\n")
    write(
        tmp_path,
        "graphspec/named.py",
        "from importlib import import_module\n\n\ndef work():\n    return import_module('tuning.run')\n",
    )
    write(tmp_path, "graphspec/clean.py", "MESSAGE = 'isolated tuning workers'\nlocal = 1\n")
    write(tmp_path, "tuning/reads.py", "from pathlib import Path\n\nPLACE = Path('local/script.py')\n")
    write(tmp_path, "tuning/imports.py", "import local.script\n")
    write(tmp_path, "tuning/allowed.py", "import graphspec\n")
    assert sorted({place for place, _, _ in crossing(tmp_path, CODE)}) == [
        "graphspec/direct.py",
        "graphspec/inside.py",
        "graphspec/named.py",
        "tuning/imports.py",
        "tuning/reads.py",
    ]
