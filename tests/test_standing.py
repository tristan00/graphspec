import ast
import hashlib
import logging
import re
import time
from functools import lru_cache
from pathlib import Path

import pytest

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("standing")
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
INFRA = "graphspec/infra"
GROUPED = ("graphspec", "tuning")
LAYERS = {
    INFRA: (),
    "graphspec/registries.py": (),
    "graphspec/budgets.py": (),
    "graphspec/schema": ("graphspec/registries.py", "graphspec/budgets.py"),
    "graphspec/facts": ("graphspec/schema",),
    "graphspec/views": ("graphspec/facts",),
    "graphspec/build": ("graphspec/views",),
    "graphspec/sources": ("graphspec/build",),
    "graphspec/operations": ("graphspec/views",),
    "graphspec/representations": ("graphspec/operations",),
    "graphspec/engine": ("graphspec/representations",),
    "graphspec/descriptions": ("graphspec/engine", "graphspec/sources"),
    "graphspec/__init__.py": ("graphspec/descriptions",),
    "tuning/statement.py": ("graphspec/__init__.py",),
    "tuning/families": ("tuning/statement.py",),
    "tuning/bank": ("tuning/families",),
    "tuning/batches": ("tuning/bank",),
    "tuning/measures": ("tuning/batches",),
    "tuning/descriptions": ("tuning/measures",),
    "tuning/fits": ("tuning/descriptions",),
    "tuning/searches": ("tuning/fits",),
    "tuning/api.py": ("tuning/searches",),
    "tuning/dashboard.py": ("tuning/searches",),
    "tuning/__main__.py": ("tuning/api.py", "tuning/dashboard.py"),
    "tuning/__init__.py": ("tuning/api.py", "tuning/dashboard.py"),
}
VERSIONED = {
    "graphspec.periods": ("graphspec/operations/periods.py",),
    "graphspec.window": ("graphspec/operations/window.py",),
    "graphspec.nodes": ("graphspec/operations/nodes.py",),
    "graphspec.derive": ("graphspec/operations/derive.py",),
    "graphspec.relations": ("graphspec/operations/relations.py",),
    "graphspec.edges": ("graphspec/operations/edges.py",),
    "graphspec.relation_forms": ("graphspec/operations/relation_forms.py",),
    "graphspec.merge": ("graphspec/operations/merge.py",),
    "graphspec.difference": ("graphspec/operations/difference.py",),
    "graphspec.nearby": ("graphspec/operations/nearby.py",),
    "graphspec.consecutive": ("graphspec/operations/consecutive.py",),
    "graphspec.neighbourhood": ("graphspec/operations/neighbourhood.py",),
    "graphspec.measures": ("graphspec/operations/measures.py",),
    "graphspec.contract": ("graphspec/operations/contract.py",),
    "derivation": (
        "graphspec/engine",
        "graphspec/operations/kernels.py",
        "graphspec/operations/operation.py",
        "graphspec/operations/parameters.py",
        "graphspec/operations/pipeline.py",
        "graphspec/operations/rows.py",
        "graphspec/operations/state.py",
        "graphspec/infra/compute.py",
        "graphspec/infra/hashing.py",
        "graphspec/infra/workspace.py",
    ),
    "views": (
        "graphspec/facts",
        "graphspec/schema",
        "graphspec/views",
        "graphspec/operations/sets.py",
        "tuning/bank/bank.py",
    ),
    "representations": ("graphspec/representations",),
    "index": ("graphspec/build/staged.py",),
}
UNVERSIONED = (
    "graphspec/__init__.py",
    "graphspec/operations/__init__.py",
    "graphspec/budgets.py",
    "graphspec/registries.py",
    "graphspec/build/__init__.py",
    "graphspec/build/calls.py",
    "graphspec/build/forms.py",
    "graphspec/build/output.py",
    "graphspec/build/parse.py",
    "graphspec/descriptions",
    "graphspec/sources",
    "graphspec/infra/__init__.py",
    "graphspec/infra/arrays.py",
    "graphspec/infra/code.py",
    "graphspec/infra/database.py",
    "graphspec/infra/files.py",
    "graphspec/infra/log.py",
    "graphspec/infra/processes",
    "graphspec/infra/runtime.py",
    "graphspec/infra/serve.py",
    "graphspec/infra/timing.py",
    "graphspec/infra/windows_files.py",
)
RECORDED = {
    "graphspec.periods": (5, "801dde91649ee01b08f81a7a8dc0821dee53a49e12cf8d45a27f1cf2ec4c7a35"),
    "graphspec.window": (8, "e4dc992b560a0ef5b38c09072e6d8ad51e35403ecd2da959dc6484eff67e4899"),
    "graphspec.nodes": (6, "54fc43580dbf8470fee9f68a2d54a179bfe2f810ee01818b36f22f3dbdf65cba"),
    "graphspec.derive": (9, "b9847e1ab20f4ade0177f9cb54c79653752cd9ce0cd9792a1ac436a26a25acf7"),
    "graphspec.relations": (5, "c4fa3ed5da0e9121609653133853386c5c2425cb5922b37c17cd2506cdf6d54d"),
    "graphspec.edges": (9, "cac33b8c4babee91f5ee7c73591a9bb5685157e4ad92e511212c3f47a5054846"),
    "graphspec.relation_forms": (3, "ccd052ae61ac23618f50925d178e80526fbe90e7832b664567358ae3723fdf51"),
    "graphspec.merge": (6, "e5bb973ef6e36bff52cc6ae0a4dc9da2aac1f0fb9b159fe8ee02670d8470ff23"),
    "graphspec.difference": (4, "7c9daef7d4fae58b7d8942dde4e68928341ec8792e211789a83b11e0588709df"),
    "graphspec.nearby": (4, "3e8a442bf7f009ea1e99752073cb39ed72b4524452e923a0bbe5e82131cbf72a"),
    "graphspec.consecutive": (4, "216674c68a8cfea86f45091c21a460b08b37cf96468ad3a69f410e454c090f8a"),
    "graphspec.neighbourhood": (3, "3127f2221864f959236caa9d4a75ded96b262d3db1774174f0a311540627b589"),
    "graphspec.measures": (2, "d11543d525c8939d5af59c1eb88f41bc2799b9d4f7a97940c337e78b54b2a20d"),
    "graphspec.contract": (4, "ec63035c66e116a0f8ad7e19c4335814d3f9544bb4a0391d932e4247bfff0dc3"),
    "derivation": (24, "95c26b3bd010fbb8c39425ef003fe24d3b3d70bb49a522eeb3cca97cb919f100"),
    "views": (17, "e727b71e057db1397b2223d928e510d055302c72ae8c5fe988ec2b91105e1463"),
    "representations": (11, "8406e03fdf0ab14c75c518d546e4579d64866bbf9df275a065db0832f9a35d87"),
    "index": (3, "88b543b0553df433070b05d9875cff180c36e2420133819f9149d993577ca405"),
}


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


@lru_cache(maxsize=None)
def parsed(source):
    started = time.perf_counter()
    tree = ast.parse(source)
    LOG.debug("parsed exit characters=%d seconds=%.4f", len(source), time.perf_counter() - started)
    return tree


@pytest.fixture(scope="module", autouse=True)
def library():
    started = time.perf_counter()
    trees = [parsed(source) for _, source in sources(ROOT, CODE)]
    LOG.debug("library parsed exit modules=%d seconds=%.4f", len(trees), time.perf_counter() - started)


def timing(tree):
    started = time.perf_counter()
    found = any(
        called(node.func) in TIMERS
        if isinstance(node, ast.Call)
        else isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(called(entry) in MARKS for entry in node.decorator_list)
        for node in ast.walk(tree)
    )
    LOG.debug("timing exit found=%s seconds=%.4f", found, time.perf_counter() - started)
    return found


def untimed(root, folders):
    started = time.perf_counter()
    found = []
    for place, source in sources(root, folders):
        tree = parsed(source)
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
    for node in ast.walk(parsed(source)):
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


def resolved(known, name):
    started = time.perf_counter()
    place = name.replace(".", "/")
    found = next((held for held in (place + ".py", place + "/__init__.py") if held in known), None)
    LOG.debug("resolved exit name=%s seconds=%.6f", name, time.perf_counter() - started)
    return found


def statements(tree):
    started = time.perf_counter()
    found, waiting = [], [tree]
    while waiting:
        node = waiting.pop()
        found.append(node)
        for name in ("body", "orelse", "finalbody", "handlers", "cases"):
            waiting.extend(child for child in getattr(node, name, ()) if isinstance(child, ast.AST))
    LOG.debug("statements exit found=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def imports(modules):
    started = time.perf_counter()
    known, found = {place for place, _ in modules}, []
    for place, source in modules:
        package = place[: -len(".py")].split("/")[:-1]
        for node in statements(parsed(source)):
            targets = []
            if isinstance(node, ast.Import):
                targets = [resolved(known, alias.name) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = ".".join(package[: len(package) - node.level + 1] + ([base] if base else []))
                targets = [
                    resolved(known, base + "." + alias.name) or resolved(known, base) for alias in node.names
                ]
            found.extend((place, node.lineno, target) for target in targets if target is not None)
    LOG.debug("imports exit found=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def group(place):
    started = time.perf_counter()
    parts = place.split("/")
    found = INFRA if place.startswith(INFRA + "/") else "/".join(parts[:2]) if len(parts) > 2 else place
    LOG.debug("group exit place=%s seconds=%.6f", place, time.perf_counter() - started)
    return found


def below(name):
    started = time.perf_counter()
    found, waiting = set(), list(LAYERS.get(name, ()))
    while waiting:
        held = waiting.pop()
        if held not in found:
            found.add(held)
            waiting.extend(LAYERS[held])
    LOG.debug("below exit group=%s groups=%d seconds=%.4f", name, len(found), time.perf_counter() - started)
    return found | ({INFRA} if name != INFRA else set())


def layered(root, folders):
    started = time.perf_counter()
    reach = {name: below(name) for name in LAYERS}
    modules = sources(root, folders)
    unplaced = sorted(
        place for place, _ in modules if place.split("/")[0] in GROUPED and group(place) not in LAYERS
    )
    crossing = sorted(
        (place, line, group(place), group(target))
        for place, line, target in imports(modules)
        if group(place) in LAYERS and group(target) != group(place) and group(target) not in reach[group(place)]
    )
    LOG.debug(
        "layered exit unplaced=%d crossing=%d seconds=%.4f",
        len(unplaced),
        len(crossing),
        time.perf_counter() - started,
    )
    return unplaced, crossing


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


def test_every_module_sits_in_a_group_and_imports_only_the_groups_below_it(tmp_path):
    assert layered(ROOT, CODE) == ([], [])
    for place in ("graphspec", "graphspec/infra", "graphspec/infra/processes", "graphspec/views", "tuning"):
        write(tmp_path, place + "/__init__.py", "")
    for place in ("registries", "infra/files", "operations/sets", "views/graph", "engine/transform"):
        write(tmp_path, "graphspec/%s.py" % place, "VALUE = 1\n")
    write(tmp_path, "tuning/searches/run.py", "from graphspec.infra.files import VALUE\n")
    write(tmp_path, "tuning/bank/bank.py", "from graphspec.views.graph import VALUE\n")
    write(tmp_path, "tuning/families/layers.py", "from ..bank import bank\n")
    write(tmp_path, "tuning/loose.py", "VALUE = 1\n")
    write(tmp_path, "graphspec/loose.py", "from graphspec.infra.files import VALUE\n")
    write(tmp_path, "graphspec/views/seen.py", "from .graph import VALUE\nfrom graphspec.operations.sets import VALUE\n")
    write(tmp_path, "graphspec/views/late.py", "def work():\n    from ..engine import transform\n\n    return transform\n")
    write(tmp_path, "graphspec/engine/fine.py", "from graphspec.infra import files\nfrom graphspec.views import graph\n")
    write(tmp_path, "graphspec/infra/pools.py", "from ..registries import VALUE\nfrom .files import VALUE as SAME\n")
    write(tmp_path, "graphspec/infra/processes/inside.py", "def work():\n    import tuning.searches.run\n\n    return 1\n")
    write(tmp_path, "graphspec/infra/whole.py", "from graphspec import registries\nfrom graphspec.infra import files\n")
    assert layered(tmp_path, CODE) == (
        ["graphspec/loose.py", "tuning/loose.py"],
        [
            ("graphspec/infra/pools.py", 1, INFRA, "graphspec/registries.py"),
            ("graphspec/infra/processes/inside.py", 2, INFRA, "tuning/searches"),
            ("graphspec/infra/whole.py", 1, INFRA, "graphspec/registries.py"),
            ("graphspec/views/late.py", 2, "graphspec/views", "graphspec/engine"),
            ("graphspec/views/seen.py", 2, "graphspec/views", "graphspec/operations"),
            ("tuning/families/layers.py", 1, "tuning/families", "tuning/bank"),
        ],
    )


def listed(root, places):
    started = time.perf_counter()
    found = sorted(
        path.relative_to(root).as_posix()
        for place in places
        for path in ((root / place).rglob("*.py") if (root / place).is_dir() else [root / place])
    )
    LOG.debug("listed exit modules=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def digested(root, places):
    started = time.perf_counter()
    value = hashlib.sha256()
    for place in listed(root, places):
        value.update(place.encode())
        value.update((root / place).read_bytes().replace(b"\r\n", b"\n"))
    LOG.debug("digested exit places=%d seconds=%.4f", len(places), time.perf_counter() - started)
    return value.hexdigest()


def versions():
    from graphspec import OPERATIONS
    from graphspec.build.staged import INDEX
    from graphspec.engine.transform import VERSION as DERIVATION
    from graphspec.representations.groups import VERSION as REPRESENTATIONS
    from tuning.bank.bank import VIEW

    started = time.perf_counter()
    found = {name: OPERATIONS.named(name).version for name in OPERATIONS.names() if name.startswith("graphspec.")}
    found.update(derivation=DERIVATION, views=VIEW, representations=REPRESENTATIONS, index=INDEX)
    LOG.debug("versions exit units=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def unrecorded(root, units, recorded, held):
    started = time.perf_counter()
    found = []
    for name, places in units.items():
        version, digest = held[name], digested(root, places)
        if name in recorded and recorded[name] == (version, digest):
            continue
        if name in recorded and recorded[name][0] == version:
            found.append("the code of %s changed and its version %d did not; raise it" % (name, version))
        else:
            found.append("record %s at version %d with the digest %s" % (name, version, digest))
    LOG.debug("unrecorded exit found=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def test_the_code_of_an_operation_or_of_a_part_that_writes_a_cache_changes_only_with_its_version(tmp_path):
    held = versions()
    assert sorted(held) == sorted(VERSIONED)
    placed = [place for places in VERSIONED.values() for place in listed(ROOT, places)] + listed(ROOT, UNVERSIONED)
    assert len(placed) == len(set(placed))
    assert sorted(set(listed(ROOT, ("graphspec",))) - set(placed)) == []
    assert unrecorded(ROOT, VERSIONED, RECORDED, held) == []
    window = {name: VERSIONED[name] for name in ("graphspec.window",)}
    source = (ROOT / window["graphspec.window"][0]).read_text(encoding="utf-8")
    write(tmp_path, window["graphspec.window"][0], source + "\nEDITED = 1\n")
    version = held["graphspec.window"]
    assert unrecorded(tmp_path, window, RECORDED, held) == [
        "the code of graphspec.window changed and its version %d did not; raise it" % version
    ]
    assert unrecorded(tmp_path, window, RECORDED, dict(held, **{"graphspec.window": version + 1}))[0].startswith(
        "record graphspec.window at version %d with the digest " % (version + 1)
    )
