import ast
import hashlib
import logging
import platform
import re
import time
import tomllib
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.xdist_group("standing")
ROOT = Path(__file__).parent.parent
CODE = ("graphspec", "tuning")
NAMED = ("graphspec", "tuning")
PRIVATE = {
    "numba": ("_data_name", "_index_key", "_index_path", "_load_index", "_py_func", "_save_data", "_save_index", "_unique_ids"),
    "numpy": ("_mmap",),
    "optuna": ("_gamma",),
    "python": ("_getfinalpathname", "_getframe", "_handle", "_winapi"),
}
PUBLIC = ("_exit",)
UNDERSCORED = re.compile(r"(?<!\w)_[^\W_]")
PROJECT = ("graphspec", "tuning", "tests", "conftest.py")
MARKS = ("timed", "staged")
LOGGERS = ("debug", "info", "warning", "error", "exception", "critical")
NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
BODIES = ("body", "orelse", "finalbody", "handlers", "cases")
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
    "graphspec.product": ("graphspec/operations/product.py",),
    "graphspec.negatives": ("graphspec/operations/negatives.py",),
    "derivation": (
        "graphspec/engine",
        "graphspec/operations/kernels.py",
        "graphspec/operations/operation.py",
        "graphspec/operations/parameters.py",
        "graphspec/operations/paths.py",
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
    "graphspec.periods": (6, "cd172dca349c911dae1c321344a5d8571ca0a87a2236b66bbf31ac55cdf41c95"),
    "graphspec.window": (9, "315a707590f159fe2bd680a8e65231f0138836be4fd3cfc6fc5fe7484e4e8e26"),
    "graphspec.nodes": (10, "0a71c751bfa6ca8d935eaa328c4032df83de03138c908b956dcfef7bd3d2b3ac"),
    "graphspec.derive": (11, "0bcbbdd6b2372962fa09fc2d0f489bdfd6204ebb949c5e0fe6279b321891958e"),
    "graphspec.relations": (6, "4f53a7e4da84a14b242dffa7bf6b3cf8b6e7e2aa7568f83d609b1762267aeaf6"),
    "graphspec.edges": (10, "1d48cae007cdd44335b5ca9ce5dd6ea5ace44bbd5ee150b5807932ca6af3c6e6"),
    "graphspec.relation_forms": (5, "d171f2d0bd7961373fe3180ac7ac6306ea350799fa0df01dfe8696a4b0e00570"),
    "graphspec.merge": (7, "85736377e2a227c72e9316de8e5e513b80fd99d58f39b955240444704b27d6bb"),
    "graphspec.difference": (5, "4d4f13f3cc158e0fddb9dde7b324af58d6f6d81be6b71bc9f1e8a670418a32d0"),
    "graphspec.nearby": (6, "6a901b29d15031a8f8017201e9dde987ab59ee4837efad6917a0ad72c84633a3"),
    "graphspec.consecutive": (6, "d244bd7273e2bb8767c3f9b7cbba8362a9014cb22c412c5d828f2c4f4ab953c9"),
    "graphspec.neighbourhood": (6, "7feff8c7492860ef0019661bd2b46a47f2373daf01faf7e4974a71762ac2629c"),
    "graphspec.measures": (3, "738ad2944004dd94058b9877cbe44c1755ace68f2466b16d92affca51993aa88"),
    "graphspec.contract": (7, "38b16e367216b1bbcc637e922fd74ef7e4d1c85e857b4067383f3568c3e5e3fd"),
    "graphspec.product": (3, "d0c09e09b3f2d702ab9d6386bfdafc3c35cddfdae8a9fc4fa21283446479d2e7"),
    "graphspec.negatives": (6, "14defe736b009f6eaf6fa92b1dd2b29508298985ba8227511aa4b5ab5b34d7a6"),
    "derivation": (38, "2be3beb0a2894c89fc01dddc56c72c0f6e12854cc550229730f5648d738304f4"),
    "views": (48, "d54f2274e64e460757472d5d99f685915ef77aee309e3355845840ca32435afc"),
    "representations": (17, "6c373d07917515df876e95c6f3b946d61eccb5d105b0face340a3d80d4901e9c"),
    "index": (4, "de51fe453ec964424b70224c54bc8f64438a811c8326745daa22a105984f3f36"),
}
CALLED = {
    "graphspec/budgets.py:GraphBudget.pair_limit": 16521,
    "graphspec/budgets.py:GraphBudget.pairs": 15046,
    "graphspec/budgets.py:GraphBudget.retain": 8673,
    "graphspec/budgets.py:GraphBudget.workspace": 9060,
    "graphspec/build/staged.py:Staged.positions": 6736,
    "graphspec/build/staged.py:numbers": 7586,
    "graphspec/engine/stitch.py:globalized": 6939,
    "graphspec/engine/transform.py:checked": 5121,
    "graphspec/facts/stored.py:Arrays.__getattr__": 505279,
    "graphspec/facts/stored.py:Arrays.__init__": 8877,
    "graphspec/facts/stored.py:Facts.__getattr__": 31659,
    "graphspec/infra/arrays.py:Bundle.hold": 47217,
    "graphspec/infra/arrays.py:Pending.__init__": 47217,
    "graphspec/infra/arrays.py:Resident.numpy": 73489,
    "graphspec/infra/arrays.py:Resident.tensor": 5149,
    "graphspec/infra/arrays.py:between": 7281,
    "graphspec/infra/arrays.py:compact": 47217,
    "graphspec/infra/files.py:Arrays.__getitem__": 17602,
    "graphspec/infra/files.py:Arrays.read": 8484,
    "graphspec/infra/files.py:aligned": 46432,
    "graphspec/infra/files.py:load_array": 26577,
    "graphspec/infra/files.py:save_array": 7088,
    "graphspec/infra/files.py:sha256_bytes": 5291,
    "graphspec/infra/files.py:sha256_json": 5271,
    "graphspec/infra/hashing.py:stacked": 7620,
    "graphspec/infra/log.py:JsonLog.format": 7,
    "graphspec/infra/timing.py:Timeline.measure": 42322,
    "graphspec/infra/timing.py:measure": 656901,
    "graphspec/infra/timing.py:timed": 26243,
    "graphspec/infra/timing.py:timed.<locals>.decorate": 26243,
    "graphspec/infra/workspace.py:allocate": 8872,
    "graphspec/infra/workspace.py:block_rows": 20760,
    "graphspec/infra/workspace.py:gathered": 21453,
    "graphspec/infra/workspace.py:held": 10104,
    "graphspec/infra/workspace.py:scatter": 14158,
    "graphspec/infra/workspace.py:take": 8027,
    "graphspec/operations/consecutive.py:selves": 12771,
    "graphspec/operations/derive.py:named": 8327,
    "graphspec/operations/edges.py:filtering": 24490,
    "graphspec/operations/edges.py:filters": 13423,
    "graphspec/operations/nearby.py:measured": 14576,
    "graphspec/operations/nodes.py:named": 6076,
    "graphspec/operations/operation.py:Operation.__call__": 5362,
    "graphspec/operations/operation.py:Operation.added": 6035,
    "graphspec/operations/operation.py:Operation.check": 5402,
    "graphspec/operations/operation.py:Operation.roles": 22552,
    "graphspec/operations/operation.py:Operation.settings": 26698,
    "graphspec/operations/operation.py:Operation.stated": 59961,
    "graphspec/operations/operation.py:Phase.once": 9031,
    "graphspec/operations/operation.py:Phase.run": 8580,
    "graphspec/operations/operation.py:Settings.__contains__": 191893,
    "graphspec/operations/operation.py:Settings.__getitem__": 240893,
    "graphspec/operations/operation.py:Settings.__init__": 26698,
    "graphspec/operations/operation.py:Settings.stated": 432786,
    "graphspec/operations/parameters.py:Parameter.__init__": 498861,
    "graphspec/operations/parameters.py:Parameter.active": 1848656,
    "graphspec/operations/parameters.py:Parameter.check": 688506,
    "graphspec/operations/parameters.py:Parameter.declaration": 50871,
    "graphspec/operations/parameters.py:Parameter.given": 18692,
    "graphspec/operations/parameters.py:Parameter.uniform": 36161,
    "graphspec/operations/parameters.py:among": 78107,
    "graphspec/operations/parameters.py:bound_for": 443490,
    "graphspec/operations/parameters.py:canonical.<locals>.read": 21605,
    "graphspec/operations/parameters.py:choice": 74958,
    "graphspec/operations/parameters.py:differs": 16226,
    "graphspec/operations/parameters.py:draw": 8985,
    "graphspec/operations/parameters.py:enabled": 874054,
    "graphspec/operations/parameters.py:equals": 56393,
    "graphspec/operations/parameters.py:flag": 45964,
    "graphspec/operations/parameters.py:holds": 282767,
    "graphspec/operations/parameters.py:narrow": 5157,
    "graphspec/operations/parameters.py:number": 135472,
    "graphspec/operations/parameters.py:real": 31628,
    "graphspec/operations/paths.py:composed": 166358,
    "graphspec/operations/paths.py:kinds": 330422,
    "graphspec/operations/paths.py:reduced": 6532,
    "graphspec/operations/paths.py:written": 89921,
    "graphspec/operations/pipeline.py:Growing.before": 10396,
    "graphspec/operations/pipeline.py:Local.__contains__": 90038,
    "graphspec/operations/pipeline.py:Local.__getitem__": 68677,
    "graphspec/operations/pipeline.py:Local.__init__": 117441,
    "graphspec/operations/pipeline.py:Offered.allowed": 45001,
    "graphspec/operations/pipeline.py:Step.__call__": 5358,
    "graphspec/operations/pipeline.py:Step.__init__": 10427,
    "graphspec/operations/pipeline.py:called": 11502,
    "graphspec/operations/pipeline.py:checked": 6290,
    "graphspec/operations/pipeline.py:entered": 12497,
    "graphspec/operations/pipeline.py:given": 10396,
    "graphspec/operations/pipeline.py:made": 30840,
    "graphspec/operations/pipeline.py:narrowed": 32880,
    "graphspec/operations/pipeline.py:probed.<locals>.probe": 10738,
    "graphspec/operations/pipeline.py:probed.<locals>.spec": 30317,
    "graphspec/operations/pipeline.py:qualified": 207455,
    "graphspec/operations/pipeline.py:roled": 32952,
    "graphspec/operations/pipeline.py:settled": 523961,
    "graphspec/operations/sets.py:chosen": 14951,
    "graphspec/operations/state.py:Segment.named": 5414,
    "graphspec/operations/state.py:Table.__init__": 14455,
    "graphspec/operations/state.py:Table.find": 15709,
    "graphspec/operations/state.py:Table.lookup": 9325,
    "graphspec/operations/window.py:by_period": 6116,
    "graphspec/registries.py:Held.__getitem__": 318952,
    "graphspec/registries.py:Held.__iter__": 92965,
    "graphspec/registries.py:Names.named": 157613,
    "graphspec/registries.py:Names.names": 92316,
    "graphspec/representations/groups.py:base.<locals>.found": 6638,
    "graphspec/schema/added.py:mapping": 14802,
    "graphspec/schema/added.py:paired": 5752,
    "graphspec/schema/declaration.py:EdgeType.__init__": 55541,
    "graphspec/schema/declaration.py:EdgeType.declaration": 28800,
    "graphspec/schema/declaration.py:Registry.attributes": 16206,
    "graphspec/schema/declaration.py:Registry.code": 14105,
    "graphspec/schema/declaration.py:Registry.codes": 54926,
    "graphspec/schema/declaration.py:Registry.group": 76518,
    "graphspec/schema/declaration.py:Registry.kind": 98165,
    "graphspec/schema/declaration.py:Registry.ordered": 6520,
    "graphspec/schema/schema.py:Schema.about": 7113,
    "graphspec/schema/schema.py:Schema.added_columns": 5571,
    "graphspec/schema/schema.py:Schema.declared": 25175,
    "graphspec/schema/schema.py:Schema.encoder": 7576,
    "graphspec/schema/schema.py:Schema.entities": 7943,
    "graphspec/schema/schema.py:Schema.kind": 21212,
    "graphspec/schema/schema.py:Schema.targets": 8680,
    "graphspec/schema/schema.py:fields": 23040,
    "graphspec/schema/schema.py:listed": 8843,
    "graphspec/schema/schema.py:named": 16822,
    "graphspec/schema/schema.py:typename": 8190,
    "graphspec/views/spans.py:encode": 5333,
    "tuning/bank/bank.py:Bank.digested": 76411,
    "tuning/bank/bank.py:Bank.joining": 19870,
    "tuning/bank/bank.py:Bank.several": 14951,
    "tuning/bank/bank.py:Bank.stated": 42219,
    "tuning/bank/bank.py:Bank.viewed": 14727,
    "tuning/batches/batching.py:moved": 5557,
    "tuning/batches/tensors.py:Features.__init__": 18744,
    "tuning/batches/tensors.py:Gather.__init__": 6248,
    "tuning/batches/tensors.py:Host.__init__": 14559,
    "tuning/batches/tensors.py:Resolver.grouped": 5515,
    "tuning/batches/tensors.py:Resolver.local": 5271,
    "tuning/batches/tensors.py:Resolver.numbers": 15010,
    "tuning/batches/tensors.py:Resolver.run": 72237,
    "tuning/batches/tensors.py:Resolver.spread": 6477,
    "tuning/batches/tensors.py:Resolver.table": 16401,
    "tuning/batches/tensors.py:array": 55397,
    "tuning/batches/tensors.py:features": 8923,
    "tuning/batches/tensors.py:joined": 18652,
    "tuning/batches/tensors.py:prepare": 71430,
    "tuning/families/affinity.py:AffinityNet.forward": 59,
    "tuning/families/event_graph.py:EventGraphNet.forward": 84,
    "tuning/families/layers.py:Asked.forward": 1046,
    "tuning/families/layers.py:CausalAttention.forward": 682,
    "tuning/families/layers.py:EvolvingLayer.forward": 4,
    "tuning/families/layers.py:GatedTCN.forward": 3,
    "tuning/families/layers.py:GraphCell.forward": 8,
    "tuning/families/layers.py:GraphCell.support": 16,
    "tuning/families/layers.py:Groups.__init__": 239,
    "tuning/families/layers.py:Head.forward": 1047,
    "tuning/families/layers.py:KeyEmbedding.forward": 64,
    "tuning/families/layers.py:MatrixGate.forward": 12,
    "tuning/families/layers.py:NodeHead.forward": 375,
    "tuning/families/layers.py:NumberHead.forward": 822,
    "tuning/families/layers.py:PathHead.forward": 103,
    "tuning/families/layers.py:References.forward": 1046,
    "tuning/families/layers.py:Relations.forward": 622,
    "tuning/families/layers.py:ScoreHead.forward": 21,
    "tuning/families/layers.py:Segments.__init__": 382,
    "tuning/families/layers.py:SelectiveSSM.forward": 8,
    "tuning/families/layers.py:SequenceHead.forward": 85,
    "tuning/families/layers.py:Spatial.forward": 521,
    "tuning/families/layers.py:TemporalEdges.forward": 224,
    "tuning/families/layers.py:TimeEncoding.forward": 1302,
    "tuning/families/layers.py:aggregate": 280,
    "tuning/families/layers.py:attribute_inputs": 143,
    "tuning/families/layers.py:chebyshev": 7,
    "tuning/families/layers.py:chebyshev.<locals>.laplacian": 17,
    "tuning/families/layers.py:masked_mean": 86,
    "tuning/families/layers.py:masked_weights": 466,
    "tuning/families/layers.py:mean": 1995,
    "tuning/families/layers.py:padded": 216,
    "tuning/families/layers.py:related": 172,
    "tuning/families/layers.py:segment_matmul": 760,
    "tuning/families/layers.py:segment_softmax": 16,
    "tuning/families/layers.py:segments": 382,
    "tuning/families/memory.py:Antisymmetric.matrix": 60,
    "tuning/families/memory.py:Antisymmetric.step": 60,
    "tuning/families/memory.py:Intensity.step": 4,
    "tuning/families/memory.py:MailboxGRU.prepare": 74,
    "tuning/families/memory.py:MailboxGRU.step": 330,
    "tuning/families/memory.py:MemoryNet.forward": 88,
    "tuning/families/memory.py:MemoryNet.forward.<locals>.clock": 342,
    "tuning/families/memory.py:MemoryNet.paired": 9,
    "tuning/families/memory.py:MemoryNet.readout": 88,
    "tuning/families/memory.py:MutualRNN.initial": 1,
    "tuning/families/memory.py:MutualRNN.read": 2,
    "tuning/families/memory.py:MutualRNN.step": 4,
    "tuning/families/memory.py:Rule.initial": 87,
    "tuning/families/memory.py:Rule.prepare": 14,
    "tuning/families/memory.py:Rule.read": 231,
    "tuning/families/memory.py:Rule.state": 223,
    "tuning/families/mixed.py:MessageLayer.forward": 138,
    "tuning/families/mixed.py:MixedNet.forward": 69,
    "tuning/families/model_spec.py:blocks": 47716,
    "tuning/families/model_spec.py:derived": 42351,
    "tuning/families/model_spec.py:groups": 45688,
    "tuning/families/model_spec.py:memory_state": 9254,
    "tuning/families/model_spec.py:unread": 87678,
    "tuning/families/model_spec.py:view": 71448,
    "tuning/families/periods.py:EndContext.forward": 135,
    "tuning/families/periods.py:enabled": 4507,
    "tuning/families/periods.py:gather_end": 11,
    "tuning/families/periods.py:lookback": 4892,
    "tuning/families/projection.py:ProjectionNet.forward": 80,
    "tuning/families/projection.py:ProjectionNet.structural": 80,
    "tuning/families/sequence.py:MixerLayer.forward": 2,
    "tuning/families/sequence.py:SequenceNet.forward": 83,
    "tuning/families/sequence.py:SequenceNet.mixer": 1,
    "tuning/families/sequence.py:SequenceNet.ssm": 2,
    "tuning/families/sequence.py:SequenceNet.transformer": 80,
    "tuning/families/signal.py:SignalNet.adjacency": 62,
    "tuning/families/signal.py:SignalNet.forward": 62,
    "tuning/families/snapshot.py:SnapshotNet.coupling": 28,
    "tuning/families/snapshot.py:SnapshotNet.entity_sequence": 201,
    "tuning/families/snapshot.py:SnapshotNet.forward": 244,
    "tuning/families/snapshot.py:SnapshotNet.framed": 30,
    "tuning/families/snapshot.py:SnapshotNet.graph_sequence": 15,
    "tuning/families/snapshot.py:SnapshotNet.observed": 244,
    "tuning/families/snapshot.py:SnapshotNet.previous": 18,
    "tuning/families/snapshot.py:SnapshotNet.referred": 244,
    "tuning/families/snapshot.py:SnapshotNet.values": 244,
    "tuning/families/tree.py:TreeNet.encode": 266,
    "tuning/families/tree.py:TreeNet.forward": 75,
    "tuning/families/walks.py:WalkNet.forward": 202,
    "tuning/families/walks.py:WalkNet.walked": 202,
    "tuning/searches/space.py:Space.draw.<locals>.choose": 10787,
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


def functions(tree):
    started = time.perf_counter()
    found, waiting = [], [(tree, "")]
    while waiting:
        node, prefix = waiting.pop()
        for child in (child for name in BODIES for child in getattr(node, name, ())):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found.append((prefix + child.name, child))
                waiting.append((child, prefix + child.name + ".<locals>."))
            elif isinstance(child, ast.ClassDef):
                waiting.append((child, prefix + child.name + "."))
            else:
                waiting.append((child, prefix))
    LOG.debug("functions exit found=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def told(statement):
    return isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) and called(statement.value.func) in LOGGERS


def returns(block, logged):
    for statement in block:
        if isinstance(statement, NESTED):
            continue
        if isinstance(statement, ast.Return) and not logged:
            return True, logged
        inner = logged or isinstance(statement, (ast.Try, ast.TryStar)) and any(map(told, statement.finalbody))
        held = [getattr(statement, name) for name in ("body", "orelse", "finalbody") if getattr(statement, name, None)]
        found = [returns(part, inner) for part in held + [entry.body for entry in getattr(statement, "handlers", ())]]
        found += [returns(entry.body, inner) for entry in getattr(statement, "cases", ())]
        if any(early for early, _ in found):
            return True, logged
        logged = inner or told(statement) or isinstance(statement, (ast.With, ast.AsyncWith)) and found[0][1]
    return False, logged


def logs(function):
    started = time.perf_counter()
    names, waiting = set(), [node for node in ast.iter_child_nodes(function) if not isinstance(node, NESTED)]
    while waiting:
        node = waiting.pop()
        if isinstance(node, ast.Call):
            names.add(called(node.func))
        waiting.extend(child for child in ast.iter_child_nodes(node) if not isinstance(child, NESTED))
    found = "perf_counter" in names and not names.isdisjoint(LOGGERS) and not returns(function.body, False)[0]
    LOG.debug("logs exit function=%s found=%s seconds=%.6f", function.name, found, time.perf_counter() - started)
    return found


def untimed(root, folders, listed):
    started = time.perf_counter()
    found, stale, held = [], [], set()
    for place, source in sources(root, folders):
        for name, function in functions(parsed(source)):
            key = "%s:%s" % (place, name)
            if compiled(function):
                continue
            timing = any(called(getattr(entry, "func", entry)) in MARKS for entry in function.decorator_list)
            timing = timing or logs(function)
            if key in listed:
                held.add(key)
                if timing:
                    stale.append(key)
            elif not timing:
                found.append(key)
    stale = sorted(stale + [key for key in listed if key not in held])
    LOG.debug("untimed exit found=%d stale=%d seconds=%.4f", len(found), len(stale), time.perf_counter() - started)
    return sorted(found), stale


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


def test_every_function_logs_its_time_or_is_compiled_or_is_listed_with_its_calls_per_run(tmp_path):
    assert untimed(ROOT, CODE, CALLED) == ([], [])
    clock = "import logging\nimport time\n\nLOG = logging.getLogger(__name__)\n\n\n"
    logged = "    started = time.perf_counter()\n    LOG.debug('work exit seconds=%.4f', time.perf_counter() - started)\n"
    write(tmp_path, "graphspec/plain.py", "def work():\n    return 1\n")
    write(tmp_path, "graphspec/constant.py", "VALUE = 1\n")
    write(tmp_path, "tuning/kernel.py", "from numba import njit\n\n\n@njit(cache=True)\ndef work():\n    return 1\n")
    write(tmp_path, "tuning/clocked.py", clock + "def work():\n" + logged)
    write(tmp_path, "tuning/deadline.py", clock + "def work():\n    return time.perf_counter() + 1\n")
    write(tmp_path, "tuning/told.py", clock + "def work():\n    LOG.info('work')\n")
    write(tmp_path, "tuning/marked.py", "from graphspec.infra.timing import timed\n\n\n@timed('work')\ndef work():\n    return 1\n")
    write(tmp_path, "tuning/staged.py", "@staged\ndef work(job, index):\n    return None\n")
    write(
        tmp_path,
        "tuning/measured.py",
        "from graphspec.infra.timing import measure\n\n\ndef work():\n    with measure('work'):\n        return 1\n",
    )
    write(
        tmp_path,
        "graphspec/nested.py",
        clock + "def work():\n    def inner():\n        return 1\n\n" + logged + "    return inner\n",
    )
    write(
        tmp_path,
        "graphspec/outer.py",
        clock + "def work():\n    def inner():\n" + logged.replace("    ", "        ") + "\n    return inner\n",
    )
    write(
        tmp_path,
        "graphspec/methods.py",
        clock + "class Held:\n    def __init__(self):\n" + logged.replace("    ", "        ") + "\n"
        "    def read(self):\n        return 1\n",
    )
    line = "LOG.debug('work exit seconds=%.4f', time.perf_counter() - started)\n"
    write(
        tmp_path,
        "tuning/early.py",
        clock + "def work(held):\n    started = time.perf_counter()\n    if held:\n        return 1\n    " + line + "    return 2\n",
    )
    write(
        tmp_path,
        "tuning/hit.py",
        clock + "def work(held):\n    if held:\n        started = time.perf_counter()\n        " + line + "    return 1\n",
    )
    write(
        tmp_path,
        "tuning/entered.py",
        clock + "def work(held):\n    started = time.perf_counter()\n    " + line + "    if held:\n        return 1\n    return 2\n",
    )
    write(
        tmp_path,
        "tuning/ended.py",
        clock + "def work(held):\n    started = time.perf_counter()\n    try:\n        if held:\n            return 1\n"
        "        return 2\n    finally:\n        " + line,
    )
    write(
        tmp_path,
        "tuning/within.py",
        clock + "def work(held):\n    with held:\n        started = time.perf_counter()\n        " + line + "    return 1\n",
    )
    write(tmp_path, "graphspec/remarked.py", "def work():\n    return 1  # perf_counter LOG.debug timed\n")
    write(tmp_path, "graphspec/quoted.py", "def work():\n    return 'time.perf_counter() LOG.debug( timed'\n")
    write(tmp_path, "graphspec/each.py", "def each():\n    return 1\n\n\ndef again():\n    return 2\n")
    listed = {"graphspec/each.py:each": 9000, "graphspec/each.py:gone": 9000, "tuning/marked.py:work": 9000}
    assert untimed(tmp_path, CODE, listed) == (
        [
            "graphspec/each.py:again",
            "graphspec/methods.py:Held.read",
            "graphspec/nested.py:work.<locals>.inner",
            "graphspec/outer.py:work",
            "graphspec/plain.py:work",
            "graphspec/quoted.py:work",
            "graphspec/remarked.py:work",
            "tuning/deadline.py:work",
            "tuning/early.py:work",
            "tuning/hit.py:work",
            "tuning/measured.py:work",
            "tuning/told.py:work",
        ],
        ["graphspec/each.py:gone", "tuning/marked.py:work"],
    )


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


def privates(root, places):
    started = time.perf_counter()
    found = set()
    for place in listed(root, places):
        source = (root / place).read_text(encoding="utf-8")
        if not UNDERSCORED.search(source):
            continue
        for node in ast.walk(parsed(source)):
            if isinstance(node, ast.Attribute) and not (isinstance(node.value, ast.Name) and node.value.id in ("self", "cls")):
                found.add(node.attr)
            elif isinstance(node, ast.Import):
                found.update(part for alias in node.names for part in alias.name.split("."))
            elif isinstance(node, ast.ImportFrom):
                found.update((node.module or "").split(".") + [alias.name for alias in node.names])
    found = sorted(name for name in found if name[:1] == "_" and name[:2] != "__")
    LOG.debug("privates exit found=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def pinned(root):
    started = time.perf_counter()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    found = {"python": project["requires-python"]}
    for line in project["dependencies"]:
        name = re.match(r"[A-Za-z0-9_.-]+", line).group(0)
        found[name] = line[len(name) :]
    LOG.debug("pinned exit dependencies=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def test_every_package_whose_private_names_the_code_reads_or_replaces_is_pinned_to_its_installed_version(tmp_path):
    assert privates(ROOT, PROJECT) == sorted({name for names in PRIVATE.values() for name in names} | set(PUBLIC))
    installed = {name: platform.python_version() if name == "python" else version(name) for name in PRIVATE}
    stated = {name: pinned(ROOT)[name] for name in PRIVATE}
    assert all(installed[name] in SpecifierSet(stated[name]) for name in PRIVATE)
    major, minor = map(int, platform.python_version_tuple()[:2])
    assert stated["python"] == ">=%d.%d,<%d.%d" % (major, minor, major, minor + 1)
    assert {name: held for name, held in stated.items() if name != "python"} == {
        name: "==" + held for name, held in installed.items() if name != "python"
    }
    write(tmp_path, "graphspec/reads.py", "import numba\n\n\ndef work(self, held):\n    return held._hidden, self._own, numba\n")
    write(tmp_path, "tuning/loads.py", "from numba.core import _internal\nimport numba._helper\n")
    assert privates(tmp_path, ("graphspec", "tuning")) == ["_helper", "_hidden", "_internal"]


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
