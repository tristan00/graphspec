import ast
import hashlib
import inspect
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
MARKS = ("timed",)
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
OPERATED = "graphspec/operations"
GROUPED = ("graphspec", "tuning")
REGISTRY = "Names"
REGISTERED = "REGISTRIES"
LOOKUP = "graphspec/registries.py"
WRITTEN = re.compile(r"^[ \t]*del .*$|\.held\.pop\(|\.held\[[^\]\n]*\][ \t]*=(?!=)", re.M)
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
    "graphspec/build/decisions.py",
    "graphspec/build/forms.py",
    "graphspec/build/generators.py",
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
    "graphspec.merge": (8, "cdc79c7a7bee0e6d9b599ed92f1308c0a644d576b2daff8f56554530a14d0b89"),
    "graphspec.difference": (5, "4d4f13f3cc158e0fddb9dde7b324af58d6f6d81be6b71bc9f1e8a670418a32d0"),
    "graphspec.nearby": (7, "1c0e2e4f13b4084875a9bd70032c220632b1f3dc72663896dc55e2074e01cec7"),
    "graphspec.consecutive": (7, "ad0ae0829b05de84542cac3c273fb7ef08582e00a0bc1cfe95c7c18dcd60f580"),
    "graphspec.neighbourhood": (6, "7feff8c7492860ef0019661bd2b46a47f2373daf01faf7e4974a71762ac2629c"),
    "graphspec.measures": (3, "738ad2944004dd94058b9877cbe44c1755ace68f2466b16d92affca51993aa88"),
    "graphspec.contract": (10, "be902bc3d6ae05296188837586c811d93ff96a7ba9168b1e487eaf4fad546927"),
    "graphspec.product": (5, "414e880458bb8727533225adab00a3c93108ef33291a5ce8f93c57d20b57b610"),
    "graphspec.negatives": (10, "b59a00bd88e9ee16c57d8d3b9a9067727aeef3e3e55c731dba7b79d6ba7f9277"),
    "derivation": (55, "bc11ddcfd7ca50bb7c7fcf7eae8d00209a798e68974730f64276a3de1398e263"),
    "views": (67, "ffaf1bd5914c84707372e709cd5bb0b23d55a29194d4e87e07fd73238be50800"),
    "representations": (18, "3a56b3ed965c44d4b9e9a96da8caa12baf06b8623d418b1d8bdc13037860d3e9"),
    "index": (6, "a6ad6b5034763801ebfb30ec58db4be41f482f0ff61717a64e76250e55ce4ec0"),
    "sort of reference edge": (1, "a1efbb0642fee3979ed3c7202d5dd2605ba1a0ed3d1d99a1761cb6a64734eb66"),
    "sort of reference graph": (1, "24c024a38804e098c970eb76d2c06bb0529d30901259df6aa5e098c9696ca51e"),
    "sort of reference node": (1, "5130e549da2053ab17d7ad3ddb45baaf74b38153c74af98f470f297b7c42489c"),
    "sort of reference nodes": (1, "c2845aac870668cb1a7ef9af24f5dcf258985a9025c5992adb89d32e82132ccb"),
    "sort of reference path": (1, "ef8b408a7f613b72f84a2cb7405fece8577b06b1af8af8d6bd349ffa280cf561"),
    "sort of reference subgraph": (1, "9206b9009931a59db36c911108cdccc89ed38d49d152154931b39a55434d36d5"),
    "rule for ties key": (1, "a4e0d0a26364c974ca8dc334460d78197a8c729db817355c5fb84b8646cebce8"),
    "rule for ties sequence": (1, "078803164ea4937fd4486327aa026370f61738c4f425568a025f9f4f9cccb4d6"),
    "choice among several facts first": (1, "31418db33e313293508900519a7abb04373c2213b67ecfbef4dfe10c67dcdc61"),
    "choice among several facts last": (1, "85fb73f00c8b8cfb028c0a5e3e27af47c1ad52b2433e9e1b3ee820ca4a70d4f3"),
    "rule for an edge about an edge lapse": (1, "6a9566d17407a138dc10fefca2e6ca1cccf008d3daeaaa0edafa1b83415dab83"),
    "rule for an edge about an edge stop": (1, "2f9c970e78c82a146854446dc2d6acbf9fe9467b0e01bc7f85dac6023739cb25"),
    "kind of target class": (1, "f78cf9068c67e8ce4efc13f3b57010d8450f194454102783e07cba4fd002ef0c"),
    "kind of target classes": (1, "79cb267fe0f48363992eba8d9303737c0616ceca794b14f7eb8d8719590bdaaa"),
    "kind of target decision": (3, "c3965a9b1c804c592171d89f950d7259ef14e3352c5becb562dc771358d03713"),
    "kind of target link": (1, "d5b0dff20b55cd1c6dcf81f4b867c0d6cad3d56787b05bb5d0e37cc521b8945d"),
    "kind of target nodes": (1, "9f79fa665dc9b89095fd97412bc0f701f399f4023a7e71db5ee268302fb6dd40"),
    "kind of target number": (1, "0433e59bfbc96aeb2fc5670d5860ee6552afa1441823330e5355564b821a3948"),
    "kind of target path": (1, "042b93e8dc67bca45549401e6ee082bf0bd8490b1d3ea363fe8f229f4ab405c3"),
    "kind of target proportions": (1, "d73466f5cd017eba6111a95921e8a4afccc98e7248f551580a7df18cc3da7373"),
    "kind of target ranking": (1, "fa3c8c64a1ad8a91d061c8c02961bfe1adb4e20cf894055f7f51af00c891f32f"),
    "kind of target sequence": (1, "b63cf67cf25d2753bfc3e794768fd085e5d285a41e9bba01fa9258076f15ed87"),
    "kind of target survival": (1, "c70b16063d22c5c97f05b65edc196fef6e9264d342a721cfb57c3a6635417d09"),
    "generator of options changed": (1, "5ef84be2f14edb817713aacffcc7e0efd5e70b65baaae8155793d9212bf37720"),
    "generator of options nearest": (1, "afab6a84f50ee779f9be060fb0590b28fc1bddaf25fe30226fcdb7b7905f5860"),
    "generator of options others": (1, "71b1606b253b6c95d1da28ae90e5e5a413e959858fb8785b650fe03a1e2e820e"),
    "generator of options ruled": (1, "20010619a3ae7fa03c0023276f696f45c00aadb0c919a909f21787974028984d"),
    "distance between states euclidean": (1, "e4e4459504ad3075425ece6c3027f370c629d6294b28762ae56fa38fe4c60211"),
    "column type category": (1,"e242885fbfd05f259fab475426d315185cfb3e691ebe8ddf2361297af9b53221"),
    "column type flag": (1, "35fcdebca490f610d7cfdc1a9aaf9b36e23519328df31803f32bb770a46c3f80"),
    "column type number": (1, "5cda0b56ca50b2e33d213c1a503ca6f7480e3e220ea4ed62f46b306058f5561b"),
    "encoder logarithm": (1, "0919b8ac7aeaa365625a2bc8b4ff2ef62d0b6e4f9d8b1c2c6dc3407ca65767df"),
    "encoder onehot": (1, "76925b82262bfb0b4e0dd341c341453836cbdd9ca88d87ab94c0fe55790eca77"),
    "encoder standard": (2, "9e75ea28a4f08757deab2b539c175c91c3296deaeabac2ae791cc270fb42623d"),
    "encoder value": (1, "6616774c8484dd0e63362850973de30dd5c90d40b2ce928c1d759c323c6dda3f"),
    "encoder of held graphs pooled": (1, "13ebcf0b70f33ce809dede5450330eb1cf15a15d85225eb8b935b48e90035f32"),
    "encoder of held graphs propagated": (1, "a93c9d323421eb2bcc6fd7bb54e5abd6790f147df313e142adcf14fb8c6054b2"),
    "rule of what a split may see without_scored": (1, "e1a7c29865951328158791db66ec25c031e56e994a0636fe97f1b899ab156bd2"),
    "rule of what a split may see without_targets": (1, "00cd86033d82aa67b9600420e183c6f51050179225e977c3b4cb37c838315564"),
    "choice of steps changes": (1, "9044190a1306a354b2777eb7ee8c7ba5c73979f2548900bad5280480d1278ec4"),
    "choice of steps tasks": (1, "f0cf89c005e77cc3e889830d43957f921a681611cc565c1deddbc2ca3e16ee75"),
    "form of storage of edges joining sets node": (1, "553cb32f29dd2d02a0a718e30ce2c9b97a111d35b72c853b27e4d138d1acff92"),
    "form of storage of edges joining sets pairs": (1, "cbe4ad8cc0f271143c76b8dc72f65ee44068724b91c7ac00779e824da3c5e85f"),
    "rule of negative sampling degree": (1, "3ee5c8bc7a9ca4599205d44ced998d1916092f98a21021340bd427fa9ba20084"),
    "rule of negative sampling hops": (1, "2195d9648d20bb9fefdf2b1175c67bfd0d5f0a25714f073383197a1f41b342cf"),
    "rule of negative sampling uniform": (1, "0e7031f7a595f5416bbd3e1b0fa4793e6a60aaf4c7a6b5962b60c07b9126f414"),
    "representation coupling": (1, "b52d854e14d862c835eb76aecc1c40aa43835299e8ee1338a26b73943842cbf0"),
    "representation events": (1, "5c6f88939ebd7bc42bfdf9c222b1db01226d2028881731819ca7c3f894dd1d11"),
    "representation mixed": (1, "b7d34f2ef59357ccf6e21a29411c518c1755dc65ceb26af077226893e6e0b933"),
    "representation neighbors": (1, "51b17a72ca7c7f338da213e8793578175b9595ec842d96fbed31ab18f1acd9f5"),
    "representation walks": (1, "62e70850779fece256ff02d8da3a783911e2e0f2ace971bb482d0a1b1be7feeb"),
}
CALLED = {
    "graphspec/budgets.py:GraphBudget.pair_limit": 16523,
    "graphspec/budgets.py:GraphBudget.pairs": 15047,
    "graphspec/budgets.py:GraphBudget.retain": 8674,
    "graphspec/budgets.py:GraphBudget.workspace": 9060,
    "graphspec/build/staged.py:Staged.positions": 6736,
    "graphspec/build/staged.py:numbers": 7586,
    "graphspec/engine/stitch.py:globalized": 6939,
    "graphspec/engine/transform.py:checked": 5114,
    "graphspec/facts/stored.py:Arrays.__getattr__": 505140,
    "graphspec/facts/stored.py:Arrays.__init__": 8874,
    "graphspec/facts/stored.py:Facts.__getattr__": 31659,
    "graphspec/infra/arrays.py:Bundle.hold": 47049,
    "graphspec/infra/arrays.py:Pending.__init__": 47049,
    "graphspec/infra/arrays.py:Resident.numpy": 73489,
    "graphspec/infra/arrays.py:Resident.tensor": 5149,
    "graphspec/infra/arrays.py:between": 7252,
    "graphspec/infra/arrays.py:compact": 47049,
    "graphspec/infra/files.py:Arrays.__getitem__": 17384,
    "graphspec/infra/files.py:Arrays.read": 8375,
    "graphspec/infra/files.py:aligned": 46317,
    "graphspec/infra/files.py:load_array": 26577,
    "graphspec/infra/files.py:save_array": 7088,
    "graphspec/infra/files.py:sha256_bytes": 5271,
    "graphspec/infra/files.py:sha256_json": 5251,
    "graphspec/infra/hashing.py:stacked": 7619,
    "graphspec/infra/log.py:JsonLog.format": 7,
    "graphspec/infra/timing.py:Timeline.measure": 43875,
    "graphspec/infra/timing.py:measure": 681191,
    "graphspec/infra/timing.py:timed": 23371,
    "graphspec/infra/timing.py:timed.<locals>.decorate": 23371,
    "graphspec/infra/workspace.py:allocate": 8870,
    "graphspec/infra/workspace.py:block_rows": 20755,
    "graphspec/infra/workspace.py:gathered": 21435,
    "graphspec/infra/workspace.py:held": 10102,
    "graphspec/infra/workspace.py:scatter": 14158,
    "graphspec/infra/workspace.py:take": 8027,
    "graphspec/operations/consecutive.py:selves": 12736,
    "graphspec/operations/derive.py:named": 5349,
    "graphspec/operations/edges.py:filtering": 24432,
    "graphspec/operations/edges.py:filters": 13344,
    "graphspec/operations/nearby.py:measured": 14480,
    "graphspec/operations/nodes.py:named": 5025,
    "graphspec/operations/operation.py:Operation.__call__": 5355,
    "graphspec/operations/operation.py:Operation.added": 5986,
    "graphspec/operations/operation.py:Operation.check": 5395,
    "graphspec/operations/operation.py:Operation.roles": 18490,
    "graphspec/operations/operation.py:Operation.settings": 22620,
    "graphspec/operations/operation.py:Operation.stated": 55927,
    "graphspec/operations/operation.py:Phase.once": 9020,
    "graphspec/operations/operation.py:Phase.run": 8568,
    "graphspec/operations/operation.py:Settings.__contains__": 139587,
    "graphspec/operations/operation.py:Settings.__getitem__": 188673,
    "graphspec/operations/operation.py:Settings.__init__": 22620,
    "graphspec/operations/operation.py:Settings.stated": 328260,
    "graphspec/operations/parameters.py:Parameter.__init__": 366790,
    "graphspec/operations/parameters.py:Parameter.active": 1154887,
    "graphspec/operations/parameters.py:Parameter.check": 556398,
    "graphspec/operations/parameters.py:Parameter.declaration": 50871,
    "graphspec/operations/parameters.py:Parameter.given": 18716,
    "graphspec/operations/parameters.py:Parameter.uniform": 36161,
    "graphspec/operations/parameters.py:bound_for": 443490,
    "graphspec/operations/parameters.py:canonical.<locals>.read": 21589,
    "graphspec/operations/parameters.py:choice": 36796,
    "graphspec/operations/parameters.py:differs": 16222,
    "graphspec/operations/parameters.py:draw": 8988,
    "graphspec/operations/parameters.py:enabled": 412159,
    "graphspec/operations/parameters.py:equals": 30012,
    "graphspec/operations/parameters.py:flag": 39977,
    "graphspec/operations/parameters.py:holds": 181121,
    "graphspec/operations/parameters.py:narrow": 5157,
    "graphspec/operations/parameters.py:number": 56253,
    "graphspec/operations/parameters.py:real": 22740,
    "graphspec/operations/paths.py:composed": 166358,
    "graphspec/operations/paths.py:kinds": 330422,
    "graphspec/operations/paths.py:reduced": 6532,
    "graphspec/operations/paths.py:written": 89921,
    "graphspec/operations/pipeline.py:Growing.before": 10389,
    "graphspec/operations/pipeline.py:Local.__contains__": 90038,
    "graphspec/operations/pipeline.py:Local.__getitem__": 68677,
    "graphspec/operations/pipeline.py:Local.__init__": 117441,
    "graphspec/operations/pipeline.py:Offered.allowed": 45001,
    "graphspec/operations/pipeline.py:Step.__call__": 5351,
    "graphspec/operations/pipeline.py:Step.__init__": 10420,
    "graphspec/operations/pipeline.py:called": 11446,
    "graphspec/operations/pipeline.py:checked": 6292,
    "graphspec/operations/pipeline.py:entered": 12441,
    "graphspec/operations/pipeline.py:given": 10389,
    "graphspec/operations/pipeline.py:made": 7701,
    "graphspec/operations/pipeline.py:narrowed": 32880,
    "graphspec/operations/pipeline.py:probed.<locals>.probe": 10912,
    "graphspec/operations/pipeline.py:probed.<locals>.spec": 26586,
    "graphspec/operations/pipeline.py:qualified": 207640,
    "graphspec/operations/pipeline.py:roled": 33438,
    "graphspec/operations/pipeline.py:settled": 29727,
    "graphspec/operations/sets.py:chosen": 14919,
    "graphspec/operations/state.py:Segment.named": 5407,
    "graphspec/operations/state.py:Table.__init__": 14442,
    "graphspec/operations/state.py:Table.find": 15697,
    "graphspec/operations/state.py:Table.lookup": 9315,
    "graphspec/operations/window.py:by_period": 6115,
    "graphspec/registries.py:Held.__getitem__": 318204,
    "graphspec/registries.py:Held.__iter__": 6153,
    "graphspec/registries.py:Names.named": 157518,
    "graphspec/registries.py:Names.names": 5580,
    "graphspec/representations/groups.py:base.<locals>.found": 6628,
    "graphspec/schema/added.py:mapping": 14760,
    "graphspec/schema/added.py:paired": 5752,
    "graphspec/schema/declaration.py:EdgeType.__init__": 55421,
    "graphspec/schema/declaration.py:EdgeType.declaration": 28646,
    "graphspec/schema/declaration.py:Registry.attributes": 16177,
    "graphspec/schema/declaration.py:Registry.code": 14104,
    "graphspec/schema/declaration.py:Registry.codes": 54806,
    "graphspec/schema/declaration.py:Registry.group": 76354,
    "graphspec/schema/declaration.py:Registry.kind": 98165,
    "graphspec/schema/declaration.py:Registry.ordered": 6505,
    "graphspec/schema/fields.py:fields": 23031,
    "graphspec/schema/fields.py:named": 16813,
    "graphspec/schema/schema.py:Schema.about": 7102,
    "graphspec/schema/schema.py:Schema.added_columns": 5561,
    "graphspec/schema/schema.py:Schema.declared": 25066,
    "graphspec/schema/schema.py:Schema.encoder": 7554,
    "graphspec/schema/schema.py:Schema.entities": 7935,
    "graphspec/schema/schema.py:Schema.kind": 21184,
    "graphspec/schema/schema.py:Schema.targets": 8676,
    "graphspec/schema/schema.py:listed": 8838,
    "graphspec/schema/schema.py:typename": 8190,
    "graphspec/views/spans.py:encode": 5332,
    "tuning/bank/bank.py:Bank.chosen": 42257,
    "tuning/bank/bank.py:Bank.digested": 76238,
    "tuning/bank/bank.py:Bank.joining": 19817,
    "tuning/bank/bank.py:Bank.several": 14919,
    "tuning/bank/bank.py:Bank.stated": 42256,
    "tuning/bank/bank.py:Bank.viewed": 14695,
    "tuning/batches/batching.py:moved": 5533,
    "tuning/batches/tensors.py:Features.__init__": 18675,
    "tuning/batches/tensors.py:Gather.__init__": 6216,
    "tuning/batches/tensors.py:Host.__init__": 14494,
    "tuning/batches/tensors.py:Resolver.grouped": 5515,
    "tuning/batches/tensors.py:Resolver.local": 5271,
    "tuning/batches/tensors.py:Resolver.numbers": 15010,
    "tuning/batches/tensors.py:Resolver.run": 72237,
    "tuning/batches/tensors.py:Resolver.spread": 6477,
    "tuning/batches/tensors.py:Resolver.table": 16401,
    "tuning/batches/tensors.py:array": 55172,
    "tuning/batches/tensors.py:features": 8887,
    "tuning/batches/tensors.py:joined": 18595,
    "tuning/batches/tensors.py:prepare": 71178,
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
    "tuning/families/model_spec.py:blocks": 6251,
    "tuning/families/model_spec.py:memory_state": 9258,
    "tuning/families/periods.py:EndContext.forward": 135,
    "tuning/families/periods.py:gather_end": 11,
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


def naming(root, folders, names):
    started = time.perf_counter()
    found = []
    for place, source in sources(root, folders):
        held = group(place)
        if held == OPERATED or OPERATED in below(held) or not any(name in source for name in names):
            continue
        for value, line in mentions(source)[1]:
            found.extend(
                (place, line, name)
                for name in names
                if re.search(r"(?<![\w.])%s(?!\w)" % re.escape(name), value)
            )
    LOG.debug("naming exit found=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return sorted(found)


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
            "tuning/staged.py:work",
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


def registered(root):
    started = time.perf_counter()
    trees = [parsed(source) for _, source in sources(root, CODE)]
    kinds = {REGISTRY} | {
        node.name
        for tree in trees
        for node in tree.body
        if isinstance(node, ast.ClassDef) and REGISTRY in map(called, node.bases)
    }
    found = {
        target.id
        for tree in trees
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and called(node.value.func) in kinds
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    LOG.debug("registered exit names=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def bypassing(root, places, names):
    started = time.perf_counter()
    found, reach = [], names | {REGISTERED}
    for place in listed(root, places):
        source = (root / place).read_text(encoding="utf-8")
        left = WRITTEN.sub("", source) if ".held" in source else ""
        if place == LOOKUP or ".held" not in left or not any(name in source for name in reach):
            continue
        looped, written, held = set(), set(), []
        for node in ast.walk(parsed(source)):
            if isinstance(node, (ast.For, ast.comprehension)) and reach & set(map(called, ast.walk(node.iter))):
                looped |= {target.id for target in ast.walk(node.target) if isinstance(target, ast.Name)}
            elif isinstance(node, ast.Subscript) and not isinstance(node.ctx, ast.Load):
                written.add(id(node.value))
            elif isinstance(node, ast.Attribute) and node.attr == "pop":
                written.add(id(node.value))
            elif isinstance(node, ast.Attribute) and node.attr == "held":
                held.append(node)
        found += [
            (place, node.lineno)
            for node in held
            if id(node) not in written
            and (called(node.value) in names or isinstance(node.value, ast.Name) and node.value.id in looped)
        ]
    LOG.debug("bypassing exit found=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return sorted(found)


def test_only_the_registries_read_what_a_registry_holds_and_other_code_reads_it_by_its_lookup(tmp_path):
    started = time.perf_counter()
    names = registered(ROOT)
    assert {"OPERATIONS", "REPRESENTATIONS", "NEGATIVES", "ENCODERS", "TIES", "MEASURES"} <= names
    assert bypassing(ROOT, PROJECT, names) == []
    write(tmp_path, LOOKUP, "def work(NEGATIVES):\n    return NEGATIVES.held['hops']\n")
    write(tmp_path, "graphspec/direct.py", "from graphspec import NEGATIVES\n\nFOUND = NEGATIVES.held['hops']\n")
    write(tmp_path, "tuning/copied.py", "from graphspec.schema import time\n\nCOPY = dict(time.TIES.held)\n")
    write(tmp_path, "tests/looped.py", "from graphspec import registries\n\nFOUND = [names.held for names in registries.REGISTRIES]\n")
    write(tmp_path, "tests/pair.py", "from graphspec import TIES, TYPES\n\nfor names in (TIES, TYPES):\n    COPY = dict(names.held)\n")
    write(
        tmp_path,
        "tests/written.py",
        "def work(NEGATIVES, entry, window, encoding):\n    NEGATIVES.held['parity'] = entry\n"
        "    del NEGATIVES.held['parity']\n    NEGATIVES.held.pop('parity', None)\n    return window.held, encoding.held\n",
    )
    assert bypassing(tmp_path, ("graphspec", "tuning", "tests"), names) == [
        ("graphspec/direct.py", 3),
        ("tests/looped.py", 3),
        ("tests/pair.py", 4),
        ("tuning/copied.py", 3),
    ]
    LOG.info("registries read by lookup exit seconds=%.3f", time.perf_counter() - started)


def test_every_module_sits_in_a_group_and_imports_only_the_groups_below_it(tmp_path):
    from graphspec import OPERATIONS

    assert layered(ROOT, CODE) == ([], [])
    names = tuple(OPERATIONS.names())
    assert naming(ROOT, CODE, names) == []
    write(tmp_path, "graphspec/schema/needs.py", "NEEDED = 'graphspec.negatives'\n")
    write(tmp_path, "graphspec/infra/hidden.py", "def work():\n    return 'graphspec.window/carried'\n")
    write(tmp_path, "graphspec/build/other.py", "NAMED = 'graphspec.negatives_of', 'mygraphspec.window'\n")
    write(tmp_path, "graphspec/engine/allowed.py", "NAMED = 'graphspec.negatives'\n")
    write(tmp_path, "tuning/searches/allowed.py", "NAMED = 'graphspec.contract'\n")
    assert naming(tmp_path, CODE, names) == [
        ("graphspec/infra/hidden.py", 2, "graphspec.window"),
        ("graphspec/schema/needs.py", 1, "graphspec.negatives"),
    ]
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


def code(entry):
    started = time.perf_counter()
    if inspect.isfunction(entry):
        found = inspect.getsource(entry)
    else:
        parts = [inspect.getsource(kind) for kind in type(entry).__mro__ if kind.__module__.split(".")[0] in CODE]
        parts += [
            name + "=" + (code(value) if callable(value) else repr(value)) for name, value in sorted(vars(entry).items())
        ]
        found = "\n".join(parts)
    LOG.debug("code exit characters=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found.replace("\r\n", "\n")


def shipped():
    from graphspec import OPERATIONS, registries

    started = time.perf_counter()
    found = {
        "%s %s" % (registry.label, name): registry.named(name)
        for registry in registries.REGISTRIES
        if registry.versioned and registry is not OPERATIONS
        for name in sorted(registry.shipped)
    }
    LOG.debug("shipped exit entries=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def unrecorded(digests, recorded, held):
    started = time.perf_counter()
    found = []
    for name, digest in digests.items():
        version = held[name]
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
    entries = shipped()
    digests = {name: digested(ROOT, places) for name, places in VERSIONED.items()}
    digests.update({name: hashlib.sha256(code(entry).encode()).hexdigest() for name, entry in entries.items()})
    held.update({name: entry.version for name, entry in entries.items()})
    assert sorted(digests) == sorted(RECORDED)
    assert unrecorded(digests, RECORDED, held) == []
    window = VERSIONED["graphspec.window"]
    source = (ROOT / window[0]).read_text(encoding="utf-8")
    write(tmp_path, window[0], source + "\nEDITED = 1\n")
    version = held["graphspec.window"]
    edited = {"graphspec.window": digested(tmp_path, window), "encoder standard": digests["column type number"]}
    assert unrecorded(edited, RECORDED, held) == [
        "the code of graphspec.window changed and its version %d did not; raise it" % version,
        "the code of encoder standard changed and its version %d did not; raise it" % held["encoder standard"],
    ]
    assert unrecorded(edited, RECORDED, dict(held, **{"graphspec.window": version + 1}))[0].startswith(
        "record graphspec.window at version %d with the digest " % (version + 1)
    )
