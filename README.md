**graphspec**

A declared graph specification and a search over temporal graph model designs, with the
domain supplied rather than hardcoded.

- `graphspec/` — the domain-free core. Node and edge type declarations, code allocation,
  endpoint validation, per-type attributes, construction recipes, modifiers, fingerprints
  (`registry.py`, `declarations.py`), the domain binding (`domain.py`) and the search space
  derived from it (`search.py`).
- `totalwar/` — the Total War reward-prediction domain, plus the cache builder, data packer,
  model roster and search runner.
- `churn/` — the Texas policy churn domain.
- `dashboard/` — results server for a running search.
- `tests/` — invariants, run against both domains.

The Total War data path is the one thing this repo cannot do alone: `totalwar/tw_source.py`
imports the tw_stack corpus reader lazily, so building a Total War cache needs that checkout and
its database. Everything else — the spec, the search, the models and the whole test suite — runs
standalone. The fixed CatBoost baseline lives in tw_stack as `research/graph_baseline.py`; this
repo reads the `baseline.json` it writes but cannot produce it.

History was path-filtered out of tw_stack, so no commit hash is shared with it. The extraction
point was tw_stack commit 3e08d6e, which exists in that repo, not this one.

Install: `pip install -r requirements.txt -r requirements-test.txt`, then `pytest`.
