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

Extracted from tw_stack with history at commit 3e08d6e.
