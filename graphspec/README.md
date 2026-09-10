**Graph spec core**

Domain-agnostic machinery for declaring a typed relational graph and the recipes a GNN search explores over it. No domain content lives here.

`declarations.py` holds `EdgeType`, `Modifier` and `Recipe`; `registry.py` holds `Registry`, which allocates codes, indexes types, validates the declaration and fingerprints it.

A domain module builds one `Registry` and re-exports what its own package needs. Two exist: `totalwar/spec.py` (Total War reward prediction) and `churn/spec.py` (Texas policy churn).

| Concept | Meaning |
|---|---|
| Node type | A kind of entity. Positions are assigned by declaration order and are never compared ordinally. |
| Edge type | Name, family, permitted source and target kinds, an attribute schema with a default, and whether it is symmetric. The registry allocates a canonical code and, for directed types, a reverse code. |
| Family | A group of edge types selected together by a recipe. |
| Recipe | A named graph: a view, a node-type set, a set of families and an optional lifted layout. |
| Layout group | A set of alternative base recipes a model may choose among; models sharing a group share its search coordinate. |
| Modifier | An optional family plus the base recipes it applies to and the search parameter it occupies. Must cover all or none of a layout group. |
| Marker | A reserved code above every relation, for inputs that are not relations. |
| Construction | Windows, lag sets, fanouts and seeds the builders read instead of literals. |

**Extraction bindings.** An edge type may declare how it is read from source records, so regular emitters iterate the spec rather than a literal table:

- `collection` with `subject_field` — one edge per entry of a nested collection.
- `field` — a boolean-gated edge, emitted when the named flag is true.
- `column` — a foreign-key column naming the target entity.

At most one binding per type. Irregular emitters stay hand-written; extraction is domain logic, not boilerplate.

**Invariants** are enforced at construction: unique names, every endpoint a declared node type, dense contiguous codes, defaults matching attribute schemas, layout groups and modifier bases naming declared recipes, and no partial layout-group coverage. `tests/test_graphspec_core.py` runs the same invariant suite against every domain.

**Fingerprint.** `declaration()` serialises everything including construction recipes and domain extras; `fingerprint()` hashes it. Pin that hash into any cache, plan or run built under the spec so an edit invalidates a bank rather than silently reinterpreting it.

**Domain and search.** `domain.py` binds a registry to a model roster: which graph each model consumes (`graphs`), which input blocks it needs (`blocks`), which arrays its view requires (`views`), and any orthogonal binary context switches (`switches`). `search.py` derives the exploration space from that — `choices`, `structure_metadata`, `structure` and `suggest_structure` — with no domain content. Both `totalwar` and `churn` build a `Domain`, and `tests/test_graphspec_core.py` runs the same search invariants against both: every declared design round-trips through `structure`, every proposal is a declared design, and switches are per-domain rather than baked in.

**Hard rule.** A spec may declare only edges derivable from frozen source data, deterministically and causally. Model-dependent edges would break the frozen-bank property and are not permitted.
