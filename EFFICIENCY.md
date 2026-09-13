**Efficiency report**

Measured on this workstation (24 cores, 63 GB RAM, RTX 5090) on 12 September 2026, on databases from `graphspec.synthetic`: `dev` (4 graphs, 2,000 entities, 48 steps), `medium` (20,000 entities, 24 steps, 375k facts) and `large` (200,000 entities, 24 steps, 3.4M observations, 3.8M facts), all with bounded degrees, persistent relations with removals, one query per step, two signal channels and a query-level target. Before is commit `baffe4d`; after is `HEAD`. Every change was checked for equality against the implementation it replaced: the transform equality harness, an old-versus-new comparison of every packed tensor for every model on two banks (120 cases here, 704 more in the independent review), and CPU-versus-CUDA packing equality. Client databases were removed from the machine; nothing below was measured on them.

**Separation**

Resource decisions live in `graph_tuning_infra` and nowhere else: `compute` chooses the device for dense linear algebra by matrix size, names the training device and thread count, reserves GPU memory and sets the budget of derived facts a variant may hold; `arrays` owns the resident cache of a graph's tensors and its eviction limit; `processes` owns the process-pool map and the default worker count; `files` owns the array store, its atomic rename and the size-bounded pruning of variant folders. `graphspec` imports no torch; `tuning` receives the device as an argument and holds no cache size, process count, disk budget or GPU fraction. The boundary test in the infra repository fails if it ever imports graphspec, tuning or dashboard.

**Where the time went**

Real runs recorded before this work spent 3 to 20 times longer packing model inputs than computing on the GPU, and packing repeats every epoch. Variant materialisation took seconds per graph. Packing was therefore the first target, the dense graph operators the second, and the transform stages the third.

**Packing** (`tuning/data.py`, `tuning/gather.py`, `tuning/batching.py`)

Milliseconds per training window including batching, on CUDA, 24 windows per case, history 16.

| Model | dev (2k entities) before | after | large (200k entities, 1.4M observations per window) before | after |
|---|---:|---:|---:|---:|
| M01 Relational snapshot GRU | 7.3 | 2.9 | 661 | 238 |
| M06 RGCN-GRU | 7.3 | 2.8 | 694 | 241 |
| M13 Entity-coupling gated | 7.7 | 3.6 | 898 | 344 |
| M15 Event-graph gated | 50.4 | 13.9 | 4,900 | 1,473 |
| M17 TGAT | 22.5 | 1.5 | 49.2 | 4.7 |
| M21 GraphMixer | 0.8 | 1.7 | 29.9 | 4.1 |
| M23 HyperEvent | 1.9 | 1.6 | 30.4 | 5.0 |
| M25 CAWN | 0.8 | 0.9 | 40.1 | 4.4 |
| M12 MPNN over derived nodes | 45.0 | 12.3 | 3,700 | 1,708 |
| M19 JODIE | 31.3 | 2.4 | 2,885 | 144 |

What changed: every per-entity or per-event Python loop became a numba kernel (latest observation, neighbour sampling, tree levels, interaction histories with their structural features, reachability, walk identities); each open graph keeps the arrays that packing searches in RAM and its feature matrices on the training device, so feature rows are gathered on the device; DAG and mixed edges are selected by binary search on a sorted column instead of masking the whole table; frame edges are sliced by per-step bounds instead of sixteen scans; mixed edge features and DAG gaps are computed on the device; searches use the array's own dtype, which stops numpy copying a whole column to int64 on every call. Two packed keys read by no model, `common_x` and `history_events`, were dropped. What remains on the large graph is proportional to the data a window carries: 26M DAG edges and 1.4M observation rows per window for the event and snapshot families. A fixed floor of 1 to 3 ms of Python per window remains; packing a whole batch in one pass would be the next step.

**Dense operators** (`graphspec/operators.py`, `graphspec/kernels.py`, `graph_tuning_infra/compute.py`)

Seconds for one transform of the dev graph (48 steps of about 1,600 present entities) with the named operator.

| Operator | Before | After |
|---|---:|---:|
| ppr, top 8 | 19.4 | 1.0 |
| heat, top 8 | 26.3 | 1.1 |
| fosr, budget 4 | 33.7 | 1.8 |
| sdrf, budget 4 | 5.8 | 2.5 |

What changed: connected components come from a union-find kernel; diffusion and the FoSR decomposition run per component in float64, on the device the infra layer picks by component size; top-k uses a stable sort; FoSR reuses each component's decomposition until an added edge changes it; SDRF's balanced Forman curvature runs in a parallel numba kernel. Results equal the previous implementation to float64 rounding; they differ only where two candidates tie to within 1e-17, where either choice is arbitrary.

**Transform stages** (`graphspec/operators.py`, `graphspec/transform.py`)

Seconds for the full transform, 107 cases; median speedup 1.33x, 81 cases faster, one within noise slower, every case completes on both sides. Peak memory after includes about 0.5 GB for torch loaded in the measuring process.

| Case | Graph | Before | After |
|---|---|---:|---:|
| identity | large | 0.9 | 0.3 |
| window 4 | large | 2.1 | 1.4 |
| degree | large | 1.3 | 0.9 |
| hops 2 | large | 7.7 | 5.4 |
| completion of teams | medium | 14.0 | 8.6 |
| coupling ordinal | large | 3.8 | 3.1 |
| walks | large | 4.1 | 3.5 |
| events periods | large | 53.5 | 49.2 |
| mixed | large | 10.7 | 10.1 |

What changed: a relation whose edge filters sit at their identity values is no longer filtered, and a stage that returns its input unchanged is not verified again. The event and mixed representation groups were not rewritten and dominate their cases. The earlier out-of-memory results were artefacts of the previous benchmark graphs, which used preferential attachment and so held hubs of thousands of neighbours; on bounded-degree graphs two-hop closure on 200k entities takes 5.4 s at 2.3 GB. One legal spec can still run away: completing a large node type into a clique produced 768M facts and 46 GB on the medium graph. Derivation now stops at the infra fact budget (200M facts) with a clear error, so such a trial fails instead of exhausting the machine.

**Variants in the search** (`tuning/variants.py`, `tuning/run.py`)

Before a worker starts, the runner materialises the trial's variant for every graph in a process pool and prunes the least recently used variants until `variants/` fits the disk budget, both with defaults from the infra layer (`--workers`, `--variant-gb`). A variant build that fails, including one that exceeds the fact budget, is recorded as the trial's failure and the search continues. A per-stage cache keyed by spec prefix was not built: the stages that dominate are the last ones, so trials only share work when the whole spec matches, which the variant store already covers.

**Verification**

`tests/`: 24 tests in about 14 s, covering the spec space, every operator, the causal prefix property, seeds, protection, replay, representations, the kernels against numpy references, the bank and packing paths, variant preparation and pruning, the fact budget, and generated databases at all three target levels. `graph_tuning_infra`: 27 tests in 4 s including the boundary test. Two adversarial reviews of the diffs each ran their own equality checks and found issues that were fixed: a pool chunk size that defeated parallel builds, unguarded variant failures, fallbacks and dead code, missing timing logs, a rename race in the array store, ignored flags on resume, residual policy in the logic layer, a generator that produced unbuildable label schedules, and a misleading benchmark case.
