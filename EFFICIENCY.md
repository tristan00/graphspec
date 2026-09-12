**Efficiency report**

Measured on this workstation (24 cores, 63 GB RAM, RTX 5090) on 12 September 2026. Before is commit `baffe4d`; after is `HEAD`. Every change was checked for equality against the previous implementation before it was kept: the transform equality harness (576 specs over 24 synthetic graphs), an old-versus-new comparison of every packed tensor for every model on the toy bank and the churn bank (118 cases, plus 3600 blocks in the adversarial review, no differences apart from the two dead keys removed), and CPU-versus-CUDA packing equality.

**Where the time went**

The real runs recorded under `artifacts/` before this work spent 3 to 20 times longer packing model inputs than computing on the GPU, and packing repeats every epoch. On the churn bank (8 graphs, up to 164k entities, 47,856 query rows, 37,617 training windows at history 16) one window packed in 16 ms for snapshot models and 1.3 s for event models, so no family could finish an epoch inside a 600 s fit. Variant materialisation, by contrast, took 2 to 4 s per graph. The packing path was therefore the first target, the dense graph operators the second, and the transform stages the third.

**Packing** (`tuning/data.py`, `tuning/gather.py`, `tuning/batching.py`)

Milliseconds per training window including batching, on CUDA, 60 windows each, churn bank.

| Model | Graph 0 (17k entities) before | after | Graph 4 (156k entities) before | after |
|---|---:|---:|---:|---:|
| M01 Relational snapshot GRU | 4.4 | 4.0 | 26.0 | 4.0 |
| M06 RGCN-GRU | 4.0 | 5.2 | 28.1 | 5.3 |
| M13 Entity-coupling gated | 4.8 | 3.6 | 35.3 | 4.3 |
| M15 Event-graph gated | 105.5 | 6.3 | 825.1 | 8.6 |
| M17 TGAT | 23.5 | 1.8 | 68.6 | 2.8 |
| M21 GraphMixer | 1.6 | 3.4 | 6.6 | 3.8 |
| M23 HyperEvent | 3.2 | 1.9 | 11.2 | 4.3 |
| M25 CAWN | 1.6 | 1.5 | 6.7 | 1.7 |
| M12 MPNN over derived nodes | 6.5 | 4.7 | 39.3 | 7.2 |
| M19 JODIE | 74.1 | 2.0 | 645.0 | 5.8 |

What changed: every per-entity or per-event Python loop became a numba kernel (latest observation, neighbour sampling, tree levels, interaction histories with structural features, reachability, walk identities); each open graph keeps the arrays that packing searches in RAM and its feature matrices on the training device, so feature rows are gathered on the device; DAG and mixed edges are selected by binary search on a sorted column instead of masking the whole table (34M rows on graph 4); searches use the array's own dtype, which stops numpy copying a 34M-row column to int64 on every call. Two packed keys read by no model, `common_x` and `history_events`, were dropped. The remaining cost is a floor of 2 to 5 ms of Python per window; packing a whole batch in one pass would be the next step.

**Dense operators** (`graphspec/operators.py`, `graphspec/kernels.py`)

Seconds for one transform of a single-step graph, the whole spec with the named operator.

| Operator | 4,000 entities before | after | 8,000 entities before | after |
|---|---:|---:|---:|---:|
| ppr, top 8 | 1.8 | 0.7 | 7.2 | 1.5 |
| heat, top 8 | 7.1 | 0.9 | 23.1 | 3.5 |
| fosr, budget 4 | 11.5 | 2.0 | 98.5 | 10.8 |
| sdrf, budget 4 | 23.6 | 2.8 | 140.5 | 9.9 |

What changed: connected components come from a union-find kernel; diffusion and the FoSR decomposition run per component in float64, on the GPU for components of at least 1024 entities; top-k uses a stable sort; FoSR reuses each component's decomposition until an added edge changes it; SDRF's balanced Forman curvature runs in a parallel numba kernel. Results are identical except at exact floating-point ties (score gaps of 1e-17 or an argmin over equal values), where either implementation's choice is arbitrary.

**Transform stages** (`graphspec/operators.py`, `graphspec/transform.py`)

Seconds for the full transform of the benchmark graphs (200k entities and 24 steps for large, 20k and 24 for medium), 119 cases; median speedup 1.33x, 76 cases faster, 4 within noise slower, no case changed status.

| Case | Graph | Before | After |
|---|---|---:|---:|
| identity | large | 4.1 | 0.6 |
| window 4 | large | 5.9 | 3.0 |
| degree | large | 5.0 | 2.4 |
| projection places | large | 10.5 | 6.9 |
| completion | medium | 16.8 | 10.7 |
| hops 2 | medium | 20.0 | 13.4 |
| coupling ordinal | large | 11.0 | 7.8 |
| mixed | large | 14.0 | 10.0 |
| events next | large | 61.5 | 65.0 |

What changed: a relation whose edge filters sit at their identity values is no longer filtered (2.7 s of the 4.1 s identity transform on large), and a stage that returns its input unchanged is not verified again. The event representation groups (`events next`, `events periods`, `events window`) did not change and dominate their cases; four cases (`hops 2` large, `hops 3` medium, `events periods` large, `events window capped` large) still exceed 16 GB because their output is that large (18M pairs per step for hops 2 on the large graph), before and after.

**Variants in the search** (`tuning/variants.py`, `tuning/run.py`)

Before a worker starts, the runner materialises the trial's variant for every graph in a process pool (`--workers`, default 4), so the fit budget is spent training; on the churn bank a variant of all 8 graphs takes about 30 s serially. After each trial it prunes the least recently used variants until `variants/` fits `--variant-gb` (default 300 GB). A variant build that fails is recorded as the trial's failure and the search continues. A per-stage cache keyed by spec prefix was not built: the stages that dominate (derived relations, edge filters, representations) are the last ones, so trials only share work when the whole spec matches, which the variant store already covers.

**Verification**

`tests/`: 22 tests in about 8 s warm, covering the spec space, every operator, the causal prefix property, seeds, protection, replay, representations, the kernels against numpy references, the bank and packing paths, and variant preparation and pruning. `D:\graph_tuning_infra`: 22 tests in 3 s including the boundary test. An adversarial review of the diff found and led to fixes for a pool chunk size that defeated parallel builds, unguarded variant failures, three fallbacks, dead code, missing timing logs, one README figure, silently ignored runtime flags on resume, and a rename race in the array store when two processes build the same variant.
