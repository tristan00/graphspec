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

What changed: every per-entity or per-event Python loop became a numba kernel (latest observation, neighbour sampling, tree levels, interaction histories with their structural features, reachability, walk identities); each open graph keeps the arrays that packing searches in RAM and its feature matrices on the training device, so feature rows are gathered on the device; DAG and mixed edges are selected by binary search on a sorted column instead of masking the whole table; frame edges are sliced by per-step bounds instead of sixteen scans; mixed edge features and DAG gaps are computed on the device; searches use the array's own dtype, which stops numpy copying a whole column to int64 on every call. Two packed keys read by no model, `common_x` and `history_events`, were dropped. Every small array of a window is then bundled into one device transfer (`graph_tuning_infra.arrays.Bundle`), so a window costs a handful of copies instead of about fifty; the fixed cost per window is about 1 ms. What remains on the large graph is proportional to the data a window carries: 26M DAG edges and 1.4M observation rows per window for the event and snapshot families.

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

Seconds for the full transform, 107 cases; median speedup 1.40x, 79 cases faster, one within noise slower, every case completes on both sides. Peak memory after includes about 0.5 GB for torch loaded in the measuring process.

| Case | Graph | Before | After |
|---|---|---:|---:|
| identity | large | 0.9 | 0.3 |
| window 4 | large | 2.1 | 1.4 |
| degree | large | 1.3 | 0.9 |
| hops 2 | large | 7.7 | 5.4 |
| completion of teams | medium | 14.0 | 8.3 |
| coupling ordinal | large | 3.8 | 3.3 |
| coupling periods | large | 10.8 | 8.5 |
| walks | large | 4.1 | 3.6 |
| events next | large | 8.4 | 5.2 |
| events window capped | large | 19.2 | 9.2 |
| events periods | large | 53.5 | 12.9 |
| mixed | large | 10.7 | 10.9 |

What changed: a relation whose edge filters sit at their identity values is no longer filtered, and a stage that returns its input unchanged is not verified again. In the event group the boundary links are enumerated in two parallel passes with an analytic duplicate rule instead of a growing hash table, the DAG links in two parallel passes per event, the 49M output rows are ordered by a parallel bucket sort instead of `np.lexsort`, and their validation runs in a kernel; the group fell from 20.6 s to 4.0 s on the large graph with identical rows. The mixed group was not rewritten and is bound by its 7.5M derived nodes. The earlier out-of-memory results were artefacts of the previous benchmark graphs, which used preferential attachment and so held hubs of thousands of neighbours; on bounded-degree graphs two-hop closure on 200k entities takes 5.4 s at 2.3 GB. One legal spec can still run away: completing a large node type into a clique produced 768M facts and 46 GB on the medium graph. Derivation now stops at the infra fact budget (200M facts) with a clear error, so such a trial fails instead of exhausting the machine.

**Variants in the search** (`tuning/variants.py`, `tuning/run.py`)

Before a worker starts, the runner materialises the trial's variant for every graph in a process pool and prunes the least recently used variants until `variants/` fits the disk budget, both with defaults from the infra layer (`--workers`, `--variant-gb`). A variant build that fails, including one that exceeds the fact budget, is recorded as the trial's failure and the search continues. A per-stage cache keyed by spec prefix was not built: the stages that dominate are the last ones, so trials only share work when the whole spec matches, which the variant store already covers.

**Verification**

`tests/`: 24 tests in about 14 s, covering the spec space, every operator, the causal prefix property, seeds, protection, replay, representations, the kernels against numpy references, the bank and packing paths, variant preparation and pruning, the fact budget, and generated databases at all three target levels. `graph_tuning_infra`: 28 tests in 4 s including the boundary test. Two adversarial reviews of the diffs each ran their own equality checks and found issues that were fixed: a pool chunk size that defeated parallel builds, unguarded variant failures, fallbacks and dead code, missing timing logs, a rename race in the array store, ignored flags on resume, residual policy in the logic layer, a generator that produced unbuildable label schedules, and a misleading benchmark case.

**13 September 2026: the Total War bank**

Measured on the same workstation on the 100-campaign Total War bank (10,109 steps, 18.5M observations, 56.5M facts, 602 relation codes, 40 attributed relations, 128 features), history 16, 8,916 training windows in 272 batches per epoch. Before is commit `4358652`; after is `HEAD`. Equality was checked as before: the 69 preset model configurations, on the synthetic dev bank and a ten-campaign Total War bank, with and without turn context, pack bit-identical tensors and reproduce the reference forward outputs and gradients; the two antisymmetric memory presets differ in gradient by the same amount they differ from themselves between two runs on identical code.

**Where the time went**

Per batch of about 30 windows in steady state, before: the snapshot view packed in 1,320 ms and computed in 480 ms; the event graph packed in 57 ms and computed in 21 ms; the mixed view packed in 161 ms. Three causes. The attribute gather searched a sorted relation column with a query of another dtype, so numpy cast the whole column on each of the 80 calls per window: 27 of the snapshot view's 48 ms per window. The relational layer looped over the relation kinds present, copying the full edge-message tensor out of place at every kind and again for every attributed relation: roughly 20 GB of memory traffic per layer per batch. And packing ran on one core between GPU steps, re-running every epoch.

**What changed**

- The attribute gather searches once for all relation codes in the column's own dtype (`tuning/data.py`).
- The relational layer gathers the source rows once in relation order, splits them per kind, multiplies each slice by its kind's matrix, and accumulates into a node buffer with one `index_add_`; nothing is copied per kind (`tuning/layers.py`). The align gather uses `index_select` so its backward is an index-add rather than a sort.
- Packing produces a recipe: numpy arrays plus markers for the device work. `Features` names a resident matrix, row indices and per-row weights, so a masked gather, the mean of two rows, or the average of an observation and a query resolve as one device expression; `Frames` names the per-step edge ranges and the device slices the batch's edge table instead of receiving sixteen restated copies; `MixedEdges` sends a code and a gap per edge instead of sixteen floats. `recipe_batch` concatenates windows in numpy; `materialise` bundles every array of the batch into one buffer and resolves the markers on the device (`tuning/data.py`, `tuning/batching.py`, `tuning/turn_inputs.py`).
- `packing.Packer` runs the recipes in `--workers` spawned processes that share the machine's cores between their numba kernels and hand each bundled batch to the training process through shared memory; the training process copies it through pinned memory and resolves it. The bundle stores every int64 array whose values fit as int32 and widens it on the device (`graph_tuning_infra.arrays`).
- The trial budget is the runner's clock from parameter selection to the pruning after the worker exits, with the variant build inside it; no extension (`tuning/run.py`, `tuning/budget.py`).

**Results**

Second-epoch seconds per view (two-epoch fits, 8 packing processes): signal 2, walks 4, sequence 6, tree 7, event graph 9, projection 12, affinity 16, mixed 26, memory 33, snapshot 53. The runner's measurement of the same views before this work, on the first build of the bank: sequence 36, event graph 120, memory 254, snapshot about 560. The snapshot batch that computed in 480 ms computes in 92 ms; its main-process cost per batch is now 137 ms of transfer for 182 MB and 58 ms of device gathers, against a worker wait under 20 ms. The bundle of a snapshot batch fell from 397 MB to 182 MB and of a mixed batch from 231 MB to 49 MB.

**Data**

The first build of the Total War database carried NaN in the `side` attribute of every present `fought_with` fact of nine campaigns; the builder writes only 0 or 1 there and a rebuild from the corpus reproduced neither. `graphspec.truth.parse` now refuses a present fact whose attribute is not finite, so a bank cannot be built from such a database; the search runs on the rebuilt database, verified against the corpus on 2,884 checks.

**Rewiring**

Five of the first twenty trials of the search timed out inside the variant build. The cause was `derive/rewiring` with the SDRF curvature: for every candidate edge the balanced Forman curvature was recomputed by a kernel that scans every node of the component twice, about 80 µs a call, and a step with two well-connected endpoints offers tens of thousands of candidates; a graph of 80 steps took 375 s in that stage. The curvature now runs over neighbour lists with the candidate edge applied virtually (`kernels.curvature_with`), which visits the same nodes the dense scan tested and so returns the same integers and the same floats: identical values on 2,446 candidate evaluations against the old kernel. The same graph builds in 6.9 s.

**Loading**

The 100-campaign database build spent 59 of its 62 minutes copying relation tables into LadybugDB, and two tables set the pace: `edge_skill_link` at 5,300 rows a second and `edge_action_result` at 3,300, against 61,000 for the larger `edge_grants` and 276,000 for observations. At 10 campaigns the same tables copied at over a million rows a second, at 40 campaigns skill links fell to 6,500; the rate collapses with the table's size, not with the row, and chunked copies and pre-sorted rows changed nothing. Three things were done. The loader emitted every skill-tree link once per skill set that contained it, up to eleven identical facts per step; identical catalogue facts are now stated once, which halves that table. Relation tables are declared with forward-only storage, which halves the copy at every scale. And the bank no longer reads the database at all: `graphspec.staged` reads the staged parquet tables, partitions them by graph in one pass (the partition is saved beside the staging and reused by every worker), and serves per-graph tables that match the database reader row for row; `tuning.build` opens whichever source the task names. On 10 campaigns the bank builds in 3.0 s from the staging against 6.5 s from the database, and the copy step, 41 s there and hours at 1,000 campaigns, is no longer on the path. The comparison also showed LadybugDB returning 0.0 for 7,694 attribute cells the loader wrote as NaN on absent facts, the same class of corruption that put NaN into present `fought_with` facts of nine campaigns on the first build; the parquet path carries the loader's values exactly.

**14 September 2026: the 1,000-campaign bank**

Measured on the same workstation on the 1,000-campaign Total War bank (74,486 steps, 117.7M observations, 329.8M facts), history 8, hidden 32, one epoch of 1,019 batches with 8 packing processes, each fit alone on the machine. Before is commit `fb76773`; after is `HEAD`. Packed batches were checked for bit-identity against the previous code: 108 batches over twelve view configurations on the synthetic dev bank, the 100-campaign bank and this bank, with and without turn context, and the one-epoch fits reproduce the previous loss and R² exactly.

**Where the time went**

An event-graph epoch took 91 s: 11 s of GPU compute, 80 s in the training process. The packing processes were waited on for 6 s of that. The training process spent 33 s bringing each graph's feature matrix to the device through a memory-map copy at about 1 GB/s, and 57 s resolving batches, because every window of a batch was gathered on its own and every int32 index widened on its own: about 2,000 kernel launches per batch of 62 windows. Each trial also built its graph variant for 1,000 graphs in 160 to 185 s: the pool never capped numba's threads, so every process spawned a thread per core, the validation compared 594 relation codes against the whole edge table, and every variant wrote a fresh copy of the 27 MB feature matrix per graph.

**What changed**

- `resolve` works per batch: the resident matrices of the batch's graphs are concatenated into one table per key, each window's indices are shifted by its graph's offset, and one gather serves every window; indices travel as int32 and are used as such (`tuning/data.py`). Resolving a resident batch fell from 41 ms to 4.5 ms.
- The resident cache reads arrays with direct file reads instead of memory-map copies, pins them and copies them over a side stream, holds 192 graphs and evicts the one loaded first, which is what a chunked schedule with a prefetcher needs; a loader thread in the training process walks the batch schedule ahead of training and loads coming graphs, throttled so that no graph still needed is evicted (`graph_tuning_infra/arrays.py`, `tuning/packing.py`).
- The variant pool gives each process its share of the cores; the validation slices a code-sorted edge table; a variant's feature and query matrices are hard links to the truth's files when the stages leave them unchanged; the source digest ignores line endings, which git had been flipping (`tuning/variants.py`, `graphspec/representations.py`, `graph_tuning_infra/files.py`, `graph_tuning_infra/sources.py`).

**Results**

Seconds per epoch, one epoch, history 8, hidden 32: event graph 91 → 26, tree 84 → 26, sequence 53 → 19; the other views were not measured on this bank before this work: signal 15, walks 23, projection 31, affinity 35, memory 79 (64 s of it GPU compute), mixed 99 (61 s GPU compute), snapshot 187 (81 s GPU compute and 64 s copying its 180 MB batches out of shared memory). The event-graph training process now spends 2 s copying, 3 s transferring and 7 s resolving per epoch beside 12 s of GPU compute. The variant build for 1,000 graphs fell from 184 s to 52 s wall on 8 processes and writes 19 GB instead of 46 GB. Every source change still rebuilds the variants of the trials that follow it.

**14 September 2026: the packer's transport and the prune step**

Measured on the same workstation and bank, history 8, hidden 32, one epoch with 8 packing processes, each fit alone on the machine. Before is commit `948ae6b`; after is `HEAD`. Every view's packed batches were checked for bit-identity against the single-process `materialise` path on the synthetic dev bank and the 10-campaign bank, twenty batches per view over four slots, so every slot was reused and regrown.

**Where the time went**

The snapshot epoch spent 64 of its 187 s in the training process copying batches: a worker concatenated its bundle into a fresh array, `torch.multiprocessing` copied that into a fresh shared-memory section for the queue, the training process faulted that section in while copying it into pinned memory, and only then moved it to the device. Four passes over 147 MB per batch, 1,198 batches, under memory contention from eight processes doing the same. The runner's prune step walked every file of the variant store after every trial, 19 s at ten variants, and deleted a 19 GB variant inside the trial budget, about 100 s each time.

**What changed**

- Each worker writes its bundle straight into one of a fixed set of shared-memory slots, two per worker, that the training process registers with CUDA (`cudaHostRegister`) and keeps for the whole fit; a slot grows to a quarter more than the largest batch it has seen. A batch is one device copy from the slot, 164 GB in 5.3 s over the epoch, and the slot is free again as soon as that copy returns. Two 180 MB copies per batch are gone from the worker and one from the training process (`graph_tuning_infra/arrays.py`, `tuning/packing.py`).
- Pruning sizes a variant by a `scandir` walk only when its entry count changes, renames a doomed variant into `trash/` beside the store and deletes it from a daemon thread; the trial clock sees the rename (`graph_tuning_infra/files.py`).
- A relational spatial layer given a batch without edges returns zero messages instead of raising in `torch.cat` (`tuning/layers.py`); the seventh trial of the 1,000-campaign search failed on such a batch.

**Results**

Snapshot epoch 187 s → 111 s; the training process's time waiting on packed batches fell from 105 s to 27 s per epoch (5 s of device copies, 13 s resolving markers, 9 s waiting on workers), against 83 s of GPU compute, so the view is now compute bound. The holdout score is unchanged (R² 0.232; the loss agrees to nine digits, the residual being the CUDA `index_add_` atomics). Prune sizing of the 16-variant, 226 GB store fell from 19 s to 4.5 s on a cold cache and to milliseconds on a warm one; deletion no longer counts against the trial. The configuration of the trial that failed on an edgeless batch now completes in 227 s of its 600 s budget with R² 0.482. Forty snapshot batches of this bank streamed through 16 slots are bit-identical to the single-process path.

**Verification**

`tests/`: 9 tests in 6 s, covering the bundle layout against the packed buffer, slot transfer and growth, the packer against `materialise` for the snapshot and event-graph views on a generated bank, prune's trash and sweeper, the size cache and the stamp check, and the relational layer with and without edges.
