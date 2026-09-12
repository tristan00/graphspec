**Model search**

`tuning` turns a graph database into a frozen bank of truth graphs, then searches graph-transform specs and temporal graph models over it on a CUDA GPU. Models and the search read structure from the bank's pinned declaration. No code here names a node type, relation or attribute. The database contract and task file are in the [root README](../README.md), and the transforms in [graphspec/README.md](../graphspec/README.md).

**Bank**

`python -m tuning.build` reads the database once and writes each graph's truth arrays under `truth/`. Training never touches the database. Labels live in `supervision.json`, apart from the model inputs. `Bank` refuses a manifest from another bank version or whose declaration does not reproduce its fingerprint.

A trial's configuration carries its spec under `graph`. `Bank.arrays` canonicalises the spec against the model's representation groups, then materialises the variant of a graph on first use under `variants/<digest>/graph_NNNN/`. The runner materialises every graph of a trial's variant in parallel before the worker starts (`--workers`) and prunes the least recently used variants to the `--variant-gb` budget, so the fit budget is spent on training. The digest covers the spec without the parameters read only when packing (`time/history`, `time/context`, `neighbors/order`, `neighbors/decay`), the groups and the `graphspec` and `graph_tuning_infra` sources, so trials that differ only in those share a variant until the transform code changes. It memory-maps the arrays and keeps the eight most recent graphs open; each open graph also holds the arrays that packing searches in RAM and its feature matrices on the training device (`Bank.resident`).

**Packing**

`pack` turns one training window into model inputs. Every per-entity or per-event loop lives in a numba kernel in `gather.py`: the latest observation of an entity before a tick, neighbour sampling (recent, uniform or decay-weighted Gumbel top-k keyed by the seeded hash), the levels of a temporal-neighbour tree, interaction histories with their structural features, the events reachable from a prediction's entities, and walk identities. Feature rows are gathered on the device from the resident matrices, and edge tables are cut by binary search on their sorted columns. A block of graph 4 of the churn bank (156k entities, 9,441 steps) packs in 1 to 8 ms on the GPU for every family.

**Models**

Each family reads the base arrays plus the representation groups listed; only those groups' parameters enter its spec space. Each is a compact version adapted to regression on a target. Entity numbers are graph-local correspondence keys; no model learns an embedding per entity id.

| IDs | Version | Groups | Defining computation |
|---|---|---|---|
| M01–M03 | Relational snapshot GRU / LSTM / Transformer | — | Typed state encoder, an actor/query feedback round, then a sequence of graph and query embeddings through the respective temporal operator. |
| M04 | GCN-GRU | — | Normalized spatial aggregation followed by aligned per-entity recurrent state. |
| M05 | GraphSAGE-GRU | — | Separate root and mean-neighbor transforms followed by aligned recurrence. |
| M06–M07 | RGCN-GRU / RGCN-TCN | — | Relation-specific weight matrices, relation attributes, and recurrent or gated dilated convolutional time processing. |
| M08 | T-GCN | — | Graph convolution inside reset, retention and candidate gates. |
| M09 | DCRNN | — | Separate canonical forward and reverse diffusion supports inside the recurrent gates. |
| M10 | EvolveGCN-H | — | Top-k summaries of observed nodes update graph-convolution weight matrices recurrently. |
| M11 | DySAT | — | Structural attention followed by masked per-entity temporal attention. |
| M12 | MPNN over derived nodes | mixed | The message-passing network in `message_net.py`, with graph-layer GRU updates, over the mixed graph. |
| M13–M14 | Entity-coupling gated / attention | coupling | Spatial encoding, lag- and relation-aware propagation over coupling and boundary edges between dated entity copies, then spatial propagation. |
| M15–M16 | Event-graph gated / attention | events | Explicit event edges with initial feature anchors, causal lag features and a participant-conditioned query readout. |
| M17 | TGAT | neighbors | Recursive temporal-neighbor attention, learnable time encoding and historical feature lookup at each recursive cutoff. |
| M18 | Typed TGN | neighbors | Typed messages, simultaneous-group mailboxes, GRU memory updates and temporal neighborhood embedding. Prediction precedes the current query's memory update. |
| M19 | JODIE | — | Separate actor and subject recurrent states, coupled updates and elapsed-time projection. New observations refresh side features without erasing recurrent memory. Offered only with queries. |
| M20 | DyRep | neighbors | Separate association and interaction processes, learned intensities, current-topology attention and neighborhood-dependent memory. Regression replaces the point-process objective. |
| M21 | GraphMixer | neighbors | Masked token and channel mixers over interaction histories, plus a separate mean-neighbor encoder. |
| M22 | DyGFormer | neighbors | Paired endpoint histories, cross-history occurrence counts, patch projection and joint Transformer processing. |
| M23 | HyperEvent | neighbors | Twelve pair-relative structural features spanning direct occurrence and one- and two-hop overlap, followed by a Transformer. |
| M24 | TPNet | neighbors | Seeded Gaussian projections of temporal walk matrices, order-dependent exponential decay and simultaneous-group propagation. Pairwise projection inner products join the endpoint features in the head. |
| M25 | CAWN | walks | Causal walks, anonymous occurrence identities relative to both query endpoints, bidirectional walk encoding and attention over walks. |
| M26 | TNCN | neighbors | TGN-style memory embeddings plus explicit common-neighbor memory aggregation and endpoint-overlap terms. |
| M27 | DyGMamba | neighbors | Node-history and pair-time selective state-space blocks with input-dependent discretization, using portable PyTorch scans. |
| M28 | NAViS | — | Per-actor gated linear affinity state and a recent global affinity-profile buffer over query categories. Regression replaces affinity ranking. Offered for query and graph targets. |
| M29 | CTAN | — | Anti-symmetric state transform with damping, graph propagation and explicit numerical integration on the event clock. |
| M30 | MTGNN | — | Learned directed adjacency over signal channels, mix-hop propagation, gated dilated temporal convolutions and a query-conditioned head. Offered only with signals. |

References: [TGAT](https://arxiv.org/abs/2002.07962), [TGN](https://arxiv.org/abs/2006.10637), [JODIE](https://arxiv.org/abs/1908.01207), [DyRep](https://arxiv.org/abs/1803.04051), [GraphMixer](https://arxiv.org/abs/2302.11636), [DyGFormer](https://arxiv.org/abs/2303.13047), [HyperEvent](https://arxiv.org/html/2507.11836v2), [TPNet](https://arxiv.org/html/2410.04013), [CAWN](https://arxiv.org/abs/2101.05974), [TNCN](https://arxiv.org/html/2406.07926v2), [DyGMamba](https://arxiv.org/html/2408.04713v3), [NAViS](https://arxiv.org/html/2510.06940v3), [CTAN](https://proceedings.mlr.press/v235/gravina24a.html) and [MTGNN](https://arxiv.org/abs/2005.11650).

**Training protocol**

- Every fit sees every training row exactly once per epoch, for 2–12 complete epochs, and scores the complete validation split. Each epoch checks the exact row ids for omissions, duplicates and validation rows.
- A training window ends at a labelled step s and covers steps s−H+1 to s, with H = `time/history`. It supervises only the labels at s. A node-target window holds at most 1,024 labels, so a step with more labels gets several windows over the same steps.
- With `time/context` c above 0, a window that starts in period p starts instead at the first step of period max(0, p−c). Each family gets its period-end readout. Replayed steps add no supervised rows.
- A batch groups windows of equal length, `batch sequences × 16 // H` of them. It holds at most 1,024 frames, 1,024 prediction rows and 1,048,576 entity-frames, where a window contributes its steps times its graph's entity count.
- Fits and model checks may allocate 90% of GPU memory.
- Sequence and memory states reset at window boundaries. Event models start from the last features known before the window.
- Targets are standardised with the training mean and sample standard deviation. Predictions are converted back before scoring raw MSE and R².
- The saved checkpoint is reloaded and must reproduce the first validation batch.
- Each fit runs in its own process with a 600-second deadline. At the deadline a fit may finish its current epoch only if its earlier complete epochs averaged strictly under 120 seconds; it is then scored. Otherwise it times out without a score. The supervisor applies the same rule, and stops an extended worker that reports no progress for 600 seconds.

**Search**

| Selector | How trials are chosen |
|---|---|
| `--selector tpe` | Optuna's multivariate, grouped TPE with 10 random startup trials and 64 expected-improvement candidates. `model` is categorical. Model parameters are named `M##/<parameter>` and spec parameters `graph/<parameter>`. A spec parameter enters a trial only when the model reads its group and its condition holds. |
| `--selector random` | A uniform model, then each model parameter and each active spec parameter independently. |

For every proposal the runner rebuilds an in-memory study from the trial files; there is no external study database. A result enters the study only if it matches the frozen supervision fingerprint, covers every training row, scores the complete validation split and has a finite score. Otherwise the proposal fails. Timeouts and crashes enter TPE as failures ranked worse than every finite score. Interrupted trials are excluded.

`selection.json` records:

- the selector and its seeds; for TPE also its settings, the proposed parameters and the failure history
- the spec digest, the spec parameters that differ from their defaults, and the model's spec space
- prior trial ids with result hashes
- the bank manifest hash, the supervision hash and the declaration fingerprint

Configuration and selection are saved before the worker starts, and a resumed trial reuses them.

Model parameters draw from these ranges. Only the parameters a model uses enter its search space. Spec parameters and their ranges are in [graphspec/README.md](../graphspec/README.md).

| Parameter | Models | Range |
|---|---|---|
| Hidden width | all | 16, 24, 32, 48, 64 |
| Complete epochs | all | 2–12 |
| Learning rate | all | 0.0001–0.005, logarithmic |
| Weight decay | all | 0.0000001–0.01, logarithmic |
| Gradient clipping | all | 0.5–2 |
| Layers | M01–M17, M21–M23, M27, M30 | 1, 2, 3; M17 uses 1, 2 |
| Attention heads | M03, M11, M22, M23 | 1, 2, 4 |
| Dropout | M01–M07, M11–M16, M22, M23 | 0–0.5 |
| Neighbors | M17, M18, M20–M24, M26, M27 | 4, 8, 10, 12, 16, 20, 30, 32, 64; M17 uses 4, 8, 12, 16, 20 |
| Diffusion hops | M09, M30 | 1–4 |
| Patch size | M22 | 1, 2, 4, 8, 16 |
| Projection hops, width, decay | M24 | 1–4; 16, 32, 64, 128, 256; 0.0000001–1, logarithmic |
| Walks, walk hops | M25 | 1, 2, 4, 8, 16, 32; 1–4 |
| State width | M27 | 4, 8, 16, 32, 64 |
| Affinity buffer | M28 | 1, 2, 4, 8, 16, 32 |
| Integration steps, damping | M29 | 1–16; 0.001–1, logarithmic |
| Signal neighbors | M30 | 1–20 |

**Run directory**

| File | Contents |
|---|---|
| `plan.json` | Bank path and manifest hash, fingerprints, seed, budget, selector, workers, variant budget and every planned trial. |
| `runner.lock`, `running.json` | One runner per directory; the active worker's process id and trial. |
| `train_blocks_H.json` | The training windows for history H. |
| `trial_NNNNNN_<selector>_<model>_<spec digest>/` | One trial. |
| `trial_*/config.json`, `trial_*/selection.json` | The configuration, with its spec under `graph`, and how it was chosen. |
| `trial_*/worker_NN.log` | JSON events, one per line: fit entry, training progress, epochs, validation and timing logs. A failing worker ends the log with its traceback. |
| `trial_*/source/` | The `tuning`, `graphspec` and `graph_tuning_infra` sources the worker ran. |
| `trial_*/result.json`, `predictions.npz`, `checkpoint.pt` | Scores, timings, parameter count, peak GPU memory and row hashes; validation predictions; weights. |
| `trial_*/failure.json`, `timeout.json`, `fit_progress.json` | Failure records and the budget heartbeat. |
| `results.csv`, `results.jsonl`, `progress.md`, `coverage.json` | Every attempt, including failures and the reference baseline; the best score and spec digest per model in `progress.md`; completed fits per model in `coverage.json`. |

A file named `STOP` in the run directory stops the run after the current fit, and stops a resumed run before its first trial. The runner refuses to resume while the worker named in `running.json` is still alive.

**Verification**

`python -m tuning.check_models BANK --output checks.json [--graph N] [--spec spec.json] [--device cuda]` runs forward, backward and checkpoint reload for every model the bank offers. Each model uses its default spec, with any of its active parameters taken from `spec.json`; a key that no model's space declares is refused. It runs on the last window of graph `N`, or of the graph with the most observations. A model that runs out of GPU memory is recorded and the check continues; the command then exits with the list of those models. Any other error stops it.
