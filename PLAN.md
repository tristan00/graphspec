**Plan: dataset-independent graph tooling and an unattended search**

Dataset independence is the library's baseline requirement, not a new standard. The library builds a truth graph from what a user states about their data, derives variants, and tunes models over them. It never knows a dataset. Two datasets, Churn and Total War, were let into the library as packages, CLI verbs, a database helper, docs, and tests. That is the defect this plan removes, together with the reason a search needs an operator.

Status rule while working this plan: anything blocked stays at the top of every status report until it clears. A section may not begin until every subsection before it is done.

**What is wrong today, measured**

- `graphspec/__main__.py` registers `churn` and `totalwar` as commands. `pyproject.toml` packages both and declares a `totalwar` extra. `graphspec/infra/postgres.py` exists only for Total War. `CHURN.md`, `TOTALWAR.md`, `tests/test_churn.py`, and `tests/test_adapters.py` are dataset code and docs inside the library. `tests/test_library.py:60` lists both packages by name.
- The bank builder, transforms, models, and search have no dataset references. The hardcoding is confined to the items above.
- The public loader cannot express the contract. `GraphTables` covers nodes, observations, edges, and node labels only. It has no queries, no query edges, no signals, no query-level or graph-level labels. It takes steps, not times. It restates nothing and derives no removals. It has no summary and reports errors by table at build time.
- Because of that, Total War (3,140 lines) hand-builds every contract table (`totalwar/tables.py`), numbers `seq`, restates persistent facts at every step, diffs each step against the previous to emit removal rows (`totalwar/graph.py`), keeps its own id and kind bookkeeping (`totalwar/entities.py`), and verifies contract rules itself (`totalwar/verify.py`). Roughly a third of the package is graph mechanics that belong to the library. The rest reads the game database and declares node types, relations, features, and rewards.
- Churn (499 lines) uses `GraphTables` and is closer, but still computes per-cutoff snapshots and counts by hand (`churn/snapshot.py`), which is feature engineering a graph model should derive from structure.
- Both adapters import internals: `graphspec.staged.IDS`, `graphspec.infra.staging`, `graphspec.infra.features`, `graphspec.infra.runtime`, `graphspec.infra.processes`, `graphspec.infra.postgres`.
- Operating a search needs a person or an agent. Runs execute code from the git checkout when started with the checkout as working directory, so each trial's fresh worker picks up whatever is on disk. Resume requires the full original command line, and `--workers` and `--variant-gb` silently reset to defaults when omitted (`tuning/run.py:98`). Changing a setting means STOP file, wait for the trial boundary, delete STOP, relaunch. Nothing keeps the runner alive when the terminal or agent that launched it exits. The dashboard is read-only. `runner_timing.json` and `timing_workers/` are written per trial and read by nothing.

**Acceptance test**

A person who has never seen the library writes a script that reads their source and states facts about the graph through the builder: nodes with attributes, edges with the times they hold, attribute changes at times, labels, and queries if their problem has decision moments. They run `inspect`, `build`, and `search run`. The only logic they write is reading their source and the label rule. Churn and Total War are redone exactly this way; whatever remains in their scripts beyond reading the source and stating facts is a missing builder feature, not something to leave in the script.

---

**1. The graph builder**

1.1 A `Graph` builder where each call states one fact and nothing else. Keys are the user's strings. Times are dates, timestamps, or integers. The user numbers nothing.

```python
from graphspec import Graph, Stage

g = Graph("customers")
g.node("client", "c1", region="north")
g.node("policy", "p9", start="2024-01-01", end="2025-01-01", premium=120.0)
g.edge("holds", "client", "c1", "policy", "p9", start="2024-01-01", end="2025-01-01")
g.attributes("client", "c1", time="2024-06-01", tenure=5.0)
g.label("client", "c1", time="2025-01-01", y=1.0)
g.query(
    key=17, time="2024-03-01", category="renew",
    actor=("policy", "p9"), subject=("client", "c1"), context=("client", "c1"),
    premium=120.0, signal={"margin": 0.2},
)
with Stage(staging) as stage:
    g.write(stage)
```

1.2 The library owns every mechanic: ids; steps from the distinct times per graph, or a declared calendar period; `seq`; carrying static attributes and open intervals forward and restating them at each step; removal rows when an interval ends; feature hashing from attribute dicts by value type, numbers as numbers and strings as categories, into one vector of a default width; query tables and their three edges; label tables at node, query, or graph level decided by what the label attaches to; splits by key hash with a fraction, or by time with the last periods held out, when not given; the manifest.

1.3 Errors name the call that broke the rule, at call time where the rule is local: an edge to an undeclared node, a query naming an entity with no attributes at its time, two labels for one node at one time, an interval ending before it starts. Rules that need the whole graph, such as every graph having a label, are checked at `write` and name the graph and rule. No error is reported by internal table name.

1.4 `g.summary()` and `graphspec inspect STAGING` print what was built: graphs, node types and counts, relations and their FROM/TO pairs, steps per graph, feature width, queries, labels per split and level.

1.5 Many graphs: `Stage` collects graphs written in any order or from parallel processes, and `Stage.merge` combines shards. Node types and relations are ordered by name, so the fingerprint does not depend on insertion order.

1.6 The file layout is the table form of the same calls, one row per call, read by `graphspec build --data DIR`: `nodes/<kind>` with key, optional graph, optional time, attributes; `edges/<relation>` with from kind and key, to kind and key, optional graph, start, optional end, attributes; `labels` with kind, key, optional graph, time, y, optional split; optional `queries`. Parquet or CSV. Same mechanics, same errors, same summary. The task file goes; the labels are the target.

1.7 The public surface is `Graph`, `Stage`, `build_bank`, `configuration`, `fit`, `open_bank`, and the CLI verbs `inspect`, `build`, `check`, `search`, `dashboard`, `database`, `synthetic`. Nothing outside the library imports from `graphspec.infra`. `GraphTables` and `write_task` are removed once section 2 no longer uses them.

1.8 Docs: `docs/contract.md` stays the reference for the staged tables and the time and causality rules. `docs/loader.md` is rewritten as the builder walkthrough from a two-table source to a running search. `README.md` describes one path: build the graph, build the bank, search.

**2. Both datasets become builder scripts and leave the library**

2.1 Churn as a builder script: read the CSV; one `node` per client and per policy with the policy's interval; one `edge` per policy from its client, holding over the same interval; the label rule for retention at the prediction date. The snapshot and count code is deleted; the model derives counts from structure. Measured against the current 499 lines.

2.2 Total War as a builder script: read PostgreSQL with `psycopg` directly; one call per entity, relationship interval, attribute change, decision query, and reward label. The reference tables, action catalogue, and reward rule remain because they are the game. Table building, sequencing, removal derivation, id bookkeeping, and contract verification are deleted. Measured against the current 3,140 lines.

2.3 Fingerprint guard: build both banks from the new scripts and compare `spec_sha256` and `supervision_sha256` with the existing `churn` bank and `bank_tw4000`. Ordering types and relations by name may change them. Existing banks, runs, and the recorded trials stay valid regardless; a run continues on the bank it was started with. New runs use new banks.

2.4 Delete `churn/`, `totalwar/`, `graphspec/infra/postgres.py`, `CHURN.md`, `TOTALWAR.md`, `tests/test_churn.py`, `tests/test_adapters.py`, the `totalwar` extra, both entries in `graphspec/__main__.py`, and both names in `tests/test_library.py`. The scripts live with their data under the runtime home, not in the library.

2.5 `synthetic` becomes a builder script the library ships and its test runs through `inspect`, `build`, and one fit, covering intervals, attribute changes, queries, signals, and each label level. This is the coverage the deleted tests provided.

**3. A search that runs without an operator**

3.1 Runs execute the installed package, never the checkout. `scripts/graphspec.ps1` stops using the checkout as working directory. `build`, `check`, `fit`, and `search` refuse to start when `tuning` or `graphspec` resolves inside a git checkout. Reinstall before starting a run. Until this lands, do not edit `tuning/` while a search is running, because each trial's worker imports it fresh.

3.2 Resume takes only `--output RUN`. Bank, workers, variant budget, and limit come from `plan.json`. Any flag given replaces the stored value and emits the existing `*_replaced` event. The unconditional reset at `tuning/run.py:98` is deleted.

3.3 `RUN/settings.json` holds selector, random trials, random params, seconds per trial, bounds, and trial limit. Start and resume flags write it. The runner re-reads it at every trial boundary and applies changes through the same replacement code, so changing a setting is editing one file with no stop and no relaunch. `plan.json` keeps version 9 and its shape; the first resume under the new code creates `settings.json` from the values `plan.json` already holds. STOP stays as the one way to stop.

3.4 `search run` starts the runner as a process that survives its launcher: one `Popen` with `DETACHED_PROCESS`, `CREATE_BREAKAWAY_FROM_JOB`, and `CREATE_NEW_PROCESS_GROUP` on Windows and `start_new_session` elsewhere, stdout and stderr to `RUN/runner.log`, `--foreground` for tests. The runner already records its own pid in `runner.json`; the launcher reports nothing and resolves nothing.

3.5 A test that an interrupted trial retries as a new attempt, so nobody repairs one by hand.

3.6 The dashboard writes STOP and `settings.json`, against 3.3. It stays read-only until then.

**4. Cleanup**

4.1 Delete the `runner_timing.json` and `timing_workers/` writes.

4.2 Rename in-memory `turn` to `period` in `tuning/`. The stored keys `reward_mean` and `reward_sd` stay because every bank stores them; renaming them would invalidate banks or require reading both names.

4.3 Test suite stays under 20 seconds.

**Order and current state**

Order is 1, 2, 3, 4. Sections 1 and 2 touch `graphspec/`, the adapters, tests, and docs only. Section 3 edits `tuning/`, which the live search imports, so it starts after the search is stopped or after 3.1 is done from a reinstalled package.

As of 2026-09-18 the Total War search continues in `D:/graphspec-runtime/runs/totalwar-4000-unrestricted` under the Windows scheduled task `graphspec-search-totalwar-4000`, which is the external workaround 3.4 replaces. It runs the checkout at `D:/graphspec`. The dashboard serves on port 8791. Stop the search with a STOP file in the run directory before starting section 3.
