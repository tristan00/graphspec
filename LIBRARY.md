**Plan: graphspec as a PyPI library**

Written 14 September 2026. The goal is one installable distribution that lets a user stage their own temporal graph, build a bank, and search graph transforms and models over it, with the mechanism layer kept apart from the graph logic the way `graph_tuning_infra` is today. Sections are ordered; each depends on the ones before it.

**Where things stand**

Two repositories: this one holds `graphspec/` (the transform logic), `tuning/` (bank, models, search), `dashboard/` (results server) and `totalwar/` (one client loader); `graph_tuning_infra` holds files, arrays, databases, processes, logging, search plumbing, the HTTP server, hashing and the feature encoder, imports nothing from here, and has a test that fails if it ever does. Three things stand between this and a library: the split across two repositories with a relative install, code that only runs on this workstation (Windows file locks, a CUDA-only training path, a pinned LadybugDB wheel that needs two DLLs copied by hand), and the absence of a public interface: users would have to read `tuning.build` and `tuning.run` to know what to call.

**1. One distribution, two layers**

1.1 Move `graph_tuning_infra` into this repository as the subpackage `graphspec.infra`, keeping its module names (`graphspec.infra.files`, `graphspec.infra.arrays`, and so on) and its git history through a subtree merge. Delete the sibling repository's install line from `requirements.txt`.

1.2 Keep the separation as a rule the tests enforce: `graphspec.infra` imports only third-party packages and the standard library, never `graphspec.core`, `graphspec.tuning` or `graphspec.dashboard`. Port `test_boundary.py` to walk the package and assert it. The rule for what belongs where stays as it is written at the top of EFFICIENCY.md: resource decisions, storage, processes, devices and servers are infra; anything that knows what a node, relation, step or spec is lives outside it.

1.3 Rename the current top-level `graphspec` package to `graphspec.core` so the distribution and the import root share one name, and move `tuning` to `graphspec.tuning`, `dashboard` to `graphspec.dashboard`. Update every import, the `PACKAGES` tuple in the runner and the source digest, which hashes package directories by name.

1.4 Leave `totalwar/` out of the wheel. It is a client of the library, not part of it, and it depends on a private corpus. Keep it in the repository under `examples/totalwar/` so it stays tested against the library, or move it to its own repository; either way it imports `graphspec` like any user would.

**2. Runs anywhere the dependencies run**

2.1 File lock: `infra.processes.Lock` uses `msvcrt`. Replace it with a lock that works on Windows, Linux and macOS: create the lock file with `O_CREAT | O_EXCL` and write the owner's pid, release by unlinking, and treat a stale file whose pid is dead as free. One implementation, no platform branches.

2.2 Process launch: the runner passes `CREATE_NO_WINDOW` to `Popen`; pass it only on Windows through `creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)`.

2.3 CUDA: `compute.reserve`, the packer's `cudaHostRegister` and the fit refuse to run without a CUDA device. Keep CUDA as the supported training device, but make the refusal a clear error at `fit` entry rather than a crash inside `torch.cuda.cudart`, and let the CPU path that the tests already use for packing stay a supported way to run the test suite. A CPU training path is not a goal.

2.4 LadybugDB: it is only needed by `Database` (reading a bank from a `.lbug` file), `copy` and `verify`. Make it an optional dependency (`graphspec[ladybug]`), and document the DLL step for Windows. The staging path, which is the one training uses, needs only pyarrow. `psycopg` is only for the Total War loader and leaves the core dependencies with it.

2.5 Python and dependency ranges: the code runs on 3.14 here; declare `>=3.11` only after the test suite passes on 3.11 and 3.12 in CI. Pin lower bounds for numpy, numba, torch, pyarrow and optuna to the versions tested.

**3. A public interface**

3.1 Define the top-level API in `graphspec/__init__.py` and keep it small:

- `stage(directory)`: a `Stage` for writing contract tables, with `merge` for shards.
- `build_bank(task, output, workers)`: the bank builder, taking a task file or a dict.
- `check_models(bank, output)`: every preset configuration once.
- `search(bank, output, selector, seconds, trials, workers, bounds)`: the runner, with `resume`.
- `report(run)`: the results table.
- `fit(bank, config, output)`: one configuration, for users who want a single model.
- `Registry`, `Bank`, `spec.space`, `spec.canonical` for those who build configurations by hand.

Everything else is internal and may change. The console entry points `graphspec-build`, `graphspec-check`, `graphspec-run`, `graphspec-report`, `graphspec-dashboard` and `graphspec-synthetic` wrap these.

3.2 The contract document (the table schema, time and causality rules, target levels, task file) moves from the root README into `docs/contract.md` unchanged, and the README becomes the library front page: install, the five-command walkthrough on synthetic data, links.

3.3 Errors a user can cause, wrong schema, missing observation, non-finite attribute, unbuildable label schedule, already raise `ValueError` with a sentence naming the table and rule. Keep that, and make sure every one of them is raised before any long work starts, so a bad staging fails in seconds.

**4. Infra decisions become parameters**

4.1 The constants that describe this workstation live in `infra.compute`, `infra.arrays`, `infra.processes` and `infra.files`: GPU fraction, resident graph limit, worker count, variant disk budget, fact budget, load pool size. Keep them as module defaults but read them from one `Settings` object that a user can construct or load from a `graphspec.toml`, passed down from the public entry points. No environment variables, no globals mutated after import.

4.2 The batching budgets in `tuning.batching` (frames, rows, entity-frames, facts) belong to the same object; they are the memory policy of the training device.

**5. Tests and continuous integration**

5.1 The suites: 11 tests here and 27 in infra today. Add the tests a library needs and the ones we have been running by hand: the staged reader against the database reader on a synthetic graph, packing equality between the worker processes and the single-process path for every view, the reference-output comparison for every model configuration on a generated bank (four batches, saved at a known commit, compared within the measured noise), the loader shard equality, and the boundary test. Keep the whole suite under twenty seconds on the GPU and skip the CUDA tests cleanly without one.

5.2 CI on GitHub Actions: Linux and Windows, Python 3.11 through 3.14, CPU only, running everything that does not need CUDA. The CUDA suite runs on this workstation before a release.

5.3 Versioning: `0.x` until the interface in section 3 has been used by someone other than us. A `CHANGELOG.md` with one line per user-visible change.

**6. Documentation**

6.1 `docs/contract.md` (from the README), `docs/transforms.md` (from graphspec/README.md), `docs/models.md` (from MODELS.md and tuning/README.md), `docs/tutorial.md` (synthetic data end to end), `docs/loader.md` (how to write a loader, with `examples/totalwar` as the worked example: contract tables, shards, block ids, determinism).

6.2 EFFICIENCY.md and TOTALWAR.md stay as engineering records, not user docs.

**7. Publishing**

7.1 `pyproject.toml` with the package data (`dashboard/index.html`), extras (`ladybug`, `postgres`, `dashboard`, `test`), entry points and classifiers. `python -m build` produces a wheel and an sdist; install the wheel into a fresh virtual environment on a clean machine and run the tutorial before every release.

7.2 Publish to TestPyPI first, then PyPI through GitHub's trusted publishing on a tagged release. The name `graphspec` must be checked for availability; if taken, the fallback is decided by you, not by the plan.

**Order and effort**

Sections 1 and 2 are mechanical and touch every import; do them together in one change with the tests green at the end. Section 3 is the design work and the one that deserves review before it is written. Sections 4 to 7 follow in order. Nothing here changes what the search computes; the equality checks used throughout EFFICIENCY.md apply to the whole reorganisation.

**Decisions for you**

- Whether `totalwar` stays in this repository as an example or moves out.
- Whether the distribution keeps the `graphspec` name or gets a new one if PyPI has it.
- Whether a CPU training path is wanted for users without a CUDA device. It is not in this plan.
