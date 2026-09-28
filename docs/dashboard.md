Run the tuning dashboard with the existing runtime environment:

```powershell
./scripts/graphspec.ps1 -RuntimeHome D:/graphspec-runtime dashboard --runs D:/graphspec-runtime/runs --port 8791
```

Open [the dashboard](http://127.0.0.1:8791). The server binds to localhost. `--runs` accepts a folder of runs or one run containing `plan.json`. The default is `GRAPHSPEC_HOME/runs`.

Analyze starts with an explicit choice of what to plot. Every saved model, graph, and training parameter is searchable in the parameter chooser. Choose `graph/derive/hops`, for example, to plot scores against its numeric values and inspect counts, minimum or maximum, and median for each value. Trial order provides a chronological plot. Validation MSE is minimized; R² is maximized. Duration and model parameter count are also available as measures. Group comparisons describe observed results and do not establish parameter importance.

The trial explorer supports arbitrary parameter columns and sorting. Numeric values sort numerically; unused parameters stay last in either direction. Parameter filters support equality, text matching, numeric ranges, and presence or absence. Multiple conditions combine with AND and apply to both the graph and table. Clicking a grouped result opens its exact matching trials. CSV exports include all matching rows and the chosen parameter columns; grouped summaries have their own export.

Parameter choosers now browse slash and colon paths as folders. Breadcrumbs move up the hierarchy. Search applies within the current branch, with Search all fields available to broaden it. Large search results have Show more. Search results distinguish settings from recorded results.

Discover ranks saved configuration fields for any chosen numeric or categorical field, including nested recorded results and configuration settings. Numeric targets open directly in the graph; categorical targets open the trial table. Identical recorded input columns share a ranking row. Trial filters apply before ranking. Rankings are explicit snapshots, updated with Update ranking, and are cleared when the target, run, or trial filters change.

The ranking implementation uses deterministic five-fold held-out predictions. Numeric inputs use a regression split with at least five training rows per side; categorical inputs use category averages shrunk by five observations toward the training mean. Numeric targets use squared error; categorical targets use squared probability error. Gain is the reduction in held-out error relative to the training-fold mean or class proportions. Missing inputs and targets are excluded, at least 20 usable rows are required, and the target itself and identical copies are omitted. Rankings are exploratory associations, not causal estimates or significance tests; adaptive search and screening many fields can inflate apparent signals. See the [cross-validation reference](https://scikit-learn.org/stable/modules/cross_validation.html) for the held-out evaluation principle.

Discovery calculation runs in a worker so navigation remains available. Changing the criteria cancels obsolete work. `node --test tests/test_dashboard_discovery.mjs tests/test_dashboard_parameters.mjs` checks hierarchy navigation, target types, scale invariance, noise, exact target exclusion, missing values, and shared parameter logic.

Summary metric boxes are banned. Model family is not a default chart, grouping, table column, or filter. It is available as an ordinary saved parameter only when chosen explicitly. Run activity and context use a compact status line and an expandable detail section.

Select up to four trials for a parameter-by-parameter comparison. Selections persist through filtering and refreshes, and clear when switching runs. The address stores the selected run, view, filters, sorting, page, plotted parameter, measure, chosen columns, and comparison selection. Missing parameters, null, zero, and false remain distinct. Trial numbers are one-based positions in the saved plan.

The inspector exposes the saved configuration, actual proposal phase, epoch losses, timings, failure traceback, and the latest worker log. Long logs are limited to their last 24 KB. Proposal histories are loaded only for the inspected trial. The explorer identifies the configured search strategy; the inspector distinguishes its random startup phase when recorded.

Automatic refresh runs every five seconds after a request finishes and pauses while the page is hidden. Turning it off only pauses the display. Errors mark retained results as stale; switching runs clears the previous run immediately. The dashboard only reads tuning records and provides no training controls.

Verification is isolated from training. `tests/test_dashboard.py` creates temporary run records and exercises the API, records, bounded log reads, and error handling. Run it as a short standalone suite:

```powershell
$env:GRAPHSPEC_HOME = 'D:/graphspec-runtime'
& D:/graphspec-runtime/environment/Scripts/python.exe -B -m unittest discover -s tests -p test_dashboard.py -q
```

The `dashboard_explorer.cjs`, `dashboard_details.cjs`, `dashboard_resilience.cjs`, and `dashboard_parameters.cjs` files in `tests` are Playwright page functions. They run against a separate fixture server on port 8792. Generate that server's input using `fixture` and `write` in `tests/test_dashboard.py`: `search-alpha` with 32 trials, `search-beta` with 3 trials, `empty-run` with no trials, and `broken-run/plan.json` with version 0. Each browser group runs independently and completes in a few seconds. Pure parameter logic is checked with `node --test tests/test_dashboard_parameters.mjs`.
