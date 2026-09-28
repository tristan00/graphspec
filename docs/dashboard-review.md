Dashboard review, 16 September 2026

The review used the existing in-app browser tab and the real tuning records. Earlier implementations and git history were not read.

Confirmed problems corrected:

- A leftover dashboard was still listening on port 8766. Its dashboard processes were identified by command line and listener, then stopped. The current server remains on 8791.
- Numeric equality treated `2.0` differently from `2`. The live run returned zero matches before the fix and 29 matches afterward. Equivalent numeric input now compares numerically; strings retain exact matching.
- Refresh collapsed expanded run details. Open disclosures now survive rendering, including saved-record and proposal disclosures in the inspector.
- Keyboard skip navigation replaced the URL state and reset the selected run and view. It now moves focus to the main content while preserving the URL.
- Numeric charts shrank labels on narrow screens. Charts now retain readable dimensions and scroll inside their container. At a 375-pixel viewport, labels had 16-pixel-high bounds and the page did not overflow horizontally.

The live review exercised both the active run and the 1,306-trial run: parameter search and keyboard selection, numeric equality, exact group drilldown, pagination, numeric sorting in both directions, status filtering, empty search results, disabled empty exports, comparison selection and removal, differing parameters, inspector configuration and failure logs, keyboard inspector tabs, keyboard chart inspection, categorical duration graphs, chronological R² graphs, reload state, and manual refresh.

Independent checks: 28 Python dashboard and runner tests passed in 2.86 seconds; eight Node parameter tests passed in 0.09 seconds. JavaScript syntax checks passed. The numeric equality regression is included in `tests/test_dashboard_parameters.mjs`.

The separate fixture-browser suites described in `dashboard.md` were run during implementation; they were not rerun as part of this review. This review instead checked the actual delivered tab, including the runtime process inventory that those fixture suites did not cover.

Final state: one dashboard listener on localhost port 8791 and one in-app dashboard tab. The virtual-environment launcher has no listening socket; its child owns the single server. The active run is open with no parameter, filter, comparison, or extra column selected, and live refresh enabled. An expanded proposal disclosure remained open while the displayed update time advanced during live polling. No browser errors or warnings were reported.

Discovery changes were subsequently reviewed in the same tab. Verified branch navigation and scoped search, Search all fields, keyboard selection and Escape, additional search results, preservation during refresh, ranking pagination and search, target and filter changes, cancellation, empty subsets, numeric graph drilldown, categorical rankings, and the 1,306-trial run. The target chooser includes saved configuration and nested result fields; GPU memory and status were exercised as additional targets. At 375 pixels the chooser stayed inside the viewport. Twenty-one Node checks and 28 Python checks passed; Python lint and JavaScript syntax checks passed. Final state has one listener and one tab, with Discover open and no target or parameter preselected.
