**Policy retention at a fixed prediction date**

The churn adapter reads one CSV, directly or from a ZIP, and writes a static, heterogeneous graph at prediction date T. E is one calendar year later; February 29 maps to February 28 in a non-leap year. The scalar regression target `retained` is one when a client active at T is active at E, otherwise zero.

A policy is in force on `[effective_date, min(no_longer_in_force_date, expiration_date))`. A client is active if any policy is in force. The end date is exclusive, so a same-day renewal can preserve activity. The cohort is constructed only from policies starting on or before T. Outcomes use the complete source separately. `--data-through` must establish complete outcome coverage through E; a future scheduled expiration or effective date does not establish that coverage.

```powershell
./scripts/graphspec.ps1 -Home D:/graphspec-runtime churn --source C:/Users/trist/Downloads/TX.csv.zip --output D:/graphspec-runtime/datasets/churn --prediction 2025-08-31 --data-through 2026-09-08 --features 64
./scripts/graphspec.ps1 -Home D:/graphspec-runtime build --task D:/graphspec-runtime/datasets/churn/churn.task.json --output D:/graphspec-runtime/banks/churn --workers 1
```

Run these commands from the checkout with an external virtual environment at `D:/graphspec-runtime/environment/Scripts/python.exe`. The launcher requires `-Home` and sets `GRAPHSPEC_HOME` before Python starts. Direct Python or installed CLI use also requires an explicit absolute `GRAPHSPEC_HOME`. The source ZIP can remain on C:; Windows outputs must stay on the configured runtime volume, with caches and temporary files under that home. See [storage details](docs/loader.md).

The TX example uses T = 2025-08-31 and E = 2026-08-31, the last completed month end before the CSV's ZIP timestamp of 2026-09-08. That timestamp supports the chosen coverage assumption; it does not independently verify administrative completeness. Dates are explicit inputs so other extracts can use their own cutoffs. All designed descriptive fields are included by default. `--known-fields` can select a subset, or `none` can retain only clients and policies. Raw status and end-date columns cannot be added as unrestricted features.

**Information boundary**

Only policies with effective date at or before T become graph nodes. A policy ending after T exposes neither its end date nor whether it will be cancelled; it has an active indicator and its age at T. Policies ending strictly before T contribute elapsed time since ending and their cancellation flag. Clients also receive counts of those past cancellations. An ending exactly at T makes a policy inactive under the interval rule, but its cancellation flag is conservatively excluded. Scheduled expiration dates are not numeric features, even though an original contractual expiration might have been known at T.

Client, location, industry, and other descriptive snapshot attributes are used under the agreed problem definition, even though they may have changed. They do not need a separate historical-availability confirmation. The CSV has no recorded-at timestamps or attribute version history, so the cutoff applies to effective policy dates and endings, rather than reconstructing when an administrative edit was entered. The build record preserves the descriptive-field policy and cancellation cutoff.

The feature builder never receives outcomes or split membership. Changing cancellation flags for policies ending at or after T, changing future end dates while they remain after T, or altering future-policy attributes leaves the input graph and cohort unchanged. Future end dates may alter labels at E. Changing a cancellation flag on a policy ending before T changes its historical features. Renewals connect only policies already visible by T, and only to an earlier-or-same-effective-date predecessor.

**Nodes and relationships**

Client and policy are the central nodes, joined by `holds`. All historical policies are retained as context, including already ended policies and clients without an active policy; only active clients receive labels.

| Source field | Shared node |
|---|---|
| producer_key | producer |
| placement_specialist_key | placement_specialist |
| account_executive_key | account_executive |
| servicer_key | servicer |
| branch_key | branch |
| billing_company_key | billing_company |
| issuing_company_key | issuing_company |
| policy_type_key | policy_type |
| client_state_code, client_city | client_state, client_city |
| issuing_state_code | issuing_state |
| naics_code, sic_code | naics, sic |
| integrated_platform | platform |
| pol_income_type | income_type |
| line_of_business_code | line_of_business |
| policy_type_group_code | policy_group |

Each policy links to each available descriptive attribute node. Clients also link to observed geography and industry nodes. Additional relationships connect city to state, policy type to policy group, group to line of business, and renewing policy to predecessor. Numeric NAICS codes of two to six digits and SIC codes of two to four digits connect to successive code prefixes down to two digits, preserving leading zeros. This is a prefix hierarchy; it does not invent missing digits or interpret malformed codes as official taxonomy entries. Cities are normalized and qualified by state. Missing/sentinel keys do not create shared hubs. Issuing and billing companies and staff roles retain separate identity namespaces because shared identifiers across those domains have not been established.

`db_branch_key` and `ds_servicer_key` duplicate branch and servicer identifiers in the supplied file; the adapter checks their equivalence on visible rows and omits duplicate nodes. `fact_policy_key` is a redundant row identifier. Display names duplicating categorical codes do not become separate nodes or feature channels. Shared-node paths represent co-membership without creating quadratic client cliques.

Policy features are age, active status, elapsed time since a past ending, and past cancellation status. Client features are historical policy count, active-policy count, past-cancellation count, and tenure. Entity types and descriptive categorical codes are encoded with the existing feature encoder; arbitrary numeric identifiers are not interpreted as numerical measurements. Feature width is configurable. Zero-length or reversed intervals never establish activity, but issued records effective by T remain historical context.

Clients are split by a seeded identity hash independent of their outcomes, using the requested validation fraction. This is a transductive client holdout at one cutoff, not a temporal backtest. Both partitions must contain both outcomes for the current regression scorer. The adapter does not retry seeds or select clients based on outcomes. Additional cutoffs should be separate builds so later observations cannot enter an earlier bank through graph joining.

The output includes a task, staged Parquet tables, and a build record containing source checksum, cutoff, coverage declaration, field allowlist, split settings, node counts, and outcome counts. Banks and model fits use the same code as Total War; churn has node targets and no query-role tables.
