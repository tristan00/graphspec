from datetime import date

import pyarrow as pa
import pyarrow.csv as csv
import pytest

from churn.build import active_clients, anniversary, build, outcomes, truth
from graphspec import build_bank, configuration, open_bank


def policies():
    return pa.Table.from_pylist(
        [
            dict(
                policy_key="old",
                client_key="a",
                producer_key="agent",
                expiring_policy_key="future",
                pol_policy_effective_date="2023-01-01",
                pol_no_longer_in_force_date="2024-08-01",
                pol_expiration_date="2026-01-01",
                pol_policy_cancelled_flag="True",
            ),
            dict(
                policy_key="other",
                client_key="b",
                producer_key="agent",
                expiring_policy_key="",
                pol_policy_effective_date="2023-04-01",
                pol_no_longer_in_force_date="2027-01-01",
                pol_expiration_date="2027-01-01",
                pol_policy_cancelled_flag="False",
            ),
            dict(
                policy_key="future",
                client_key="a",
                producer_key="future-agent",
                expiring_policy_key="old",
                pol_policy_effective_date="2024-09-01",
                pol_no_longer_in_force_date="2027-01-01",
                pol_expiration_date="2027-01-01",
                pol_policy_cancelled_flag="False",
            ),
            dict(
                policy_key="ended",
                client_key="c",
                producer_key="former-agent",
                expiring_policy_key="",
                pol_policy_effective_date="2022-01-01",
                pol_no_longer_in_force_date="2024-01-01",
                pol_expiration_date="2024-01-01",
                pol_policy_cancelled_flag="False",
            ),
        ]
    )


def test_future_updates_change_labels_but_never_inputs_or_cohort():
    original = policies()
    rows = original.to_pylist()
    rows[0]["pol_no_longer_in_force_date"] = "2024-12-01"
    rows[0]["pol_policy_cancelled_flag"] = "False"
    rows[0]["pol_expiration_date"] = "2028-01-01"
    rows[2]["producer_key"] = "unseen-agent"
    rows[2]["pol_no_longer_in_force_date"] = "2024-12-31"
    changed = pa.Table.from_pylist(rows)
    first = truth(original, date(2024, 1, 1), ("producer_key", "expiring_policy_key"))
    second = truth(changed, date(2024, 1, 1), ("producer_key", "expiring_policy_key"))
    cohort = sorted(active_clients(original, date(2024, 1, 1)))
    same = sorted(active_clients(changed, date(2024, 1, 1)))
    assert same == cohort == ["a", "b"]
    assert first.tables.keys() == second.tables.keys()
    assert all(table.equals(second.tables[key]) for key, table in first.tables.items())
    assert outcomes(original, date(2025, 1, 1), cohort) == [1.0, 1.0]
    assert outcomes(changed, date(2025, 1, 1), cohort) == [0.0, 1.0]
    assert "future" not in first.lookup["policy"]
    assert "future-agent" not in first.lookup["producer"]
    assert first.tables[("edge_renews", "policy", "policy")].num_rows == 0


def test_cohort_uses_in_force_intervals_and_future_cancellations_are_censored():
    graph = truth(policies(), date(2024, 1, 1), ("producer_key",))
    cohort = sorted(active_clients(policies(), date(2024, 1, 1)))
    assert cohort == ["a", "b"]
    assert set(graph.lookup["policy"]) == {"old", "other", "ended"}
    assert set(graph.lookup["producer"]) == {"agent", "former-agent"}
    assert outcomes(policies(), date(2024, 8, 1), cohort) == [0.0, 1.0]
    assert outcomes(policies(), date(2024, 9, 1), cohort) == [1.0, 1.0]


def test_calendar_year_and_unsupported_fields():
    assert anniversary(date(2024, 2, 29)) == date(2025, 2, 28)
    with pytest.raises(ValueError, match="graph fields"):
        truth(policies(), date(2024, 1, 1), ("pol_policy_cancelled_flag",))
    with pytest.raises(ValueError, match="graph fields"):
        truth(policies(), date(2024, 1, 1), ("pol_expiration_date",))


def test_past_cancellations_are_features_but_later_flags_are_not():
    original = policies()
    fields = ("producer_key", "expiring_policy_key")
    first = truth(original, date(2024, 1, 1), fields)
    rows = original.to_pylist()
    for row in rows:
        row["pol_policy_cancelled_flag"] = "unavailable"
    second = truth(pa.Table.from_pylist(rows), date(2024, 1, 1), fields)
    assert all(table.equals(second.tables[key]) for key, table in first.tables.items())
    rows = original.to_pylist()
    rows[3]["pol_no_longer_in_force_date"] = "2023-12-31"
    past = truth(pa.Table.from_pylist(rows), date(2024, 1, 1), fields)
    rows[3]["pol_policy_cancelled_flag"] = "True"
    cancelled = truth(pa.Table.from_pylist(rows), date(2024, 1, 1), fields)
    observation = ("observation", None, None)
    assert not past.tables[observation].equals(cancelled.tables[observation])
    assert all(
        table.equals(cancelled.tables[key]) for key, table in past.tables.items() if key != observation
    )


def test_industry_hierarchy_uses_only_codes_on_visible_policies():
    table = policies().append_column("naics_code", pa.array(["541110", "5412", "999999", "541110"]))
    table = table.append_column("sic_code", pa.array(["0111", "0112", "9999", "01"]))
    graph = truth(table, date(2024, 1, 1), ("naics_code", "sic_code"))
    assert set(graph.lookup["naics"]) == {"54", "541", "5411", "54111", "541110", "5412"}
    assert set(graph.lookup["sic"]) == {"01", "011", "0111", "0112"}
    assert graph.tables[("edge_naics_parent", "naics", "naics")].num_rows == 5


def test_churn_adapter_builds_a_node_bank_through_the_shared_writer(tmp_path):
    rows = []
    for index in range(64):
        row = policies().to_pylist()[0]
        row.update(
            policy_key=str(index + 1),
            client_key="client-" + str(index),
            expiring_policy_key="",
            pol_no_longer_in_force_date="2026-01-01" if index % 2 else "2024-08-01",
        )
        rows.append(row)
    source = tmp_path / "policies.csv"
    csv.write_csv(pa.Table.from_pylist(rows), source)
    task = build(source, tmp_path / "churn", "2024-01-01", "2025-01-01", ("producer_key",))
    manifest = build_bank(task, tmp_path / "bank")
    bank = open_bank(tmp_path / "bank")
    assert manifest["target"]["level"] == "node" and manifest["rows"] == 64
    assert set(bank.targets) == {0, 1} and not bank.registry.extras["queries"]
    assert set(bank.registry.nodes) == {"client", "policy", "producer"}
    from tuning.check_models import probe

    cfg = configuration(
        tmp_path / "bank", model={"view": "snapshot", "hidden": 16}, graph={"time/history": 1}
    )
    assert probe(bank, bank.blocks(0, 1, cfg)[0], cfg, "cpu")["ok"]
    with pytest.raises(ValueError, match="data-through"):
        build(source, tmp_path / "too-soon", "2024-01-01", "2024-12-31", ("producer_key",))
