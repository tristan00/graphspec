import numpy as np

from graphspec.infra.files import read_json, write_json
from tuning import catalog, panel, references
from tuning.sequences import POINTS, Sequences


def store(root):
    random = np.random.default_rng(0)
    table = {name: [] for name in POINTS}
    for graph in range(12):
        dates = np.cumsum(random.integers(20, 120, 13)).astype(np.float64)
        levels = np.cumsum(random.normal(size=(13, 2)) * 0.1, 0)
        for step in range(12):
            for series in range(2):
                for later in range(step + 1, 13):
                    row = (
                        graph,
                        step,
                        int(dates[step]),
                        series,
                        dates[later] - dates[step],
                        levels[later, series] - levels[step, series],
                        graph >= 9,
                    )
                    for name, value in zip(POINTS, row):
                        table[name].append(value)
    table = {name: np.asarray(values) for name, values in table.items()}
    held = table["target"][table["held"]]
    prediction = held[None] * np.array([1.0, 0.5, -1.0])[:, None] + random.normal(size=(3, len(held))) * 0.01
    root.mkdir()
    np.savez_compressed(root / "sequences.npz", prediction=prediction, **table)
    write_json(
        root / "sequences.json",
        dict(
            series=["level", "trend"],
            scale=[1.0, 2.0],
            weights=None,
            models=[dict(name="trial_%06d_random" % index, view="tree") for index in range(3)],
        ),
    )
    return root


def test_histories_rebuild_levels_from_stored_changes(tmp_path):
    held = Sequences(store(tmp_path / "store")).held
    order, date, level = references.histories(held)
    first = order[0]
    points = slice(held["start"][first], held["start"][first + 1])
    later = order[1:12]
    assert np.allclose(date[later], held["tau"][points][:11])
    assert np.allclose(level[later], held["target"][points][:11])


def test_panel_scores_every_row_and_names_every_measure(tmp_path):
    from tuning.page import write

    root = store(tmp_path / "store")
    result = panel.plain(panel.build(root, 1, 50, ["level"]))
    names = [row["name"] for row in result["rows"]]
    assert names[:3] == ["T0 tree", "T1 tree", "T2 tree"] and len(names) == 11
    assert {entry["key"] for entry in result["catalog"]} >= set(result["measures"])
    assert all(catalog.entry(name)["family"] for name in result["measures"])
    error = result["measures"]["point/mae"]["level"]
    assert np.argmin(error["decisions"][:3]) == 0 and error["confidence"][0] == 1
    assert all(
        low <= value <= high for low, value, high in zip(error["low"], error["decisions"], error["high"])
    )
    assert result["measures"]["ranking/spearman, terminal"]["trend"]["decisions"][2] < -0.9
    assert result["ranks"]["all series"]["point/mae"][0] == 1
    assert result["relative"]["relative/mae ratio"]["reference: no change"]["0"][0] < 0.1
    assert abs(sum(result["ensemble"]["level"]["ensemble/rank histogram"]) - 1) < 1e-9
    write(tmp_path / "panel.html", result)
    assert "point/mae" in (tmp_path / "panel.html").read_text(encoding="utf-8")
    assert read_json(root / "sequences.json")["series"] == ["level", "trend"]
