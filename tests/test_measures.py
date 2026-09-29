import math

import numpy as np
import pytest

from tuning import measures, pathwise, pooled


def walk(seed, size, width):
    path = np.cumsum(np.random.default_rng(seed).normal(size=(size, width)), 0) * 0.3
    return path - path[0]


def test_signature_kernel_equals_the_inner_product_of_signatures():
    first, second = walk(0, 6, 2), walk(1, 5, 2)
    expected = 1 + pathwise.signature(first, 10) @ pathwise.signature(second, 10)
    assert pathwise.signature_kernel(first, second, 6) == pytest.approx(expected, rel=1e-6)


def test_signature_starts_with_the_total_increment_and_obeys_the_shuffle_identity():
    path = walk(2, 7, 2)
    found = pathwise.signature(path, 2)
    assert found[:2] == pytest.approx(path[-1])
    assert found[3] + found[4] == pytest.approx(found[0] * found[1])


def test_alignments_are_zero_on_equal_paths_and_see_a_shift():
    values = np.array([0, 1.0, 2, 3, 2, 1])
    late = np.r_[0, values[:-1]]
    u = np.linspace(0, 1, 6)
    assert pathwise.warp(values, values, u, 0.0) == (0.0, 6, 0.0)
    cost, cells, distortion = pathwise.warp(values, late, u, 0.0)
    assert (cost, cells) == (1.0, 7) and distortion == pytest.approx(0.2)
    assert pathwise.soft_distortion(values, late, u, 0.01) == pytest.approx(0.2, abs=1e-6)
    assert pathwise.twed(values, values, u, 0.001, 1.0) == 0
    assert pathwise.erp(values, values, 0.0) == 0 and pathwise.msm(values, values, 0.5) == 0
    assert pathwise.edits(values, values, 0.25) == (0.0, 0.0)
    assert pathwise.curves(values, values + 1, u) == (1.0, 1.0)
    assert pathwise.warp(values, late, u, 10.0)[0] == pytest.approx(np.abs(values - late).sum())


def test_soft_divergence_is_zero_only_on_equal_paths():
    values = np.array([0, 1.0, 2, 3, 2, 1])

    def divergence(a, b):
        return (
            pathwise.soft_warp(a, b, 0.01)
            - pathwise.soft_warp(a, a, 0.01) / 2
            - pathwise.soft_warp(b, b, 0.01) / 2
        )

    assert divergence(values, values) == 0 and divergence(values, values + 0.5) > 1


def test_elastic_distance_separates_warping_from_scaling():
    grid = np.linspace(0, 1, pathwise.GRID + 1)
    wave = np.sin(2 * np.pi * grid)
    warped = pathwise.elastic(wave, np.sin(2 * np.pi * grid**1.5))
    scaled = pathwise.elastic(wave, 2 * wave)
    assert pathwise.elastic(wave, wave) == (0.0, 0.0)
    assert warped[0] < scaled[0] / 3 and warped[1] > 0.1


def test_shape_distance_ignores_scale_and_transport_sees_displaced_mass():
    wave = np.sin(np.linspace(0, 6, 33))
    assert pathwise.shape_distance(wave, 3 * wave) == pytest.approx(0)
    assert pathwise.transport(wave, wave, 1.0) == 0
    first, second = np.zeros(8), np.zeros(8)
    first[2], second[5] = 1, 1
    assert pathwise.transport(first, second, 1.0) == 3


def test_compare_scores_a_perfect_and_a_biased_path():
    tau = np.array([30.0, 90, 200, 400, 700, 1000, 1400, 1800])
    values = np.array([0.1, 0.3, 0.2, 0.5, 0.4, 0.7, 0.6, 0.9])
    truth = measures.Truth(tau, values, 1825.0)
    exact, summary = measures.compare(truth, values.copy(), 1.0)
    for name in ("point/mae", "elastic/dtw", "signature/kernel", "geometry/frechet", "spectral/fredf"):
        assert exact[name] == pytest.approx(0, abs=1e-9)
    assert exact["direction/sign"] == exact["dynamics/turning f1"] == exact["path/kendall"] == 1
    assert summary == truth.functionals and summary[0] == 0.9
    biased, _ = measures.compare(truth, values + 0.2, None)
    assert biased["point/mae"] == pytest.approx(0.2) and biased["bias/mean error"] == pytest.approx(0.2)
    assert biased["path/pearson"] == pytest.approx(1) and "point/level ape" not in biased
    assert biased["horizon/mae 365-730 days"] == pytest.approx(0.2)
    still, _ = measures.compare(truth, np.zeros(8), None)
    assert math.isnan(still["direction/sign"]) and "path/pearson" not in still


def test_ranking_and_pooled_scores_on_known_orderings():
    actual = np.arange(20.0)
    groups = np.repeat([0, 1], 10)
    right = pooled.ranking(actual, actual * 2 + 1, groups)
    wrong = pooled.ranking(actual, -actual, groups)
    assert right["spearman"] == pytest.approx(1) and wrong["spearman"] == pytest.approx(-1)
    assert right["regret"] == 0 and wrong["regret"] == 9
    assert right["ndcg"] == pytest.approx(1) and right["rank violation"] == 0
    assert right["rank biased overlap"] == pytest.approx(1) and wrong["pairwise accuracy"] == 0
    noise = np.random.default_rng(0).normal(size=200)
    found = pooled.build(noise, np.stack((noise, noise * 0.5 + 1)), np.ones(200))
    assert found["pooled/r squared"][0] == pytest.approx(1)
    assert found["pooled/miscalibration"][0] == pytest.approx(0, abs=1e-12)
    assert found["pooled/mincer zarnowitz slope"][1] == pytest.approx(2)
    assert found["pooled/miscalibration"][1] == pytest.approx(((1 - noise / 2) ** 2).mean())
    assert found["pooled/murphy mean dominated"].tolist() == [0, 1]


def test_tracked_measures_are_recorded_per_series():
    from types import SimpleNamespace

    from tuning.catalog import TRACKED
    from tuning.tracking import track

    random = np.random.default_rng(1)
    taus = np.tile(np.arange(1, 11) * 50.0, 8).astype(np.float32)
    targets = random.normal(size=80).astype(np.float32)
    bank = SimpleNamespace(
        taus=taus,
        targets=targets,
        series=np.tile(np.repeat([0, 1], 10), 4),
        query_of=np.repeat(np.arange(4), 20),
        series_names=["level", "trend"],
        scale=np.array([1.0, 2.0]),
        split=dict(train_indices=np.arange(40)),
    )
    held = np.arange(40, 80)
    exact = track(bank, held, targets[held].astype(np.float64))
    assert set(exact) == {"level", "trend"} and set(exact["level"]) == set(TRACKED)
    assert exact["trend"]["elastic/dtw"] == 0 and exact["trend"]["dynamics/turning f1"] == 1
    moved = track(bank, held[::-1].copy(), targets[held][::-1].astype(np.float64) + 1)
    assert moved["level"]["dynamics/area between"] == pytest.approx(1 - 0.05, abs=0.06)
    assert moved["trend"]["geometry/hausdorff"] == pytest.approx(0.5)
