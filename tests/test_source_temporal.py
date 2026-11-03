from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from functools import partial

import pyarrow as pa
import pytest

from graphspec.schema.temporal import Duration, Interval, Time, context, days, unit

pytestmark = pytest.mark.extended


@pytest.mark.parametrize("own,target,value,start,end", [
    ("hour", "second", 2, 7200, 10799),
    ("week", "day", 0, -3, 3),
    ("week", "day", -1, -10, -4),
    ("month", "day", 0, 0, 30),
    ("month", "day", 1, 31, 58),
    ("month", "day", 649, 19754, 19782),
    ("year", "day", 1972, 730, 1095),
    ("year", "month", 1969, -12, -1),
    ("month", "week", 0, 1, 4),
    ("year", "week", 1970, 1, 52),
])
def test_coarse_times_use_explicit_calendar_boundaries(own, target, value, start, end):
    with context(target):
        for side, expected in (("start", start), ("end", end)):
            result = Time()(pa.array([value, None]), dict(unit=own, at=side))
            assert result.type == pa.int64()
            assert result.to_pylist() == [expected, None]


@pytest.mark.parametrize("own,text,expected", [
    ("second", "1969-12-31 23:59:59.999999", -1),
    ("minute", "1970-01-01 00:01:59.999999", 1),
    ("hour", "1970-01-01 01:59:59.999999", 1),
    ("day", "1970-01-02 23:59:59.999999", 1),
    ("week", "1969-12-28 23:59:59.999999", -1),
    ("week", "1970-01-04 23:59:59.999999", 0),
    ("month", "1970-02-28 23:59:59.999999", 1),
    ("year", "2018-12-31 23:59:59.999999", 2018),
])
def test_text_times_floor_to_their_declared_unit(own, text, expected):
    with context(own):
        result = Time()(pa.array([text]), dict(unit=own, format="%Y-%m-%d %H:%M:%S.%f"))
    assert result.to_pylist() == [expected]


def test_timestamps_use_the_epoch_and_utc_offsets():
    native = pa.array([datetime(1970, 1, 1, tzinfo=timezone(timedelta(hours=1)))])
    with context("hour"):
        assert Time()(native, dict(unit="hour")).to_pylist() == [-1]
    with context("day"):
        assert Time()(pa.array([date(1969, 12, 31)]), dict(unit="day")).to_pylist() == [-1]
        assert Time()(pa.array(["1970-01-01+0100"]), dict(
            unit="day", format="%Y-%m-%d%z",
        )).to_pylist() == [-1]


def test_negotiation_names_both_conflicting_columns():
    for left, right in (("turn", "day"), (None, "day"), ("turn", "round")):
        with pytest.raises(ValueError, match=r"a.joined.*b.started"):
            unit([("a.joined", dict(unit=left)), ("b.started", dict(unit=right))])
    with pytest.raises(ValueError, match=r"a.joined.*requires at.*b.started"):
        unit([("a.joined", dict(unit="year")), ("b.started", dict(unit="day"))])
    with pytest.raises(ValueError, match=r"b.started.*at is refused"):
        unit([("a.joined", dict(unit="year", at="end")), ("b.started", dict(unit="day", at="start"))])


def test_negotiation_chooses_finest_time_and_checks_parameters():
    assert unit([("a.joined", dict(unit="year", at="end")), ("b.started", dict(unit="day"))]) == "day"
    assert unit([("a.turn", dict(unit="turn")), ("b.turn", dict(unit="turn"))]) == "turn"
    assert unit([("a.position", dict(unit=None))]) is None
    assert unit([]) is None
    with pytest.raises(ValueError, match=r"a.joined.*requires unit"):
        unit([("a.joined", {})])


@pytest.mark.parametrize("own", [None, "turn", "second", "year"])
def test_same_unit_keeps_exact_counts_and_nulls(own):
    with context(own):
        result = Time()(pa.array([-(1 << 63), 0, (1 << 63) - 1, None]), dict(unit=own))
    assert result.to_pylist() == [-(1 << 63), 0, (1 << 63) - 1, None]


@pytest.mark.parametrize("value", [0.5, True, float("inf"), float("nan"), Decimal("1.1")])
def test_numeric_times_refuse_nonwhole_counts(value):
    with context("day"), pytest.raises(ValueError, match="row 1.*whole units"):
        Time()(pa.array([value]), dict(unit="day"))


def test_temporal_parameters_never_supply_guesses():
    with context("day"):
        for parameters, value, message in [
            ({}, 1, "requires unit"),
            (dict(unit="day"), "1970-01-01", "requires format"),
            (dict(unit="day", at="start"), 1, "at is refused"),
            (dict(unit="year"), 1970, "requires at"),
            (dict(unit="year", at="middle"), 1970, "at is start or end"),
            (dict(unit="day", format="%Y-%m-%d"), "bad", "row 1"),
            (dict(unit="day"), Decimal(1 << 63), "row 1.*outside int64"),
        ]:
            with pytest.raises(ValueError, match=message):
                Time()(pa.array([value]), parameters)
    with context("turn"), pytest.raises(ValueError, match="calendar unit"):
        Time()(pa.array(["1"]), dict(unit="turn", format="%Y"))


def test_nested_conversion_context_restores_the_outer_unit():
    with context("day"):
        with context("hour"):
            assert Time()(pa.array([1]), dict(unit="day", at="end")).to_pylist() == [47]
        assert Time()(pa.array([1]), dict(unit="day")).to_pylist() == [1]
    with pytest.raises(ValueError, match="negotiated unit"):
        Time()(pa.array([1]), dict(unit="day"))


def test_unrepresentable_native_timestamp_names_its_row():
    values = pa.array([0, (1 << 63) - 1], type=pa.timestamp("s"))
    with context("day"), pytest.raises(ValueError, match="row 2"):
        Time()(values, dict(unit="day"))


def test_unrepresentable_source_timestamp_names_source_column_and_row(tmp_path, monkeypatch):
    from graphspec import READERS, Builder, BuildPlan, source

    def reader(place, columns, parameters):
        yield pa.table(dict(moment=pa.array([0], type=pa.timestamp("s"))))
        yield pa.table(dict(moment=pa.array([(1 << 63) - 1], type=pa.timestamp("s"))))

    monkeypatch.setattr(READERS, "held", {name: READERS.named(name) for name in READERS.names()})
    READERS.register("large_timestamp", reader)
    declaration = source("native", "large_timestamp", dict(moment=dict(type="time", unit="day")))
    builder = Builder(dict(events=declaration), tmp_path)
    with pytest.raises(ValueError, match="source events.*column moment row 2"):
        list(builder.read("events", BuildPlan(2 << 20, 1)))


@pytest.mark.parametrize("rows", [[], [None]])
def test_native_text_time_requires_format_without_present_values(rows):
    values = pa.array(rows, type=pa.string())
    for column in (values, values.cast(pa.large_string()), values.dictionary_encode()):
        with context("day"), pytest.raises(ValueError, match="text time requires format"):
            Time()(column, dict(unit="day"))


@pytest.mark.parametrize("rows", [[], [None], [[None, None]]])
def test_native_text_interval_endpoints_require_format_without_present_values(rows):
    for kind in (pa.list_(pa.string()), pa.list_(pa.large_string()), pa.list_(pa.string(), 2)):
        with context("day"), pytest.raises(ValueError, match="text time requires format"):
            Interval()(pa.array(rows, type=kind), dict(unit="day"))


@pytest.mark.parametrize("own", ["second", "month", "year", "turn"])
def test_duration_keeps_fractional_counts_in_its_own_unit(own):
    result = Duration()(pa.array(["1.5", "-2", None]), dict(unit=own))
    assert result.type == pa.float64()
    assert result.to_pylist() == [1.5, -2.0, None]


def test_native_duration_has_only_fixed_unit_conversion():
    values = pa.array([timedelta(seconds=90), None], type=pa.duration("us"))
    assert Duration()(values, dict(unit="minute")).to_pylist() == [1.5, None]
    with pytest.raises(ValueError, match=r"row 1.*no fixed conversion.*month"):
        Duration()(values, dict(unit="month"))
    for parameters in ({}, dict(unit=None), dict(unit="")):
        with pytest.raises(ValueError, match="requires a nonempty unit"):
            Duration()(pa.array([1]), parameters)


@pytest.mark.parametrize("resolution,expected", [("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9)])
def test_native_duration_retains_its_fractional_second_precision(resolution, expected):
    values = pa.array([1, -1, None], type=pa.duration(resolution))
    assert Duration()(values, dict(unit="second")).to_pylist() == [expected, -expected, None]
    minutes = Duration()(values, dict(unit="minute")).to_pylist()
    assert minutes[:2] == pytest.approx([expected / 60, -expected / 60], rel=1e-15, abs=0)
    assert minutes[2] is None


def test_dictionary_native_duration_retains_nanoseconds():
    values = pa.DictionaryArray.from_arrays(pa.array([0, None, 1]), pa.array([1, -1], type=pa.duration("ns")))
    assert Duration()(values, dict(unit="second")).to_pylist() == [1e-9, None, -1e-9]


def test_interval_keeps_open_ends_and_row_alignment():
    values = pa.array([[0, 1], [None, 2], [3, None], None])
    with context("day"):
        result = Interval()(values, dict(unit="week", at="end"))
    assert result.type == pa.list_(pa.int64(), 2)
    assert result.to_pylist() == [[3, 10], [None, 17], [24, None], None]
    with context("day"), pytest.raises(ValueError, match="row 2.*exactly two times"):
        Interval()(pa.array([[0, 1], [2]]), dict(unit="day"))


def test_native_nan_interval_endpoints_stay_null():
    values = pa.array([[float("nan"), 1.0], [2.0, float("nan")], [float("nan"), float("nan")], None])
    with context("day"):
        result = Interval()(values, dict(unit="day"))
    assert result.to_pylist() == [[None, 1], [2, None], [None, None], None]


def test_calendar_boundaries_cover_centuries_without_fixed_month_lengths():
    for year in (1, 400, 1600, 1900, 1969, 1970, 2000, 2024, 9999):
        for month in (1, 2, 3, 12):
            assert days(year, month) == (date(year, month, 1) - date(1970, 1, 1)).days


def test_missing_time_replacement_uses_declared_format_and_graph_unit():
    from graphspec.schema.columns import convert
    from graphspec.schema.missing import apply, normalize

    statement = dict(type="time", unit="week", format="%Y-%m-%d", at="end", nulls=["NA"],
                     missing=dict(name="value", value="1970-01-05"))
    converter = partial(convert, given=statement)
    with context("day"):
        values = converter(normalize(pa.array(["NA", "1970-01-01", None]), statement))
        result = apply(values, statement, "events.week", 0, converter)
    assert result.type == pa.int64()
    assert result.to_pylist() == [10, 3, 10]
