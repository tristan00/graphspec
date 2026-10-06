from graphspec import join, nodes, source, tasks, unpivot

DESCRIPTION = (
    "A power grid. `substations.csv` has one row per substation: its name, its region and its voltage class. "
    "`lines.csv` has one row per power line between two substations: its length and capacity, the day it "
    "entered service and, for some, the day it was retired. `load.parquet` holds a reading of the load of "
    "every substation every fifteen minutes, in one wide table with one column per substation. `outages.csv` "
    "has one row per outage of a line: when it began, when it ended, and its cause, which is entered by an "
    "engineer some days after the outage ends, the day of entry being a column. For every line, at the start "
    "of every day it is in service, predict whether it will have an outage within the next seven days; the "
    "readings and the causes may be used only as they were known at that moment; the lines of two named "
    "regions are kept apart for scoring."
)

substations = source(
    "substations.csv",
    reader="csv",
    columns=dict(name="identifier", region="category", voltage="category"),
)
lines = source(
    "lines.csv",
    reader="csv",
    columns=dict(
        line="identifier",
        origin="identifier",
        destination="identifier",
        length="number",
        capacity="number",
        entered=dict(type="time", unit="day", at="start", format="%Y-%m-%d"),
        retired=dict(type="time", unit="day", at="start", format="%Y-%m-%d", missing="omit"),
    ),
)
loads = source(
    "load.parquet",
    reader="parquet",
    columns={"time": dict(type="time", unit="minute"), "*": "number"},
)
outages = source(
    "outages.csv",
    reader="csv",
    columns=dict(
        line="identifier",
        began=dict(type="time", unit="minute", format="%Y-%m-%d %H:%M"),
        ended=dict(type="time", unit="minute", format="%Y-%m-%d %H:%M"),
        cause=dict(type="category", outcome="recorded"),
        recorded=dict(type="time", unit="day", at="end", format="%Y-%m-%d"),
    ),
    through="max(began)",
)
sited = join(lines, substations, on=dict(name="origin"), take=dict(near="region"), unmatched="stop")
placed = join(sited, substations, on=dict(name="destination"), take=dict(far="region"), unmatched="stop")
readings = unpivot(loads, keep=["time"], name="substation", value="load")
substation = nodes(
    substations,
    key="name",
    features=dict(region="onehot", voltage="onehot"),
    holds="always",
    known=None,
)
reading = nodes(
    readings,
    of=substation,
    key="substation",
    features=dict(load="standard"),
    holds=dict(name="change", at="time"),
    known=None,
)
line = nodes(
    placed,
    key="line",
    features=dict(length="standard", capacity="standard"),
    links=dict(
        origin=dict(of=substation, key="origin", absent="stop"),
        destination=dict(of=substation, key="destination", absent="stop"),
    ),
    holds=dict(name="state", since="entered", until="retired"),
    known=None,
)
outage = nodes(
    outages,
    key="row()",
    features=dict(cause="onehot"),
    links=dict(line=dict(of=line, key="line", absent="stop")),
    holds=dict(name="state", since="began", until="ended"),
    known=None,
)
risk = tasks(
    dict(name="nodes", of=line, times=dict(first=None, last=None, every="span(1, 'day')")),
    at="moment",
    references=dict(line=dict(sort="node", of=line, key="line", absent="stop")),
    target=dict(
        kind="class",
        value="'outage' if count(later.line) > 0 else 'clear'",
        later=dict(rows=outages, on=dict(line="line"), at="began", until="ended", within="span(7, 'day')"),
        known="moment + span(7, 'day')",
        about="line",
        utility=dict(name="weighted", better="lower", weights=dict(outage=1.0, clear=0.0)),
    ),
    split="'score' if near in ['north', 'coast'] or far in ['north', 'coast'] else 'train'",
)
