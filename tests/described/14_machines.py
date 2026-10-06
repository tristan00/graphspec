from graphspec import join, nodes, source, tasks

DESCRIPTION = (
    "Machines with logs of sensors; predict the days until failure, where most had not failed "
    "when recording ended"
)

fleet = source(
    "machines.csv",
    reader="csv",
    columns=dict(machine="identifier", model="category"),
)
logs = source(
    "logs.parquet",
    reader="parquet",
    columns=dict(
        machine="identifier",
        day=dict(type="time", unit="day"),
        heat=dict(type="number", missing="omit"),
        shake=dict(type="number", missing="omit"),
        failed=dict(type="flag", outcome="day"),
    ),
)
kept = join(logs, fleet, on=dict(machine="machine"), take=dict(model="model"), unmatched="stop")
machine = nodes(
    kept,
    key="machine",
    features=dict(model="onehot", heat="standard", shake="standard"),
    holds=dict(name="change", at="day"),
    known=None,
)
failure = tasks(
    kept,
    at="day",
    references=dict(unit=dict(sort="node", of=machine, key="machine", absent="stop")),
    target=dict(
        kind="survival",
        series="failure",
        width=7,
        intervals=8,
        value=dict(
            duration="(min(later.day, where=later.failed) if any(later.failed) else max(later.day)) - day",
            happened="any(later.failed)",
        ),
        later=dict(rows=logs, on=dict(machine="machine"), at="day", within=None),
        known="min(later.day, where=later.failed) if any(later.failed) else max(later.day)",
        about="unit",
        utility=dict(name="weighted", better="lower", weights=dict(failure=1.0)),
    ),
    split=dict(name="hash", key="machine", parts=dict(train=0.7, validation=0.15, test=0.15), salt="machines"),
)
