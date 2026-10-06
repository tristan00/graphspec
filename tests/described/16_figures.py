from graphspec import nodes, order, source, tasks

DESCRIPTION = "Quarterly figures that are revised later; predict with what was believed on the day"

releases = source(
    "figures.csv",
    reader="csv",
    columns=dict(
        firm="identifier",
        quarter="identifier",
        closes=dict(type="time", unit="day", format="%Y-%m-%d"),
        reported=dict(type="time", unit="day", format="%Y-%m-%d"),
        sales="number",
    ),
)
versions = order(
    releases,
    within=["firm", "quarter"],
    by=["reported"],
    gap=None,
    columns=dict(revised="following(reported)"),
)
firm = nodes(releases, distinct=["firm"], holds="always", known=None)
figure = nodes(
    versions,
    key=["firm", "quarter"],
    features=dict(sales="standard"),
    links=dict(firm=dict(of=firm, key="firm", absent="stop")),
    holds=dict(name="state", since="closes", until=None),
    known="reported",
    replaced="revised",
)
coming = tasks(
    versions,
    where="place == 0",
    at="reported",
    references=dict(issuer=dict(sort="node", of=firm, key="firm", absent="stop")),
    target=dict(
        kind="number",
        series="sales",
        value="first(later.sales, by=later.reported)",
        later=dict(rows=versions, on=dict(firm="firm"), at="reported", within=None),
        known="min(later.reported)",
        about="issuer",
        utility=dict(name="weighted", better="higher", weights=dict(sales=1.0)),
    ),
    split=dict(name="time", parts=dict(train=0.7, validation=0.1, test=0.2)),
)
