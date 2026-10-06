from graphspec import edges, nested, nodes, source, tasks

DESCRIPTION = "Trees of nested records with ordered children, and two more sets of edges over the same nodes"

programs = source(
    "programs.json",
    reader="json",
    columns=dict(
        program="identifier",
        root="nested",
        name=dict(type="list", of="category", outcome=None),
    ),
)
flows = source(
    "flows.csv",
    reader="csv",
    columns=dict(program="identifier", written="identifier", read="identifier"),
)
tokens = source(
    "tokens.csv",
    reader="csv",
    columns=dict(program="identifier", before="identifier", after="identifier"),
)
spread = nested(
    programs,
    record="root",
    children="children",
    columns=dict(form="category", leaf="flag"),
    depth=None,
)
part = nodes(
    spread,
    key="path",
    graph="program",
    features=dict(form="onehot", leaf="value"),
)
child = edges(
    spread,
    where="parent is not missing",
    graph="program",
    source=dict(of=part, key="parent", absent="stop"),
    target=dict(of=part, key="path", absent="stop"),
    directed=True,
    features=dict(place="value"),
)
flow = edges(
    flows,
    graph="program",
    source=dict(of=part, key="written", absent="stop"),
    target=dict(of=part, key="read", absent="stop"),
    directed=True,
)
succession = edges(
    tokens,
    graph="program",
    source=dict(of=part, key="before", absent="stop"),
    target=dict(of=part, key="after", absent="stop"),
    directed=True,
)
named = tasks(
    programs,
    graph="program",
    references=dict(whole=dict(sort="graph")),
    target=dict(
        kind="sequence",
        length=5,
        value="name",
        known=None,
        about=None,
        utility=dict(name="weighted", better="higher", weights=1.0),
    ),
    split=dict(name="graph", parts=dict(train=0.8, validation=0.1, test=0.1), salt="trees"),
)
