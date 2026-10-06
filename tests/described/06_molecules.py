from graphspec import edges, nodes, source, tasks

DESCRIPTION = (
    "Many small graphs in one file with a column naming the graph, nodes identified by row number; "
    "a target per graph"
)

atoms = source(
    "atoms.csv",
    reader="csv",
    columns=dict(molecule="identifier", element="category", charge="number"),
)
bonds = source(
    "bonds.csv",
    reader="csv",
    columns=dict(one="identifier", other="identifier", order="number"),
)
labels = source(
    "molecules.csv",
    reader="csv",
    columns=dict(molecule="identifier", mutagenic=dict(type="category", outcome="never")),
)
atom = nodes(
    atoms,
    key="row()",
    graph="molecule",
    features=dict(element="onehot", charge="standard"),
)
bond = edges(
    bonds,
    source=dict(of=atom, key="one", absent="stop"),
    target=dict(of=atom, key="other", absent="stop"),
    directed=False,
    features=dict(order="value"),
)
mutagenic = tasks(
    labels,
    graph="molecule",
    references=dict(whole=dict(sort="graph")),
    target=dict(
        kind="class",
        value="mutagenic",
        known="never",
        about=None,
        utility=dict(name="weighted", better="higher", weights=1.0),
    ),
    split=dict(name="graph", parts=dict(train=0.8, validation=0.1, test=0.1), salt="molecules"),
)
