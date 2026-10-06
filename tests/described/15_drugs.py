from graphspec import edges, nodes, source, tasks

DESCRIPTION = "Drugs that are each a graph of atoms; predict which pairs interact"

atoms = source(
    "atoms.csv",
    reader="csv",
    columns=dict(drug="identifier", atom="identifier", element="category"),
)
bonds = source(
    "bonds.csv",
    reader="csv",
    columns=dict(drug="identifier", one="identifier", other="identifier", order="number"),
)
trials = source(
    "pairs.csv",
    reader="csv",
    columns=dict(one="identifier", other="identifier", interact=dict(type="flag", outcome=None)),
)
drug = nodes(atoms, distinct=["drug"], held="propagated")
atom = nodes(
    atoms,
    key="atom",
    inside=dict(of=drug, key="drug", absent="stop"),
    features=dict(element="onehot"),
)
bond = edges(
    bonds,
    inside=dict(of=drug, key="drug", absent="stop"),
    source=dict(of=atom, key="one", absent="stop"),
    target=dict(of=atom, key="other", absent="stop"),
    directed=False,
    features=dict(order="value"),
)
interacts = tasks(
    trials,
    references=dict(
        one=dict(sort="node", of=drug, key="one", absent="stop"),
        other=dict(sort="node", of=drug, key="other", absent="stop"),
    ),
    target=dict(
        kind="class",
        value="'interacts' if interact else 'apart'",
        known=None,
        about=None,
        utility=dict(name="weighted", better="higher", weights=dict(interacts=1.0, apart=0.0)),
    ),
    split=dict(name="hash", key=["one", "other"], parts=dict(train=0.8, validation=0.2), salt="drugs"),
)
