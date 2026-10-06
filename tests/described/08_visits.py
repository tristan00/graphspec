from graphspec import closure, edges, group, join, nodes, sets, source, tasks

DESCRIPTION = (
    "Visits in order per patient, each with sets of codes; codes under a table of parents, "
    "all ancestors joined"
)

stays = source(
    "visits.csv",
    reader="csv",
    columns=dict(
        patient="identifier",
        visit="identifier",
        number=dict(type="time", unit=None),
        ward="category",
    ),
)
found = source(
    "diagnoses.csv",
    reader="csv",
    columns=dict(visit="identifier", code="identifier"),
)
tree = source(
    "parents.csv",
    reader="csv",
    columns=dict(code="identifier", parent=dict(type="identifier", missing="omit")),
)
lineage = closure(tree, source="code", target="parent", most=None)
patient = nodes(stays, distinct=["patient"], holds="always", known=None)
code = nodes(tree, key="code", holds="always", known=None)
under = edges(
    lineage,
    source=dict(of=code, key="code", absent="stop"),
    target=dict(of=code, key="parent", absent="stop"),
    directed=True,
    features=dict(steps="value"),
    holds="always",
    known=None,
)
visit = sets(
    stays,
    key="visit",
    features=dict(ward="onehot"),
    holds=dict(name="event", at="number"),
    known=None,
    members=[
        dict(rows=stays, set="visit", node=dict(of=patient, key="patient", absent="stop"), role="'patient'"),
        dict(rows=found, set="visit", node=dict(of=code, key="code", absent="skip"), role="'code'"),
    ],
)
coded = group(found, by=["visit"], columns=dict(codes="list(code)"))
listed = join(stays, coded, on=dict(visit="visit"), take=dict(codes="codes"), unmatched="drop")
coming = tasks(
    listed,
    at="number",
    references=dict(who=dict(sort="node", of=patient, key="patient", absent="stop")),
    target=dict(
        kind="classes",
        value="first(later.codes, by=later.number)",
        later=dict(rows=listed, on=dict(patient="patient"), at="number", within=None),
        known="min(later.number)",
        about="who",
        utility=dict(name="weighted", better="higher", weights=1.0),
    ),
    split=dict(name="hash", key="patient", parts=dict(train=0.8, validation=0.1, test=0.1), salt="visits"),
)
