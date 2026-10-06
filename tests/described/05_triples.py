from graphspec import edges, nodes, source, tasks

DESCRIPTION = (
    "Triples of head, relation and tail with thousands of relations, some about other triples; "
    "predict the tail"
)

stated = source(
    "triples.csv",
    reader="csv",
    columns=dict(
        id="identifier",
        head=dict(type="identifier", missing="omit"),
        about=dict(type="identifier", missing="omit"),
        relation="category",
        tail="identifier",
    ),
)
entity = nodes(stated, distinct=["head", "tail"])
fact = edges(
    stated,
    where="about is missing",
    key="id",
    source=dict(of=entity, key="head", absent="stop"),
    target=dict(of=entity, key="tail", absent="stop"),
    relation="relation",
    directed=True,
)
qualifier = edges(
    stated,
    where="about is not missing",
    source=dict(of=fact, key="about", absent="skip"),
    target=dict(of=entity, key="tail", absent="stop"),
    relation="relation",
    directed=True,
    about="lapse",
)
tail = tasks(
    stated,
    where="about is missing",
    references=dict(head=dict(sort="node", of=entity, key="head", absent="stop")),
    features=dict(relation="onehot"),
    target=dict(
        kind="link",
        series="tail",
        relation=fact,
        value="tail",
        known=None,
        about="head",
        utility=dict(name="weighted", better="higher", weights=1.0),
    ),
    split=dict(name="hash", key="id", parts=dict(train=0.8, validation=0.1, test=0.1), salt="triples"),
)
