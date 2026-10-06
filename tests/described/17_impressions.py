from graphspec import decisions, nodes, source

DESCRIPTION = (
    "Impressions in which a policy showed items in three positions, each with the probability of being "
    "shown; the actor is the policy, the outcome is the click"
)

shown = source(
    "impressions.csv",
    reader="csv",
    columns=dict(
        impression="identifier",
        second=dict(type="time", unit="second"),
        policy="identifier",
        position="number",
        item="identifier",
        chance="number",
        click=dict(type="flag", outcome="second + 1"),
    ),
)
item = nodes(shown, distinct=["item"], holds="always", known=None)
policy = nodes(shown, distinct=["policy"], holds="always", known=None)
impression = decisions(
    shown,
    key="impression",
    at="second",
    references=dict(shower=dict(sort="node", of=policy, key="policy", absent="stop")),
    actors="shower",
    series=["click"],
    about=None,
    utility=dict(name="weighted", better="higher", weights=dict(click=1.0)),
    split=dict(name="time", parts=dict(train=0.6, validation=0.2, test=0.2)),
    options=dict(
        rows=shown,
        decision="impression",
        key="position",
        chosen="True",
        available="True",
        together="'set'",
        order="position",
        logged="chance",
        outcome=dict(click="1.0 if click else 0.0"),
        known="second + 1",
    ),
    actions=dict(
        rows=shown,
        decision="impression",
        option="position",
        kind="'show'",
        actor="policy",
        kinds=dict(
            show=dict(
                roles=dict(what=dict(sort="node", of=item, keys="item", count=[1, 1])),
                parameters=dict(),
            ),
        ),
    ),
)
