from graphspec import database, edges, tasks

DESCRIPTION = "A database with declared keys and views: a target, the columns of time, and who is eligible"

shop = database(
    "shop.sqlite",
    reader="sqlite",
    tables=["*"],
    unit="day",
    times="typed",
    through="time('2025-06-30')",
    encoders=dict(number="standard", category="onehot", flag="value"),
    missing="omit",
    absent="stop",
)
review = edges(
    shop.review,
    source=dict(of=shop.customer, key="customer", absent="stop"),
    target=dict(of=shop.item, key="item", absent="stop"),
    directed=True,
    features=dict(stars="standard"),
    holds=dict(name="event", at="written"),
    known=None,
)
bought = edges(
    shop.purchases,
    source=dict(of=shop.customer, key="customer", absent="stop"),
    target=dict(of=shop.item, key="item", absent="stop"),
    directed=True,
    features=dict(quantity="logarithm"),
    holds=dict(name="event", at="placed"),
    known=None,
)
leaves = tasks(
    dict(
        name="nodes",
        of=shop.customer,
        times=dict(first="time('2024-01-01')", last="time('2025-03-01')", every=30),
    ),
    at="moment",
    earlier=dict(rows=shop.orders, on=dict(customer="id"), at="placed", within=90),
    eligible="count(earlier.id) > 0",
    references=dict(buyer=dict(sort="node", of=shop.customer, key="id", absent="stop")),
    target=dict(
        kind="class",
        value="'leaves' if count(later.id) == 0 else 'stays'",
        later=dict(rows=shop.orders, on=dict(customer="id"), at="placed", within=90),
        known="moment + 90",
        about="buyer",
        utility=dict(name="weighted", better="lower", weights=dict(leaves=1.0, stays=0.0)),
    ),
    split=dict(name="time", parts=dict(train=0.7, validation=0.15, test=0.15)),
)
