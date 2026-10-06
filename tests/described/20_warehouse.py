from graphspec import edges, join, nodes, sets, source, tasks

DESCRIPTION = (
    "A warehouse. `orders.csv` has one row per order: its number, the customer, when it was placed, and "
    "when it shipped, which is empty until it ships. `lines.csv` has one row per line of an order: the "
    "order, the product, the quantity, the shelf it was picked from and the worker who picked it, with the "
    "time of the pick. `shelves.csv` gives each shelf its aisle and height; `workers.csv` each worker's "
    "shift and the day they started; `products.csv` each product's weight and category, some weights "
    "missing. For each order, at the moment it is placed, predict whether it will ship more than two days "
    "later; the model may see only what was known at that moment, and orders placed in the last month of "
    "the data are for scoring."
)

orders = source(
    "orders.csv",
    reader="csv",
    columns=dict(
        number="identifier",
        customer="identifier",
        placed=dict(type="time", unit="minute", format="%Y-%m-%d %H:%M"),
        shipped=dict(
            type="time",
            unit="minute",
            format="%Y-%m-%d %H:%M",
            missing="omit",
            outcome="shipped",
        ),
    ),
    through="max(placed)",
)
lines = source(
    "lines.csv",
    reader="csv",
    columns=dict(
        order="identifier",
        product="identifier",
        quantity="number",
        shelf="identifier",
        worker="identifier",
        picked=dict(type="time", unit="minute", format="%Y-%m-%d %H:%M"),
    ),
)
shelves = source(
    "shelves.csv",
    reader="csv",
    columns=dict(shelf="identifier", aisle="category", height="number"),
)
workers = source(
    "workers.csv",
    reader="csv",
    columns=dict(
        worker="identifier",
        shift="category",
        started=dict(type="time", unit="day", at="start", format="%Y-%m-%d"),
    ),
)
products = source(
    "products.csv",
    reader="csv",
    columns=dict(product="identifier", weight=dict(type="number", missing="omit"), category="category"),
)
listed = join(lines, orders, on=dict(number="order"), take=dict(placed="placed"), unmatched="stop")
order = nodes(
    orders,
    key="number",
    links=dict(customer="customer"),
    holds=dict(name="state", since="placed", until=None),
    known=None,
)
product = nodes(
    products,
    key="product",
    features=dict(weight="standard", category="onehot"),
    holds="always",
    known=None,
)
shelf = nodes(
    shelves,
    key="shelf",
    features=dict(aisle="onehot", height="standard"),
    holds="always",
    known=None,
)
worker = nodes(
    workers,
    key="worker",
    features=dict(shift="onehot"),
    holds=dict(name="state", since="started", until=None),
    known=None,
)
contains = edges(
    listed,
    source=dict(of=order, key="order", absent="stop"),
    target=dict(of=product, key="product", absent="stop"),
    directed=True,
    features=dict(quantity="standard"),
    holds=dict(name="state", since="placed", until=None),
    known=None,
    ties=dict(name="sequence", seq="row()"),
)
pick = sets(
    lines,
    key="row()",
    holds=dict(name="event", at="picked"),
    known=None,
    members=[
        dict(rows=lines, set="row()", node=dict(of=order, key="order", absent="stop"), role="'order'"),
        dict(rows=lines, set="row()", node=dict(of=product, key="product", absent="stop"), role="'product'"),
        dict(rows=lines, set="row()", node=dict(of=shelf, key="shelf", absent="stop"), role="'shelf'"),
        dict(rows=lines, set="row()", node=dict(of=worker, key="worker", absent="stop"), role="'worker'"),
    ],
)
shipment = edges(
    orders,
    where="shipped is not missing",
    source=dict(of=order, key="number", absent="stop"),
    target=dict(of=order.customer, key="customer", absent="stop"),
    directed=True,
    holds=dict(name="event", at="shipped"),
    known=None,
)
late = tasks(
    orders,
    at="placed",
    references=dict(order=dict(sort="node", of=order, key="number", absent="stop")),
    target=dict(
        kind="class",
        value="'prompt' if met(shipped, placed + span(2, 'day')) else 'late'",
        known="placed + span(2, 'day')",
        about="order",
        utility=dict(name="weighted", better="lower", weights=dict(late=1.0, prompt=0.0)),
    ),
    split=dict(name="time", parts=dict(train=None, score="span(1, 'month')")),
)
