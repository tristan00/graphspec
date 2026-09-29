**The graph table contract**

A staging or database holds one or more graphs. A graph is one connected history: a whole book of business, a campaign, a
portfolio. One graph is enough, because labels on its nodes carry their own step and split. Time inside a graph is
a sequence of steps 0, 1, 2, … Each step holds the observations and relationship facts known at that step, and
exactly one query when the database has queries.

| Table | Kind | Columns | Meaning |
|---|---|---|---|
| `graph` | node | `id INT32` key, `name STRING` | One row per graph. Names are unique. |
| one per node type, such as `customer` | node | `id INT64` key, `graph INT32`, `entity INT32`, `key STRING` | The entities of each graph. `entity` is unique within its graph; only uniqueness matters. `key` is a readable label. |
| `observation` | node | `id INT64` key, `seq INT32`, `step INT32`, `x FLOAT[F]` | An entity's feature vector at a step. |
| `observed` | relationship | FROM every node type TO `observation` | Links each observation to its entity. |
| `query` | node, optional | `id INT64` key, `graph INT32`, `step INT32`, `period INT32`, `key INT64`, `category STRING`, `x FLOAT[F]`, optionally `signal FLOAT[S]` and `signal_observed FLOAT[S]` | Decision moments, when the history has them. |
| `query_actor`, `query_subject`, `query_context` | relationship, with `query` | FROM `query` TO node types | The three entities each query is about. |
| `edge_<relation>` | relationship | FROM/TO pairs between node types; `seq INT32`, `step INT32`, `present BOOLEAN`; any `FLOAT` attributes | Relationship facts. |
| `<target>` | node | `id INT64` key, `y DOUBLE`, `split STRING`, `step INT32` for node targets, `tau FLOAT` and `series STRING` for path targets | Labels; the name is yours. |
| `<target>_of` | relationship | FROM `<target>` TO node types, TO `query`, or TO `graph` | What each label belongs to. |

Structure:

- Node types are the source tables of `observed`, in creation order. Relations are the `edge_` tables, also in
  creation order. Order sets the relation codes and the fingerprint, so recreating tables in another order
  builds a different bank.
- The `edge_` prefix exists because node and relationship tables share one case-insensitive namespace.
  Every relationship column other than `seq`, `step` and `present` is an attribute and must be `FLOAT`.
- Relation names may not contain `:` or be `diffusion`, `rewire`, `temporal` or `untyped`; those belong to
  derived relations. Node types may not be
  named `graph`, `observation`, `observed`, `query`, `query_actor`, `query_subject`, `query_context` or `step`, and may
  not start with `edge_` or contain `:`.
- `observation.x` and `query.x` share one even width `F`. The first half holds values and the second half
  holds presence flags; `graphspec.infra.features.encoder(F // 2, numbers, categories)` builds such vectors by
  feature hashing. The bank stores them at float16 precision.
- `id` keys are unique across a whole table, not per graph, and increase down the table's rows. A staging reader resolves ids by position when they are dense from zero, or when each graph's ids are its index shifted left by 32 bits plus its row within the graph, which is how shards written in parallel stay unique.

Time and causality:

- With queries, query steps in each graph number 0 to n−1 with no gaps, and every graph has at least one query.
  Without queries, a graph's steps run from 0 to its last observation, fact or label, and a step may be empty.
- An observation or fact at step `s` is visible to a query or node label at step `s` and every later one, never
  to an earlier one.
- `seq` orders facts within a step. Observations within a step are ordered by entity. `(step, seq)` must be unique among observations, among present facts across
  all relations, and among absent facts.
- An entity has at most one observation per step. A present fact needs observations of both of its entities at
  its step, a query needs observations of its actor, subject and context at its step, and a node label needs
  an observation of its node at its step. An unchanged vector adds nothing to the event history, which records
  an observation only when `x` changes.
- A present fact holds at its own step only: the snapshot at step `s` contains exactly the facts stated at
  step `s`, so a relation that persists is restated at every step. The event history records a relation when
  it becomes present, and a removal when a `present = false` row arrives. A removal needs both entities
  observed at or before its step.
- Each query has exactly one `query_actor`, one `query_subject` and one `query_context` edge.
- `period` groups query steps into turns — a game turn, a billing cycle, a day — and never decreases. Without
  queries every step is its own period. A spec keeps these periods or replaces them with fixed-length ones.
- `key` is the query's own identifier, kept as `decision` in the bank. `category` names the kind of query;
  the sorted distinct categories form the vocabulary for affinity features.
- `signal` and `signal_observed` are optional per-query channels with an observed mask. The signal model is
  offered only when they exist and the target is on queries or graphs.

**Example schema**

A renewal model for insurance customers. Each graph is one customer's history; each step is a month.

```cypher
CREATE NODE TABLE graph(id INT32 PRIMARY KEY, name STRING);
CREATE NODE TABLE customer(id INT64 PRIMARY KEY, graph INT32, entity INT32, key STRING);
CREATE NODE TABLE policy(id INT64 PRIMARY KEY, graph INT32, entity INT32, key STRING);
CREATE NODE TABLE agent(id INT64 PRIMARY KEY, graph INT32, entity INT32, key STRING);
CREATE NODE TABLE region(id INT64 PRIMARY KEY, graph INT32, entity INT32, key STRING);
CREATE NODE TABLE observation(id INT64 PRIMARY KEY, seq INT32, step INT32, x FLOAT[128]);
CREATE NODE TABLE query(id INT64 PRIMARY KEY, graph INT32, step INT32, period INT32, key INT64, category STRING, x FLOAT[128]);
CREATE NODE TABLE renewal(id INT64 PRIMARY KEY, y DOUBLE, split STRING);
CREATE REL TABLE observed(FROM customer TO observation, FROM policy TO observation, FROM agent TO observation, FROM region TO observation);
CREATE REL TABLE query_actor(FROM query TO policy);
CREATE REL TABLE query_subject(FROM query TO customer);
CREATE REL TABLE query_context(FROM query TO agent);
CREATE REL TABLE edge_holds(FROM customer TO policy, seq INT32, step INT32, present BOOLEAN);
CREATE REL TABLE edge_sold_by(FROM policy TO agent, seq INT32, step INT32, present BOOLEAN, commission FLOAT);
CREATE REL TABLE edge_located_in(FROM customer TO region, FROM agent TO region, seq INT32, step INT32, present BOOLEAN);
CREATE REL TABLE renewal_of(FROM renewal TO query);
```

Load node tables first, then relationships, with `COPY` from Parquet. A relationship file starts with the source
and target `id` columns, followed by the properties in declared order. Load one file per FROM/TO pair and name
the pair, as `graphspec.infra.database.copy` does:

```cypher
COPY customer FROM 'customer.parquet';
COPY observation FROM 'observation.parquet';
COPY observed FROM 'observed_customer.parquet' (from='customer', to='observation');
COPY edge_located_in FROM 'located_in_agent.parquet' (from='agent', to='region');
```

**Targets on nodes, queries or graphs**

A target is any real value: a reward, an outcome, an amount. Keep the labels in a node table of your choosing,
attach each one with `<target>_of`, and name the table in the task file. The destination of `<target>_of`
decides the level.

- **On nodes** (`FROM renewal TO customer`, to one or more node types): the target table also has `step INT32`.
  Each label is predicted for its node at that step from everything visible then, so the node needs an
  observation at that step. A node can carry labels at several steps, one per step. Queries are optional.
- **On a query** (`FROM renewal TO query`): one label per query, predicted at that query's step from
  everything visible then. Queries without a label still feed the history.
- **On a query, as paths** (`FROM renewal TO query` with `tau FLOAT` and `series STRING` columns): any number
  of points per query, each a value `y` of the named series at elapsed time `tau` after the query's step,
  predicted from everything visible at that step. One query may carry several series, such as GDP and
  population. `tau` is finite and strictly positive, no two points of one query and series share a `tau`, and
  every point of one query carries the same split. Each series is standardised with its own training mean and
  deviation, the loss averages within a query and series, then over queries, then over series, and the score
  is that same loss on the validation points. A query without points still feeds the history.
- **On a graph** (`FROM renewal TO graph`): one label per graph, predicted at the graph's last query.

Query and graph targets need the query tables. A single graph with node targets needs no query tables, only
the target table and its relationship next to the tables above:

```cypher
CREATE NODE TABLE renewal(id INT64 PRIMARY KEY, y DOUBLE, split STRING, step INT32);
CREATE REL TABLE renewal_of(FROM renewal TO customer);
```

For every level:

- `split` is `train`, `validation` or `spread`, set per label. `spread` points are never trained on and never scored against their `y`. They mark one-step graphs named `<graph>@<query key>#<block>`, each the state of `<graph>` at that query with another block of actions. Scoring puts each of them in place of that step of `<graph>` and predicts the query's validation points, so the range of predictions over the blocks of one decision can enter the trial score. Such graphs are left out of training, validation and variant preparation.
- A node carries at most one label per step, a query or graph at most one label, every graph at least one, and
  every `y` is finite at float32 precision.
- Building needs at least two training labels and two validation labels that differ at float32 precision.
- Training targets are standardised with the training mean and sample standard deviation. Fits are scored by
  raw MSE and R² on the whole validation split. Path targets are scored by their validation loss, the
  training loss on the validation points, overall and per series; they have no R².

No spec removes or filters the node types a node target labels, or an observation a query needs at its step.

**Task file**

```json
{
  "name": "retention",
  "database": "retention.lbug",
  "staging": "retention.staging",
  "target": "renewal"
}
```

| Key | Meaning |
|---|---|
| `name` | The registry name, part of the fingerprint. |
| `database` | Database path, relative to the task file. |
| `staging` | Optional: the staging directory, relative to the task file. When named, the bank is built from its Parquet tables; missing staging raises an error. |
| `target` | The target table; `<target>_of` must exist. |

`name`, `target`, and one source (`staging` or `database`) are required and no other keys are accepted. Everything else a variant needs comes from the spec
a trial proposes.

