**The contract: schema, tables and bank version 8**

A source holds one or more graphs. What the graphs hold is stated once, by a schema; the tables of the source and the bank built from them are checked against it. Nothing is read from the names or the types of the tables beyond what the schema states.

**Schema**

The schema is one object, `graphspec.Schema`, built from a plain mapping and written back to one by `mapping()`. It has no defaults: a field that is absent stops the work with an error naming it. `graphspec.inspect(path)` and the verb `graphspec inspect PATH` print it for a task file or for a bank; for a bank they add the width of every kind and the relations found.

| Field | Holds |
|---|---|
| `version` | The version of the schema, 1. |
| `time` | `unit`: the name of the unit in which steps count. |
| `columns` | Every column by name, each with its `type` and its `encoder`. A column is one thing wherever it is used: a name listed by two kinds is the same column, fitted once. |
| `graph` | `columns`: the attributes of the graph itself. |
| `nodes` | Per kind of node, `columns`: the columns of that kind. Kinds keep the order stated. |
| `edges` | Per edge table: `endpoints`, the pairs of kinds it joins; `directed`, true or false; `columns`, the values its edges carry; `relation`, either `"declared"`, which makes the table one relation of its own name, or `{"column": NAME}`, which takes the relation of every row from that column. |
| `queries` | `null`, or: `key` and `grouping`, the names of the columns that identify a query and group the steps; `columns`; `references`, three names each with the kinds it may point to; `signals`, the width of the signal vectors, 0 for none. |

Types of column are registered by name in `graphspec.TYPES`, encoders in `graphspec.ENCODERS`; both accept registration from outside the library, and a name registered twice stops with an error.

| Type | Stored in a source as | Kept in the bank as |
|---|---|---|
| `number` | `FLOAT` | the value |
| `flag` | `BOOL` | 0 or 1 |
| `category` | `STRING` | the position of the value in the dictionary of the column, which holds every value stated |

A missing value is a null in the source and is kept as missing.

| Encoder | Reads | Gives | Fitted state |
|---|---|---|---|
| `value` | number, flag | the value, 0 where missing; a flag of presence | none |
| `logarithm` | number | sign times log of one plus the size; a flag of presence | none |
| `standard` | number | the value less the mean over the deviation, 0 where missing; a flag of presence | mean and deviation |
| `onehot` | category | one place per value seen, one more for every other value; a flag of presence | the values seen |

Encoders are fitted on the rows of the splits trained on, which the caller of the build states. A row is one of them when a task of those splits can see it: it lies in a graph that has such a task, at or before the last step of one. The fitted state and the dictionaries are stored in the bank. A column with no value in those rows stops the build.

The width of a kind is the sum of the widths its columns encode to. Node and query values are encoded into one table whose columns are the value parts of every column in the order stated, followed by their flags; a row holds zeros in the columns its kind does not have. The first linear layer of a model therefore holds one block of weights per column, and reads each kind through the blocks of its own columns. Values on edges are kept as stated and encoded when a batch is packed.

**Tables of a source**

A source is a staging directory of Parquet tables or a database holding the same tables.

| Table | Kind | Columns | Meaning |
|---|---|---|---|
| `graph` | node | `id INT32` key, `name STRING`, the columns of the graph | One row per graph. Names are unique. |
| one per kind of node, named as the kind | node | `id INT64` key, `graph INT32`, `entity INT32`, `key STRING` | The entities of each graph. `entity` is unique within its graph. |
| `observation` | node | `id INT64` key, `seq INT32`, `step INT32` | One observation of one entity at one step. |
| `observed_<kind>` | relationship | FROM the kind TO `observation`; the columns of the kind | The values of an entity at an observation. |
| `query` | node, when the schema states queries | `id INT64` key, `graph INT32`, `step INT32`, the grouping column `INT32`, the key column `INT64`, `category STRING`, the columns of the queries, and with signals `signal FLOAT[S]` and `signal_observed FLOAT[S]` | One query per step. |
| `query_<reference>` | relationship | FROM `query` TO the kinds the reference states | What each query refers to under that name. |
| `edge_<table>` | relationship | FROM and TO the endpoint kinds stated; `seq INT32`, `step INT32`, `present BOOLEAN`; the relation column `STRING` when the schema states one; the columns of the table | Facts. |
| `<target>` | node | `id INT64` key, `y DOUBLE`, `split STRING`, `step INT32` for node targets, `tau FLOAT` and `series STRING` for path targets | Labels; the name is stated by the task. |
| `<target>_of` | relationship | FROM `<target>` TO kinds of node, TO `query`, or TO `graph` | What each label belongs to. |

A table the schema states and the source lacks, a column of a table that the schema does not state, a column stored as another type than stated, and a relationship joining other kinds than stated each stop with an error naming it.

- Names of kinds, columns, edge tables and references hold no `:`. A kind is not named `graph`, `observation`, `query` or `step` and does not start with `observed_`, `query_` or `edge_`. A relation is not named `diffusion`, `rewire`, `temporal` or `untyped`; those belong to derived relations.
- The relations of an edge table that takes them from a column are the distinct values of that column over the whole source, in sorted order. Each is a relation in full, with its own code, the endpoints, direction and columns of its table. A row without a value in that column stops the build. A bank holds at most 32,767 relations stated or taken.
- An edge of a table without direction joins its two ends in no order. Between two entities of one kind it is kept with the lower entity first.
- `id` keys are unique across a whole table and increase down its rows. A staging reader resolves ids by position when they are dense from zero, or when each graph's ids are its index shifted left by 32 bits plus its row within the graph.

**Time and causality**

- With queries, query steps in each graph number 0 to n−1 with no gaps, and every graph has at least one query. Without queries, a graph's steps run from 0 to its last observation, fact or label, and a step may be empty.
- An observation or fact at step `s` is visible to a query or node label at step `s` and every later one, never to an earlier one.
- `seq` orders facts within a step. `(step, seq)` is unique among observations, among present facts across all relations, and among absent facts.
- An entity has at most one observation per step. A present fact needs observations of both of its entities at its step, a query needs observations of its three references at its step, and a node label needs an observation of its node at its step.
- A present fact holds at its own step only, so a relation that persists is restated at every step. A row with `present = false` removes it, and needs both entities observed at or before its step. A present fact carries a value in each of its columns.
- The grouping column of the queries groups steps and never decreases. Without queries every step is its own group.
- `category` names the kind of query; the sorted distinct categories form the vocabulary for affinity features.

**Targets**

A target is a real value. Labels live in a node table named by the task, each attached by `<target>_of`, whose destination decides the level: kinds of node (the table then has `step`), `query`, `query` with `tau` and `series` for paths, or `graph`. `split` is `train` or `validation`. A node carries at most one label per step, a query or graph at most one, every graph at least one, and every `y` is finite at float32 precision. Building needs at least two training labels and two validation labels that differ.

**Task file**

| Key | Meaning |
|---|---|
| `name` | The name of the bank's registry, part of its fingerprint. |
| `staging` | The staging directory, relative to the task file. |
| `database` | The database, relative to the task file, read when no staging is named. |
| `target` | The target table; `<target>_of` must exist. |
| `schema` | The schema, as a mapping. |

`name`, `target`, `schema` and one source are required, and no other keys are accepted.

**Bank version 8**

`tuning.build_bank(task, output, workers, trained)` reads the source twice: once to find the dictionaries, the relations taken from columns and the values of the rows trained on, and once to write the graphs. A bank of another version is refused with an error naming the version found.

`manifest.json` holds `version` 8, `task`, `target`, `trained` (the splits the encoders were fitted on), `categories`, `signals`, `graphs` and `spec`. `spec` is the declaration of the registry: `schema`, the mapping above; `encoding`, with `dictionaries` per category column, `states` per column and `relations` per edge table; `features`, `values` and `widths`, the width of the encoded table, of its value part, and of each kind; and the relations with their codes.

`supervision.json` holds `rows` and `split`. A row holds `graph`, `step`, `y`, and for targets on queries and graphs the key and the grouping of its query under the names the schema states; path rows add `tau` and `series`, node rows add `node`. `split` holds `train_indices`, `validation_indices` and `series`, a list with one entry per series of `name`, `mean` and `sd` over the training rows. A single target is one series, named as the target table.

`truth/graph_NNNN/` holds one graph as arrays, with `graph.json` giving its name, the keys of its entities and its number of steps.

| Arrays | Hold |
|---|---|
| `kinds`, `entity_token` | Per entity, its kind and the hash of kind and key. |
| `period` | Per step, its group. |
| `obs_entity`, `obs_start`, `obs_stop`, `obs_values`, `obs_x` | Spans of consecutive steps over which an entity's stated values are the same. `obs_values` holds the values as stated, one column per column of the layout, missing where the kind has no such column; `obs_x` holds them encoded. |
| `fact_source`, `fact_target`, `fact_code`, `fact_start`, `fact_stop`, `fact_rank`, `fact_values`, `order_step`, `order_position`, `order_span` | Spans of present facts, their values as stated, and their order within each step. |
| `removal_source`, `removal_target`, `removal_code`, `removal_step`, `removal_values` | Absent facts. |
| `query_key`, `query_category`, `query_references`, `query_values`, `query_x`, `query_signals` | Per query. `query_references` has one column per reference, in the order the schema names them. |
| `label_entity`, `label_step`, `label_row`, `label_id` | Per label, its row in the supervision. |
| `graph_values` | The attributes of the graph as stated. |

`Graph.stated(registry, categories)` gives a graph back as it was stated: keys for entities, names for relations and categories, values in their stated form.

In the form the operations work on, `step_graph` names for every step the graph it came from when graphs are joined, and `x_table` and `value_table` hold the rows that `obs_ref` and `fact_ref` point to.

| Name in version 7 | In version 8 |
|---|---|
| `reward_mean`, `reward_sd` | one entry of the `series` list |
| row keys `turn`, `decision` | the grouping and the key of the queries, under the names the schema states |
| the array that names, for every step of joined graphs, the graph the step came from | `step_graph` |
| `query_actor`, `query_subject`, `query_context` | the columns of `query_references`, named by the schema |
| `width` and `features` of one vector for every kind | `widths` per kind, `values`, `features` |

**After an operation**

Each of the seven stages states what it changes among `time`, `observations` and `facts`, and the schema is checked after it for those alone: a group for every step; the width of node values and the kinds of node; relations the schema states, between kinds the relation joins.
