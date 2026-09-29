**The contract: schema, time, tables and bank version 8**

A source holds one or more graphs. What the graphs hold is stated once, by a schema; the tables of the source and the bank built from them are checked against it. Nothing is read from the names or the types of the tables beyond what the schema states.

**Schema**

The schema is one object, `graphspec.Schema`, built from a plain mapping and written back to one by `mapping()`. It has no defaults: a field that is absent stops the work with an error naming it. `graphspec.inspect(path)` and the verb `graphspec inspect PATH` print it for a task file or for a bank; for a bank they add the width of every kind and the relations found.

| Field | Holds |
|---|---|
| `version` | The version of the schema, 2. |
| `time` | `model`, one of `none`, `order`, `clock`, `clocks`; `units`, the names of the units in which the graphs of the source may count time, stated when the model has a clock and empty when it has none; `ties`, the name of the rule that orders facts with the same time, or `null`. |
| `observers` | `true` when facts and tasks may name observers, `false` when none does. |
| `columns` | Every column by name, each with its `type` and its `encoder`. A column is one thing wherever it is used: a name listed by two kinds is the same column, fitted once. |
| `graph` | `columns`: the attributes of the graph itself. |
| `nodes` | Per kind of node, `columns`: the columns of that kind. Kinds keep the order stated. |
| `edges` | Per edge table: `endpoints`, the pairs of kinds it joins; `directed`, true or false; `columns`, the values its edges carry; `relation`, either `"declared"`, which makes the table one relation of its own name, or `{"column": NAME}`, which takes the relation of every row from that column. |
| `queries` | `null`, or: `key`, the name of the column that identifies a query; `grouping`, the name of a column that groups queries, or `null` when none is stated; `columns`; `references`, three names each with the kinds it may point to; `signals`, the width of the signal vectors, 0 for none. |

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

Encoders are fitted on the rows of the splits trained on, which the caller of the build states. A row is one of them when it lies in a graph that has a task of those splits and both holds and is known no later than the last of those tasks; in a graph without time every row of such a graph is one. The fitted state and the dictionaries are stored in the bank. A column with no value in those rows stops the build.

The width of a kind is the sum of the widths its columns encode to. Node and query values are encoded into one table whose columns are the value parts of every column in the order stated, followed by their flags; a row holds zeros in the columns its kind does not have. The first linear layer of a model therefore holds one block of weights per column, and reads each kind through the blocks of its own columns. Values on edges are kept as stated and encoded when a batch is packed.

**Time**

A fact is one statement about a graph: the values of a node, an edge that holds, an edge stated not to hold. Every fact carries the interval in which it holds, from `since` up to and not including `until`. A time is a whole number counted in the unit its graph states. Times of different graphs are never compared.

| Model | The facts of a graph carry | The graph states |
|---|---|---|
| `none` | no time; every fact holds always, and a task has no time | no unit |
| `order` | `since` and `until` as positions in one order; nothing is counted between two positions | no unit |
| `clock` | `since` and `until` | one of the units of the schema |
| `clocks` | `since` and `until`, the clock of what is true, and `known` and `replaced`, the clock of what is known | one of the units of the schema |

- `since` missing means the fact held before every time; `until` missing means no end is stated.
- `known` is the time from which the fact was known; missing, it is `since`. `replaced` is the time from which the fact was replaced by a correction; missing, it never was. The correction is a fact of its own, known from that time. The bank stores `known` and `replaced` only for the facts where they differ from `since` and from never.
- A fact may name the observers that can see it, as a list of names; a fact that names none is seen by all. A task may name the one observer who is looking; a task that names none sees every fact.
- Several facts about one entity may hold at one time, and an edge may hold several times between the same two nodes.

A fact is visible to a task at a time shown when it holds at the time shown, was known no later than the task's time, was not replaced at or before the task's time, and names the task's observer or none. `graphspec.clock.visible` and `graphspec.clock.allowed` are the one place that decides this.

**Rules for ties**

Facts with the same time are ordered by the rule the schema names. Rules are registered in `graphspec.TIES`; a rule states the column it reads and gives the keys to order by.

| Rule | Reads | Orders by |
|---|---|---|
| `sequence` | `seq INT32`, stated by every fact | `seq` |
| `key` | nothing | the relation, then the key of the source, then the key of the target |

Two facts about edges of one graph, both present or both absent, or two facts about one node, that hold from the same time and that the rule leaves equal stop the build with an error naming the graph and the time. With no rule named, any two such facts that hold from the same time stop it.

**Shorthand forms**

Forms are registered by name in `graphspec.FORMS`, each a `graphspec.forms.Form` made of a function from rows to facts and of whether it counts on a clock. `graphspec.forms.reduced(name, model, rows, **parameters)` reduces rows stated in a form to facts with `since` and `until`. A form registered from outside is used the same way, and a name registered twice stops with an error.

| Form | The rows state | Parameters | Each row holds |
|---|---|---|---|
| `state` | `since`, `until` | none | over the interval stated |
| `event` | `at` | none | from `at` until the next whole time |
| `snapshot` | `at`, and `seq` where the rule for ties reads it | `times`, every time at which a snapshot was taken | from the first snapshot that states it until the first later snapshot that does not; with no end when the last snapshot states it |
| `change` | `at` | `keys`, the columns that say what changes | from `at` until the next change of the same thing; with no end after the last |
| `signal` | `index`, the number of the reading from zero | `start`, `stride` | from `start` plus `index` times `stride`, for one `stride`; it counts on a clock |

Snapshots that state `seq` must keep one order of their facts over time; a snapshot that orders two facts against an earlier one stops with an error naming its time.

**Tables of a source**

A source is a staging directory of Parquet tables or a database holding the same tables. The columns of time of a fact table are those of the model: none, `since INT64` and `until INT64`, and with two clocks `known INT64` and `replaced INT64`; then the column the rule for ties reads; then `observers STRING[]` when the schema states observers.

| Table | Kind | Columns | Meaning |
|---|---|---|---|
| `graph` | node | `id INT32` key, `name STRING`, `unit STRING` when the model has a clock, the columns of the graph | One row per graph. Names are unique. |
| one per kind of node, named as the kind | node | `id INT64` key, `graph INT32`, `entity INT32`, `key STRING` | The entities of each graph. `entity` is unique within its graph. |
| `observation` | node | `id INT64` key, the columns of time | One fact about one node. |
| `observed_<kind>` | relationship | FROM the kind TO `observation`; the columns of the kind | The values the fact states. |
| `query` | node, when the schema states queries | `id INT64` key, `graph INT32`, `category STRING`, `time INT64` unless the model is `none`, `observer STRING` when the schema states observers, the grouping column `INT32` when one is stated, the key column `INT64`, the columns of the queries, and with signals `signal FLOAT[S]` and `signal_observed FLOAT[S]` | One task. |
| `query_<reference>` | relationship | FROM `query` TO the kinds the reference states | What each query refers to under that name. |
| `edge_<table>` | relationship | FROM and TO the endpoint kinds stated; `present BOOLEAN`; the columns of time; the relation column `STRING` when the schema states one; the columns of the table | Facts about edges. `present = false` states that the edge does not hold. |
| `<target>` | node | `id INT64` key, `y DOUBLE`, `split STRING`, `time INT64` for node targets unless the model is `none`, `tau FLOAT` and `series STRING` for path targets | Labels; the name is stated by the task. |
| `<target>_of` | relationship | FROM `<target>` TO kinds of node, TO `query`, or TO `graph` | What each label belongs to. |

A table the schema states and the source lacks, a column of a table that the schema does not state, a column stored as another type than stated, and a relationship joining other kinds than stated each stop with an error naming it.

- Names of kinds, columns, edge tables and references hold no `:`. A kind is not named `graph`, `observation`, `query` or `step` and does not start with `observed_`, `query_` or `edge_`. A relation is not named `diffusion`, `rewire`, `temporal` or `untyped`; those belong to derived relations.
- The relations of an edge table that takes them from a column are the distinct values of that column over the whole source, in sorted order. Each is a relation in full, with its own code, the endpoints, direction and columns of its table. A row without a value in that column stops the build. A bank holds at most 32,767 relations stated or taken.
- An edge of a table without direction joins its two ends in no order. Between two entities of one kind it is kept with the lower entity first.
- `id` keys are unique across a whole table and increase down its rows. A staging reader resolves ids by position when they are dense from zero, or when each graph's ids are its index shifted left by 32 bits plus its row within the graph.
- A task states its time unless the model is `none`. The tasks of a graph are read in order of their time, tasks of one time in the order of their rows.
- A present fact about an edge carries a value in each of its columns.
- `category` names the kind of query; the sorted distinct categories form the vocabulary for affinity features.

**Targets**

A target is a real value. Labels live in a node table named by the task, each attached by `<target>_of`, whose destination decides the level: kinds of node (the table then has `time`), `query`, `query` with `tau` and `series` for paths, or `graph`, which attaches the label to the last task of the graph. `split` is `train` or `validation`. A node carries at most one label per time, a query or graph at most one, every graph at least one, and every `y` is finite at float32 precision. Building needs at least two training labels and two validation labels that differ. A label on a node names no observer; a graph whose tasks name observers holds no labels on nodes.

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

`tuning.build_bank(task, output, workers, trained)` reads the source twice: once to find the dictionaries, the relations taken from columns, the names of the observers and the values of the rows trained on, and once to write the graphs. Every graph written is also shown once under the choice of steps that serves the bank, so that a graph the operations cannot read stops the build. A bank of another version is refused with an error naming the version found.

`manifest.json` holds `version` 8, `task`, `target`, `trained` (the splits the encoders were fitted on), `categories`, `observers` (the names of the observers, sorted), `signals`, `graphs` and `spec`. An entry of `graphs` holds the index and name of a graph, its `unit`, and its numbers of entities, of facts about nodes (`observations`), of facts about edges (`facts`), of tasks and of labels. `spec` is the declaration of the registry: `schema`, the mapping above; `encoding`, with `dictionaries` per category column, `states` per column and `relations` per edge table; `features`, `values` and `widths`, the width of the encoded table, of its value part, and of each kind; and the relations with their codes.

`supervision.json` holds `rows` and `split`. A row holds `graph`, `y`, `time` unless the model is `none`, and for targets on queries and graphs the key of its query and, where one is stated, its grouping, under the names the schema states; path rows add `tau` and `series`, node rows add `node`. `split` holds `train_indices`, `validation_indices` and `series`, a list with one entry per series of `name`, `mean` and `sd` over the training rows. A single target is one series, named as the target table.

`truth/graph_NNNN/` holds one graph as arrays of facts, `graphspec.truth.Facts`, with `graph.json` giving its name, the keys of its entities and its unit.

| Arrays | Hold |
|---|---|
| `kinds`, `entity_token` | Per entity, its kind and the hash of kind and key. |
| `obs_entity`, `obs_since`, `obs_until`, `obs_seq`, `obs_values`, `obs_x` | Per fact about a node: the node, the interval, the `seq` stated (0 where the rule reads none), the values as stated, one column per column of the layout and missing where the kind has no such column, and the values encoded. |
| `fact_source`, `fact_target`, `fact_code`, `fact_present`, `fact_since`, `fact_until`, `fact_seq`, `fact_values` | Per fact about an edge: its ends, its relation, whether it states presence or absence, the interval, the `seq`, and the values as stated. |
| `obs_known_row`, `obs_known_time`, `obs_replaced_row`, `obs_replaced_time`, and the same four under `fact_` | The facts whose `known` differs from `since`, and those that were replaced, each with the time. |
| `obs_seen_row`, `obs_seen_observer`, `fact_seen_row`, `fact_seen_observer` | One row per fact and observer it names; a fact without a row is seen by all. |
| `query_time`, `query_observer`, `query_grouping`, `query_key`, `query_category`, `query_references`, `query_values`, `query_x`, `query_signals` | Per task, in order of time. `query_observer` is -1 where none is named. `query_references` has one column per reference, in the order the schema names them. |
| `label_entity`, `label_query`, `label_time`, `label_row`, `label_id` | Per label: its node or its task, its time, and its row in the supervision. |
| `graph_values` | The attributes of the graph as stated. |

A time before every time is stored as minus two to the power 62, and the absence of an end as two to the power 62. `Facts.stated(registry, categories, observers)` gives a graph back as it was stated: keys for entities, names for relations, categories and observers, times as stated, values in their stated form.

**Views**

Steps are not stored. The operations read a view: the facts of a graph shown at chosen times, as one task after another saw them. `graphspec.view.view(facts, registry, sequence, several)` gives it, and the pipeline chooses it by two parameters of its spec.

| Parameter | Choices | Meaning |
|---|---|---|
| `time/steps` | registered in `graphspec.STEPS`; shipped: `tasks`, offered to banks with queries, and `changes`, offered to banks without | `tasks`: one step per task, showing the task's time as the task saw it; the tasks of each observer form a sequence of steps of their own, and so does the rest. `changes`: one step per time at which a fact starts, ends, becomes known or is replaced, or at which a label sits, each seen at its own time by no observer. |
| `time/several` | registered in `graphspec.SEVERAL`; shipped: `first`, `last` | Which of several facts about one node at one step the view shows, in the order of the time they hold from, then of the rule for ties. |

A choice of steps gives, for one graph, one or more sequences; a sequence holds its observer, the times it shows, the time of looking at each, the tasks at its steps and the step of each label. A view never shows a time later than the time of looking at it. The facts about edges that hold at one step are listed in the order of the rule for ties, and facts the rule leaves equal by the time they hold from. A step of a view shows the facts as its own task saw them; a later task reading earlier steps does not see there what became known afterwards.

A view is a `graphspec.truth.Graph`: spans over its steps (`obs_start`, `obs_stop`, `fact_start`, `fact_stop`, `fact_rank` with the patches `order_step`, `order_position`, `order_span`, and the absent facts as `removal_*`), the tasks and labels of its sequence with `label_step`, and `period`, the group of every step. `period` holds the grouping the tasks state where the schema names one, and otherwise numbers the steps; groups are runs of equal values and need not increase. A view needs at least one step, values of both ends of a present edge at every step it is shown, values of the three references of a task at its step, and values of the node of a label at its step. A bank keeps its views beside its variants, under `views/<digest of the two parameters>/graph_NNNN/`, with `index.json` naming for each the graph it shows and its observer.

In the form the operations work on, `Graph.dense()`, `step_graph` names for every step the graph it came from when graphs are joined, and `x_table` and `value_table` hold the rows that `obs_ref` and `fact_ref` point to.

| Name in version 7 | In version 8 |
|---|---|
| `reward_mean`, `reward_sd` | one entry of the `series` list |
| row keys `turn`, `decision` | the grouping and the key of the queries, under the names the schema states |
| row key `step`, columns `step` of the tables | `time` of tasks and labels, `since` and `until` of facts |
| facts restated at every step, and the arrays of spans over steps | facts with intervals; the spans are the arrays of a view |
| the array that names, for every step of joined graphs, the graph the step came from | `step_graph` |
| `query_actor`, `query_subject`, `query_context` | the columns of `query_references`, named by the schema |
| `width` and `features` of one vector for every kind | `widths` per kind, `values`, `features` |

**After an operation**

Each of the seven stages states what it changes among `time`, `observations` and `facts`, and the schema is checked after it for those alone: a group for every step; the width of node values and the kinds of node; relations the schema states, between kinds the relation joins.
