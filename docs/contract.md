**The contract: schema, time, tasks, tables and bank version 8**

A source holds one or more graphs. What the graphs hold is stated once, by a schema; the tables of the source and the bank built from them are checked against it. Nothing is read from the names or the types of the tables beyond what the schema states.

**Schema**

The schema is one object, `graphspec.Schema`, built from a plain mapping and written back to one by `mapping()`. It has no defaults: a field that is absent stops the work with an error naming it. `graphspec.inspect(path)` and the verb `graphspec inspect PATH` print it for a task file or for a bank; for a bank they add the width of every kind and the relations found.

| Field | Holds |
|---|---|
| `version` | The version of the schema, 3. |
| `time` | `model`, one of `none`, `order`, `clock`, `clocks`; `units`, the names of the units in which the graphs of the source may count time, stated when the model has a clock and empty when it has none; `ties`, the name of the rule that orders facts with the same time, or `null`. |
| `observers` | `true` when facts and tasks may name observers, `false` when none does. |
| `columns` | Every column by name, each with its `type` and its `encoder`. A column is one thing wherever it is used: a name listed by two kinds is the same column, fitted once. |
| `graph` | `columns`: the attributes of the graph itself. |
| `nodes` | Per kind of node, `columns`: the columns of that kind. Kinds keep the order stated. |
| `edges` | Per edge table: `endpoints`, the pairs of kinds it joins; `directed`, true or false; `columns`, the values its edges carry; `relation`, either `"declared"`, which makes the table one relation of its own name, or `{"column": NAME}`, which takes the relation of every row from that column. |
| `tasks` | `key`, the name of the column that identifies a task; `grouping`, the name of a column that groups tasks, or `null` when none is stated; `signals`, the width of the signal vectors, 0 for none; `references`, every reference by name; `kinds`, every kind of task by name. |

A reference states its `sort` and `kinds`, the kinds of node its table starts from; a sort that stores nothing states no kinds. A kind of task states:

| Field of a kind of task | Holds |
|---|---|
| `references` | The names of its references, in the order in which a family that reads a fixed number of them takes them. |
| `columns` | Its features. |
| `target` | `null`, or `kind` and `series`: `number` with the name of the one series its targets belong to, or `path` with the list of the series its points may name. The kinds of one schema state targets of one kind. |
| `weight` | `true` when every task of the kind carries a weight above zero, `false` when none does. |
| `absent` | What a reference to a node not observed at the time of the task resolves to: `stop`, which stops the build and every derivation with an error naming the reference, the task and the graph, or `keep`, which keeps the reference and gives the model nothing to read for that node. |
| `event` | `null`, or two of its references of the sort `node`: a task of the kind is then seen by later tasks as an event between the two, and its kind states `absent` as `stop`. |
| `matures` | `null`, or `reference` and `column`: the target of a task of the kind, once known, is read as the value of that column of the node the reference names. The reference is of the sort `node`, the column a column of numbers of every kind of node the reference may name, the target a number. |

Types of column are registered by name in `graphspec.TYPES`, encoders in `graphspec.ENCODERS`, sorts of reference in `graphspec.SORTS`; each accepts registration from outside the library, and a name registered twice stops with an error.

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

Encoders are fitted on the rows of the splits trained on, which the caller of the build states by name. A row is one of them when it lies in a graph that has a target of those splits and both holds and is known no later than the last task of those splits that holds a target; in a graph without time every row of such a graph is one. A target that matures counts as a value of its column when it is known by then. The fitted state and the dictionaries are stored in the bank. A column with no value in those rows stops the build.

The width of a kind is the sum of the widths its columns encode to. Values of nodes and of tasks are encoded into one table whose columns are the value parts of every column in the order stated, followed by their flags; a row holds zeros in the columns its kind does not have. The first linear layer of a model therefore holds one block of weights per column, and reads each kind through the blocks of its own columns. Values on edges are kept as stated and encoded when a batch is packed.

**Sorts of reference**

A sort is a `graphspec.references.Sort`: whether its members are nodes, edges or both, the least and the most members a task may name, and whether their order is kept. `Sort.columns()` gives the columns its table stores, `Sort.check` checks the members of a graph, `Sort.resolved` gives for each member the observations of its nodes at the time of the task. A sort made with other numbers, or a subclass, is registered under a new name and used by a schema like the shipped ones.

| Sort | Members | Stored per member |
|---|---|---|
| `node` | one node | the node |
| `edge` | one edge between two nodes | its first node, its relation, the kind and the key of its second node |
| `nodes` | one or more nodes, without order | the node |
| `path` | one or more edges in order, each sharing a node with the one before | as `edge`, and its place from zero |
| `subgraph` | one or more nodes and edges, without order | as `edge`, the relation and the second node missing for a node |
| `graph` | the graph itself | nothing |

An edge named by a reference is of a relation the schema states, between kinds the relation joins; of a relation without direction between two nodes of one kind it is kept with the lower entity first, as facts are. A reference resolves through the view: a node to its observation at the step of the task, an edge to the observations of its two nodes, the graph to everything the step shows.

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

A fact is visible to a task at a time shown when it holds at the time shown, was known no later than the task's time, was not replaced at or before the task's time, names the task's observer or none, and is not kept from the split of the task by a rule the fit states. `graphspec.clock.visible` and `graphspec.clock.allowed` are the one place that decides this.

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

**Tasks**

A task is one row of the table of tasks of its graph. It states its kind, its time unless the model is `none`, the observer who is looking where the schema states observers, its key, its grouping where one is stated, the features of its kind, its signals where the schema states a width, its weight where its kind states one, its split where it holds a target, and under each reference of its kind the members the sort of the reference holds. A task of a kind that states a target may hold one: a number, or points of a path, each with its `tau` above zero and its series. A target states the time from which it is known, or none; that time is later than the time of its task, so that a target never enters the view of its own task.

- Several tasks of one graph may share a time, and a time at which facts change may hold no task.
- A split is any name. A task that holds a target states one; a task that holds none may state none.
- A task of a kind holds at most one number. Two points of one task, one series and one `tau` stop the build.
- A target whose kind states what matures is read, from the time it is known, by every later task as the value of the stated column of the node it names, until the next target of the same node and column is known.

**Tables of a source**

A source is a staging directory of Parquet tables or a database holding the same tables. The columns of time of a fact table are those of the model: none, `since INT64` and `until INT64`, and with two clocks `known INT64` and `replaced INT64`; then the column the rule for ties reads; then `observers STRING[]` when the schema states observers.

| Table | Kind | Columns | Meaning |
|---|---|---|---|
| `graph` | node | `id INT32` key, `name STRING`, `unit STRING` when the model has a clock, the columns of the graph | One row per graph. Names are unique. |
| one per kind of node, named as the kind | node | `id INT64` key, `graph INT32`, `entity INT32`, `key STRING` | The entities of each graph. `entity` is unique within its graph. |
| `observation` | node | `id INT64` key, the columns of time | One fact about one node. |
| `observed_<kind>` | relationship | FROM the kind TO `observation`; the columns of the kind | The values the fact states. |
| `edge_<table>` | relationship | FROM and TO the endpoint kinds stated; `present BOOLEAN`; the columns of time; the relation column `STRING` when the schema states one; the columns of the table | Facts about edges. `present = false` states that the edge does not hold. |
| `task` | node | `id INT64` key, `graph INT32`, `kind STRING`, `time INT64` unless the model is `none`, `observer STRING` when the schema states observers, `split STRING`, `weight FLOAT`, the grouping column `INT32` when one is stated, the key column `INT64`, the columns of every kind of task, and with signals `signal FLOAT[S]` and `signal_observed FLOAT[S]` | One task. A column its kind does not state is missing. |
| `task_<reference>` | relationship, for every reference whose sort stores members | FROM `task` TO the kinds the reference states; with edges `relation STRING`, `to_kind STRING`, `to_key STRING`; with order `place INT32` | One member of what a task refers to under that name. |
| `target` | node | `id INT64` key, `y DOUBLE`, `known INT64` unless the model is `none`, and where the targets are paths `tau FLOAT` and `series STRING` | One number of a target. |
| `target_of` | relationship | FROM `target` TO `task` | The task a number belongs to. |

A table the schema states and the source lacks, a column of a table that the schema does not state, a column stored as another type than stated, and a relationship joining other kinds than stated each stop with an error naming it.

- Names of kinds, columns, edge tables, references and kinds of task hold no `:`. A kind of node is not named `graph`, `observation`, `task`, `target`, `target_of` or `step` and does not start with `observed_`, `task_` or `edge_`. A relation is not named `diffusion`, `rewire`, `temporal` or `untyped`; those belong to derived relations.
- The relations of an edge table that takes them from a column are the distinct values of that column over the whole source, in sorted order. Each is a relation in full, with its own code, the endpoints, direction and columns of its table. A row without a value in that column stops the build. A bank holds at most 32,767 relations stated or taken.
- An edge of a table without direction joins its two ends in no order. Between two entities of one kind it is kept with the lower entity first.
- `id` keys are unique across a whole table and increase down its rows. A staging reader resolves ids by position when they are dense from zero, or when each graph's ids are its index shifted left by 32 bits plus its row within the graph.
- The tasks of a graph are kept in order of their time, tasks of one time in the order of their rows. The key of a task is unique within its graph.
- A present fact about an edge carries a value in each of its columns.
- Every `y` is finite at float32 precision.

**Task file**

| Key | Meaning |
|---|---|
| `name` | The name of the bank's registry, part of its fingerprint. |
| `staging` | The staging directory, relative to the task file. |
| `database` | The database, relative to the task file, read when no staging is named. |
| `schema` | The schema, as a mapping. |

`name`, `schema` and one source are required, and no other keys are accepted.

**Bank version 8**

`tuning.build_bank(task, output, workers, trained)` reads the source twice: once to find the dictionaries, the relations taken from columns, the names of the observers and the values of the rows trained on, and once to write the graphs. `trained` names the splits trained on, each a split the source holds. Every series needs two targets that differ in those splits. Every graph written is also shown once under the first choice of steps, with no rule stated for any split, so that a graph the operations cannot read stops the build. A bank of another version is refused with an error naming the version found.

`manifest.json` holds `version` 8, `task`, `trained` (the splits the encoders and the scaling were fitted on), `splits` (the names of the splits, sorted), `split_rows` (the number of rows of each), `observers` (the names of the observers, sorted), `signals`, `graphs` and `spec`. An entry of `graphs` holds the index and name of a graph, its `unit`, and its numbers of entities, of facts about nodes (`observations`), of facts about edges (`facts`), of tasks and of targets. `spec` is the declaration of the registry: `schema`, the mapping above; `encoding`, with `dictionaries` per category column, `states` per column and `relations` per edge table; `features`, `values` and `widths`, the width of the encoded table, of its value part, and of each kind; the relations with their codes; and `extras`.

| Extra | Holds |
|---|---|
| `target` | `number` or `path`, the kind of target the schema states. |
| `series` | The series in the order the kinds of task state them. |
| `signals` | The width of the signals. |
| `base` | The relations stated or taken. |
| `events` | Whether a kind of task is seen as an event. |
| `single` | Whether every time of every graph and observer holds exactly one task and every kind states a target. |

`supervision.json` holds `rows` and `split`. There is one row per number of a target, in the order of the `target` table. A row holds `graph`, `kind`, the key of its task and, where one is stated, its grouping, under the names the schema states, `time` unless the model is `none`, `y`, `weight` where the kind states one and `known` where the target states one; rows of paths add `tau` and `series`. `split` holds `indices`, the rows of every split by name, and `series`, a list with one entry per series of `name`, `mean` and `sd` over the rows of the splits trained on.

`truth/graph_NNNN/` holds one graph as arrays of facts, `graphspec.truth.Facts`, with `graph.json` giving its name, the keys of its entities and its unit.

| Arrays | Hold |
|---|---|
| `kinds`, `entity_token` | Per entity, its kind and the hash of kind and key. |
| `obs_entity`, `obs_since`, `obs_until`, `obs_seq`, `obs_values`, `obs_x` | Per fact about a node: the node, the interval, the `seq` stated (0 where the rule reads none), the values as stated, one column per column of the layout and missing where the kind has no such column, and the values encoded. |
| `fact_source`, `fact_target`, `fact_code`, `fact_present`, `fact_since`, `fact_until`, `fact_seq`, `fact_values` | Per fact about an edge: its ends, its relation, whether it states presence or absence, the interval, the `seq`, and the values as stated. |
| `obs_known_row`, `obs_known_time`, `obs_replaced_row`, `obs_replaced_time`, and the same four under `fact_` | The facts whose `known` differs from `since`, and those that were replaced, each with the time. |
| `obs_seen_row`, `obs_seen_observer`, `fact_seen_row`, `fact_seen_observer` | One row per fact and observer it names; a fact without a row is seen by all. |
| `task_time`, `task_observer`, `task_grouping`, `task_key`, `task_kind`, `task_split`, `task_weight`, `task_values`, `task_x`, `task_signals` | Per task, in order of time: `task_kind` and `task_split` are positions in the kinds of the schema and in the splits of the manifest; `task_observer` and `task_split` are -1 where none is named; `task_weight` is missing where the kind states none. |
| `member_task`, `member_name`, `member_place`, `member_source`, `member_target`, `member_code` | Per member of a reference: its task, the position of its reference in the schema, its place within the reference, its node or the first node of its edge, and for an edge the second node and the relation, -1 for a node. Members are ordered by task, then by the order in which the kind states its references, then by place. |
| `target_task`, `target_row`, `target_known`, `target_y` | Per number of a target, ordered by task and row: its task, its row in the supervision, the time from which it is known and the number. |
| `graph_values` | The attributes of the graph as stated. |

A time before every time is stored as minus two to the power 62, and the absence of an end or of a time from which a target is known as two to the power 62. `Facts.stated(registry, splits, observers)` gives a graph back as it was stated: keys for entities, names for relations, kinds of task, splits and observers, times as stated, values in their stated form, and every member of a reference as its first key, its relation and its second key.

**Views**

Steps are not stored. The operations read a view: the facts of a graph shown at chosen times, as the tasks of each time saw them. `graphspec.view.sequences(facts, registry, steps, looking)` gives the sequences of a graph and `graphspec.view.view(facts, registry, sequence, several)` shows one; the pipeline chooses them by two parameters of its spec, and the statement of the fit by what each split may see.

| Parameter | Choices | Meaning |
|---|---|---|
| `time/steps` | registered in `graphspec.STEPS`; shipped: `tasks`, `changes` | `tasks`: one step per time at which a task sits, shown as the tasks of that time saw it. `changes`: one step per time at which a fact starts, ends, becomes known or is replaced, or at which a task sits. Under both, the tasks of each observer form a sequence of their own, and so do the tasks that name none. |
| `time/several` | registered in `graphspec.SEVERAL`; shipped: `first`, `last` | Which of several facts about one node at one step the view shows, in the order of the time they hold from, then of the rule for ties. |

A choice of steps gives, for one graph, one sequence per observer: the observer, the times it shows, the time of looking at each, its tasks and the step of each. A step may hold no task and may hold several. A choice that leaves a task that holds a target without a step stops with an error.

What the tasks of a split may see is a list of rules, registered in `graphspec.SEES`; a rule is a function of the facts, the registry and the sequence that says which facts about nodes and about edges may be seen. A name registered twice stops with an error.

| Rule | Keeps from the tasks of the split |
|---|---|
| `without_targets` | Every fact about an edge that a task of the same split which holds a target names in a reference. |
| `without_scored` | Every fact about a node that only tasks of the splits scored on name, and every fact about an edge of such a node. It is stated for splits trained on. |

A split for which rules are stated is shown in a sequence of its own, per observer: the steps of the observer, the facts its rules leave, the targets of its own tasks, and as tasks its own, those of the splits trained on and those that name no split. The splits without rules share one sequence, which holds their tasks, those of the splits trained on and those that name no split.

A view never shows a time later than the time of looking at it. The facts about edges that hold at one step are listed in the order of the rule for ties, and facts the rule leaves equal by the time they hold from. A step of a view shows the facts as its own tasks saw them; a later task reading earlier steps does not see there what became known afterwards.

A view is a `graphspec.truth.Graph`: spans over its steps (`obs_start`, `obs_stop`, `fact_start`, `fact_stop`, `fact_rank` with the patches `order_step`, `order_position`, `order_span`, and the absent facts as `removal_*`); its tasks with `task_step`, the step of each, `task_key`, `task_kind`, `task_values`, `task_x` and `task_signals`; the members of their references as in the bank, `member_task` numbering the tasks of the view; its targets as `label_task` and `label_row`; and `period`, the group of every step. `period` holds the grouping the tasks of the step state where the schema names one, and otherwise numbers the steps; groups are runs of equal values and need not increase. Where the schema names a grouping, every step holds a task and the tasks of one step state one group. A view needs at least one step, values of both ends of a present edge at every step it is shown, and, for the tasks of a kind that states `absent` as `stop`, values of every node a reference names at the step of the task. A bank keeps its views beside its variants, under `views/<digest of the two parameters and of what the splits may see>/graph_NNNN/`, with `index.json` naming for each the graph it shows, its observer and its split, -1 for the sequence the splits without rules share.

In the form the operations work on, `Graph.dense()`, `step_graph` names for every step the graph it came from when graphs are joined, and `x_table` and `value_table` hold the rows that `obs_ref` and `fact_ref` point to.

| Name in version 7 | In version 8 |
|---|---|
| the two names of scaling of a single target | one entry of the `series` list |
| the row keys that named a level of time and a query | the grouping and the key of the tasks, under the names the schema states |
| row key `step`, columns `step` of the tables | `time` of tasks, `since` and `until` of facts |
| facts restated at every step, and the arrays of spans over steps | facts with intervals; the spans are the arrays of a view |
| the array that names, for every step of joined graphs, the graph the step came from | `step_graph` |
| the three arrays of roles of a query, the arrays of queries and of labels | the table of tasks with its members and its targets |
| the two lists of rows of the training and the validation split | `indices`, the rows of every split by name |
| `width` and `features` of one vector for every kind | `widths` per kind, `values`, `features` |

**After an operation**

Each of the seven stages states what it changes among `time`, `observations` and `facts`, and whether it may remove a referenced node, as `references`; `graphspec.transform.CHANGES` holds the statements. The schema is checked after a stage for what it changes alone: a group for every step; the width of node values and the kinds of node; relations the schema states, between kinds the relation joins. A stage that changes observations, does not state `references` and leaves a node that a reference names without its observation at the step of the task stops with an error naming the stage and the reference.

The stage of nodes states `references`. Its parameter `nodes/referenced` says what becomes of a referenced node that the stage would remove: `keep` leaves its observation at the step of the task, without its edges; `remove` removes it, and is offered where every kind of task states `absent` as `keep`.

**The statement of a fit and the splits**

The statement of a fit, `tuning.statement.Statement`, holds beside its other fields:

| Field | Holds |
|---|---|
| `trained` | The names of the splits trained on, at least one. They are the splits the bank was built for. |
| `scored` | The names of the splits scored on, at least one, none of them trained on. |
| `sees` | For every split trained on or scored on, the list of the rules of what its tasks may see; an empty list states that they see every fact. |

A configuration carries the same three under `splits`, and a fit whose configuration and statement differ in them stops. The rows of a split neither trained on nor scored on enter no fit.

**What a family reads**

A model receives the references of the tasks of a batch as a mapping from the name of each reference to its members: `row`, the task of the batch each belongs to, and, unless the reference is the graph itself, `entity` and `index`, the node and its observation in the batch, -1 where the node is not observed. An edge gives its two nodes. `tuning.model_spec.reads` states what each family reads, and `tuning.model_spec.readable` stops a fit of a kind of task whose references the family cannot read, naming the reference.

| Family | Reads |
|---|---|
| `snapshot` without a context of earlier periods | References of every sort, any number of them, absent nodes kept. Each reference is read through a projection of its own name. |
| every other family, and `snapshot` with a context of earlier periods | One to three references of the sort `node`, in the order the kind states them, observed at the time of the task; a kind that states fewer than three gives its last reference for the places left. |

The families `signal` and `affinity` are offered where `single` holds, `affinity` where a kind of task is seen as an event.
