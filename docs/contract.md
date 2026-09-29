**The contract: schema, time, tasks, edges beyond pairs, held graphs, tables and bank version 8**

A source holds one or more graphs. What the graphs hold is stated once, by a schema; the tables of the source and the bank built from them are checked against it. Nothing is read from the names or the types of the tables beyond what the schema states.

**Schema**

The schema is one object, `graphspec.Schema`, built from a plain mapping and written back to one by `mapping()`. It has no defaults: a field that is absent stops the work with an error naming it. `graphspec.inspect(path)` and the verb `graphspec inspect PATH` print it for a task file or for a bank; for a bank they add the width of every kind and of every encoding of held graphs, and the relations found.

| Field | Holds |
|---|---|
| `version` | The version of the schema, 4. |
| `time` | `model`, one of `none`, `order`, `clock`, `clocks`; `units`, the names of the units in which the graphs of the source may count time, stated when the model has a clock and empty when it has none; `ties`, the name of the rule that orders facts with the same time, or `null`. |
| `observers` | `true` when facts and tasks may name observers, `false` when none does. |
| `columns` | Every column by name, each with its `type` and its `encoder`. A column is one thing wherever it is used: a name listed by two kinds is the same column, fitted once. |
| `graph` | `columns`: the attributes of the graph itself. |
| `nodes` | Per kind of node, `columns`, the columns of that kind, and `holds`, `null` or the statement of the graphs its nodes hold. Kinds keep the order stated. |
| `sets` | Per kind of set edge: `kinds`, the kinds of node its members may have; `roles`, the names of the roles a member may take; `members`, the columns of values every member carries; `columns`, the columns of the set edge itself. An empty mapping states none. |
| `edges` | Per edge table: `endpoints`, the pairs of kinds it joins, each a kind of node or a kind of set edge; `directed`, true or false; `columns`, the values its edges carry; `relation`, either `"declared"`, which makes the table one relation of its own name, or `{"column": NAME}`, which takes the relation of every row from that column; `about`, `null` when both ends of every pair are kinds of node, and otherwise the name of the rule for a time at which the set edge an edge is about does not hold. |
| `tasks` | `key`, the name of the column that identifies a task; `grouping`, the name of a column that groups tasks, or `null` when none is stated; `signals`, the width of the signal vectors, 0 for none; `references`, every reference by name; `kinds`, every kind of task by name. |

Kinds of node, kinds of set edge and edge tables have names distinct from one another. A column that the schema declares and that no kind of node or of set edge, no edge table, no graph held by a node, no kind of task and not the graph uses stops the schema with an error naming it.

A reference states its `sort` and `kinds`, the kinds its table starts from, each a kind of node or a kind of set edge; a sort that stores nothing states no kinds. A member of a kind of set edge names the set edge by its key. A kind of task states:

| Field of a kind of task | Holds |
|---|---|
| `references` | The names of its references, in the order in which a family that reads a fixed number of them takes them. |
| `columns` | Its features. |
| `target` | `null`, or `kind` and `series`: `number` with the name of the one series its targets belong to, or `path` with the list of the series its points may name. The kinds of one schema state targets of one kind. |
| `weight` | `true` when every task of the kind carries a weight above zero and finite at float32 precision, `false` when none does. A weight that is not stops the build with an error naming the task, its kind and its graph. |
| `absent` | What a reference to a node not observed at the time of the task resolves to: `stop`, which stops the build and every derivation with an error naming the reference, the task and the graph, or `keep`, which keeps the reference and gives the model nothing to read for that node. |
| `event` | `null`, or two of its references of the sort `node`: a task of the kind is then seen by later tasks as an event between the two, and its kind states `absent` as `stop`. |
| `matures` | `null`, or `reference` and `column`: the target of a task of the kind, once known, is read as the value of that column of the node the reference names. The reference is of the sort `node`, the column a column of numbers of every kind of node the reference may name, the target a number. |

Types of column are registered by name in `graphspec.TYPES`, encoders in `graphspec.ENCODERS`, encoders of held graphs in `graphspec.HELD`, sorts of reference in `graphspec.SORTS`, rules for edges about edges in `graphspec.ABOUT`, forms of storage of set edges in `graphspec.STORAGE`; each accepts registration from outside the library, and a name registered twice stops with an error.

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

Encoders are fitted on the rows of the splits trained on, which the caller of the build states by name. A row is one of them when it lies in a graph that has a target of those splits and both holds and is known no later than the last task of those splits that holds a target; in a graph without time every row of such a graph is one. Members of set edges and held graphs carry no time: every one of them in such a graph is a row trained on. A target that matures counts as a value of its column when it is known by then. The fitted state and the dictionaries are stored in the bank. A column with no value in those rows stops the build.

The width of a kind is the sum of the widths its columns encode to, and for a kind that holds graphs the width of its encoding of held graphs. Values of nodes, of set edges and of tasks are encoded into one table whose columns are the value parts of every column of the kinds of node, then of the kinds of set edge, then of the kinds of task, in the order stated, followed by their flags, followed by one block per kind that holds graphs, in the order of the kinds; a row holds zeros in the columns and blocks its kind does not have. The first linear layer of a model therefore holds one block of weights per column and per encoding of held graphs, and reads each kind through its own blocks. Values on edges are kept as stated and encoded at 32 bits when a batch is packed.

The encoded table of nodes, set edges and tasks, which is what models read, is stored at 16 bits. It holds magnitudes up to 65504 with about three significant digits; a magnitude below about 6e-5 keeps fewer digits, and one below about 6e-8 becomes 0. An encoded value beyond 65504, or not finite, stops the build, or the view that meets it, with an error naming the node or the task, its kind, its graph, the column, the value as stated and the encoder. The values as stated are kept at 32 bits.

**Edges joining sets**

A kind of set edge is a kind of its own. A set edge has a key, like a node; its facts, like the facts about a node, carry its interval, the time it was known and replaced, the observers who see it and the values of its own columns, and pass through the one guard of visibility. Its members are fixed: one row per member, naming the node, its role and the values of the member columns of the kind. A set edge has at least one member; a node is a member of one set edge at most once in one role; a member carries a value in every member column. The members of a set edge are observed at every time the set edge holds, as the ends of an edge are.

An edge table whose endpoints name a kind of set edge holds edges about set edges: an edge from or to a set edge, with an interval, known time and observers of its own. An edge that must be the end of another edge is stated as a set edge, of two members with roles when it joins two nodes. Its rule, named by `about`, says what becomes of an edge about a set edge at a time when that set edge is not shown to the task looking; rules are registered in `graphspec.ABOUT`, each with `kept(lacking, named)`, which gives for every such row whether it is kept.

| Rule | At a time when the set edge it is about is not shown |
|---|---|
| `stop` | The edge stops the build and every view with an error naming the edge, its ends, the graph and the time. |
| `lapse` | The edge is not shown at that time; it is shown at the other times it holds. |

The engine reads nodes and edges between two nodes. What a set edge becomes is chosen by the pipeline, through a parameter `sets/<kind>/storage` for every kind of set edge. The parameter has no default: a pipeline that does not state it stops naming it. Forms are registered in `graphspec.STORAGE` as `graphspec.sets.Form(function, nodes)`: a function from a graph of facts, the registry and the kind to a graph of facts, and whether it makes each set edge a node. A form that makes no node is not offered for a kind that an edge table is about, or that a reference of a kind of task stating `absent` as `stop` names.

| Form | A set edge becomes | Its relations |
|---|---|---|
| `node` | A node of its kind, with its facts as the values of the node, and for every fact and every member an edge from the node to the member, of the relation of the member's role, carrying the values of the member, with the interval, known time and observers of the fact | `<kind>.<role>`, one per role, from the kind of set edge to each kind of its members, directed, with the member columns |
| `pairs` | For every fact, one edge between every two distinct members, carrying the values of the set edge's own columns, with the interval, known time and observers of the fact; the set edge is no node, and roles and member values are not read | `<kind>`, between the kinds of its members, without direction, with the columns of the set edge |

These relations are relations of the registry like those stated: each has a code, is kept or dropped by `relations/<name>/keep`, filtered by the parameters of its edges and projected. A relation stated or taken whose name is one of them stops the build. The two form functions are `graphspec.sets.node` and `graphspec.sets.pairs`, with their entries in `graphspec.transform.CHANGES`: `node` changes facts; `pairs` changes observations and facts and may remove a referenced set edge. A reference to a set edge under a form that makes no node resolves as a node not observed, by the kind's statement of `absent`.

**Nodes that hold graphs**

A kind of node states under `holds` the schema of the graphs its nodes hold: `nodes`, per held kind, `columns`; `edges`, per held relation, `endpoints` (pairs of held kinds), `directed` and `columns`; `encoder`, the name of the encoder of held graphs. The columns are columns of the schema, fitted as any other. A node holds one graph, stated once, possibly empty; a held node has a key unique within the graph it belongs to; a held edge joins two held nodes of one graph, of kinds its relation joins in either order when it has no direction. A held node states only the columns of its held kind, and a held edge only those of its relation.

The encoder turns every held graph of the kind into one vector, which is the block of the kind in the encoded values of every fact about the node. Encoders of held graphs are registered in `graphspec.HELD`, each with `width(encoding, held)` and `encode(encoding, held, graphs)`, `graphs` holding `owners`, the number of nodes of the kind, and for every held node `owner` and `values`, and for every held edge `source` and `target`.

| Encoder | Gives, for a held graph | Width |
|---|---|---|
| `pooled` | The sum over its held nodes of their encoded columns, then the number of its held nodes | the width of its held columns plus one |
| `propagated` | The sum over its held nodes of their encoded columns, of the mean of the encoded columns of their neighbours over held edges in either direction (zero for a node without one), then the number of its held nodes | twice the width of its held columns plus one |

Neither shipped encoder reads the values of held edges; they are stored and read back.

**Operations that count**

Operations that count degree, core numbers, components or paths meet set edges and edges about edges in the form the pipeline chose. Each states in `graphspec.transform.COUNTS` what it counts, when a pipeline makes it act, and its treatment of the edges joining sets and of the edges about edges: `stored`, it counts the form as ordinary nodes and edges; or `refuses`, it stops. Before a derivation, an operation that acts on a graph whose schema states either kind, and that states no treatment of it, stops with an error naming the operation and the kind or table; one that refuses stops the same way.

| Operation | Counts | Set edges | Edges about edges |
|---|---|---|---|
| `nodes/degree`, `nodes/total_degree` | degree | stored: under `node` a member counts one edge per set edge it is in and a set edge's node counts its members and the edges about it; under `pairs` a member of a set edge of k members counts k − 1 | stored: one edge at the set edge's node and one at the other end |
| `nodes/core` | core numbers | stored, as for degree | stored |
| `derive/hops` | paths | stored: under `node` two members of one set edge are two hops apart, under `pairs` one | stored: through the set edge's node |
| `derive/projection` | paths of two edges | stored: projecting a relation of a role over its members gives the members that share a set edge | stored |
| `derive/diffusion` | components and paths | stored | stored |
| `derive/rewiring` | components and degree | stored | stored |
| `edges/cap` | degree | stored: the cap of a relation of a role counts the members of a set edge's node | stored |

**Sorts of reference**

A sort is a `graphspec.references.Sort`: whether its members are nodes, edges or both, the least and the most members a task may name, and whether their order is kept. `Sort.columns()` gives the columns its table stores, `Sort.check` checks the members of a graph, `Sort.resolved` gives for each member the observations of its nodes at the time of the task. A task that names under one of its references fewer or more members than the sort of the reference holds stops the build with an error naming the task, its kind, its graph, the reference and what the sort holds. A sort made with other numbers, or a subclass, is registered under a new name and used by a schema like the shipped ones.

| Sort | Members | Stored per member |
|---|---|---|
| `node` | one node, or one set edge where the reference names kinds of set edge | the node or the set edge |
| `edge` | one edge between two nodes | its first node, its relation, the kind and the key of its second node |
| `nodes` | one or more nodes or set edges, without order | the node or the set edge |
| `path` | one or more edges in order, each sharing a node with the one before | as `edge`, and its place from zero |
| `subgraph` | one or more nodes and edges, without order | as `edge`, the relation and the second node missing for a node |
| `graph` | the graph itself | nothing |

An edge named by a reference is of a relation the schema states, between kinds the relation joins; of a relation without direction between two nodes of one kind it is kept with the lower entity first, as facts are. A reference resolves through the view: a node to its observation at the step of the task, an edge to the observations of its two nodes, the graph to everything the step shows.

**Time**

A fact is one statement about a graph: the values of a node or of a set edge, an edge that holds, an edge stated not to hold. Every fact carries the interval in which it holds, from `since` up to and not including `until`. A time is a whole number counted in the unit its graph states. Times of different graphs are never compared.

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

Two facts about edges of one graph, both present or both absent, or two facts about one node or one set edge, that hold from the same time and that the rule leaves equal stop the build with an error naming the graph and the time. With no rule named, any two such facts that hold from the same time stop it.

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
| one per kind of node and per kind of set edge, named as the kind | node | `id INT64` key, `graph INT32`, `entity INT32`, `key STRING` | The entities of each graph: its nodes and its set edges. `entity` is unique within its graph. |
| `observation` | node | `id INT64` key, the columns of time | One fact about one node or one set edge. |
| `observed_<kind>` | relationship | FROM the kind TO `observation`; the columns of the kind | The values the fact states. |
| `member_<kind of set edge>` | relationship | FROM the kind of set edge TO each kind of its members; `role STRING`; the member columns | One member of a set edge. |
| `edge_<table>` | relationship | FROM and TO the endpoint kinds stated, kinds of set edge among them; `present BOOLEAN`; the columns of time; the relation column `STRING` when the schema states one; the columns of the table | Facts about edges. `present = false` states that the edge does not hold. |
| `held_<kind>`, for every kind that holds graphs | node | `id INT64` key, `graph INT32`, `owner STRING`, the key of the node that holds it, `kind STRING`, its held kind, `key STRING`, the columns of every held kind of the kind | One held node. A column its held kind does not state is missing. |
| `held_edge_<kind>` | relationship | FROM and TO `held_<kind>`; `relation STRING`; the columns of every held relation of the kind | One held edge. A column its relation does not state is missing. |
| `task` | node | `id INT64` key, `graph INT32`, `kind STRING`, `time INT64` unless the model is `none`, `observer STRING` when the schema states observers, `split STRING`, `weight FLOAT`, the grouping column `INT32` when one is stated, the key column `INT64`, the columns of every kind of task, and with signals `signal FLOAT[S]` and `signal_observed FLOAT[S]` | One task. A column its kind does not state is missing. |
| `task_<reference>` | relationship, for every reference whose sort stores members | FROM `task` TO the kinds the reference states; with edges `relation STRING`, `to_kind STRING`, `to_key STRING`; with order `place INT32` | One member of what a task refers to under that name. |
| `target` | node | `id INT64` key, `y DOUBLE`, `known INT64` unless the model is `none`, and where the targets are paths `tau FLOAT` and `series STRING` | One number of a target. |
| `target_of` | relationship | FROM `target` TO `task` | The task a number belongs to. |

A table the schema states and the source lacks, a column of a table that the schema does not state, a column stored as another type than stated, and a relationship joining other kinds than stated each stop with an error naming it.

- Names of kinds, columns, edge tables, references, roles and kinds of task hold no `:`. A kind of node or of set edge is not named `graph`, `observation`, `task`, `target`, `target_of` or `step` and does not start with `observed_`, `task_`, `edge_`, `member_` or `held_`. A relation is not named `diffusion`, `rewire`, `temporal` or `untyped`; those belong to derived relations. Member columns are not named `role`; held columns are not named `id`, `graph`, `owner`, `kind`, `key` or `relation`.
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

`tuning.build_bank(task, output, workers, trained)` reads the source twice: once to find the dictionaries, the relations taken from columns, the names of the observers and the values of the rows trained on, and once to write the graphs. `trained` names the splits trained on, each a split the source holds. Every series needs two targets that differ in those splits. Every graph written is also shown once under the first choice of steps, with every kind of set edge under the form `node`, which shows every fact, member and edge about a set edge, and with no rule stated for any split, so that a graph the operations cannot read stops the build. A bank of another version is refused with an error naming the version found.

A source with kinds of set edge or held graphs is read from a staging directory; a database source stating either stops with an error naming them.

`manifest.json` holds `version` 8, `task`, `trained` (the splits the encoders and the scaling were fitted on), `splits` (the names of the splits, sorted), `split_rows` (the number of rows of each), `observers` (the names of the observers, sorted), `signals`, `graphs` and `spec`. An entry of `graphs` holds the index and name of a graph, its `unit`, and its numbers of entities, of facts about nodes and set edges (`observations`), of facts about edges (`facts`), of members of set edges (`members`), of held nodes (`held`), of tasks and of targets. `spec` is the declaration of the registry: `schema`, the mapping above; `encoding`, with `dictionaries` per category column, `states` per column and `relations` per edge table; `features`, `values` and `widths`, the width of the encoded table, of its value part, and of each kind of node and of set edge; the relations with their codes; and `extras`.

| Extra | Holds |
|---|---|
| `target` | `number` or `path`, the kind of target the schema states. |
| `series` | The series in the order the kinds of task state them. |
| `signals` | The width of the signals. |
| `base` | The relations stated or taken, then the relations of the forms of storage of every kind of set edge. |
| `events` | Whether a kind of task is seen as an event. |
| `single` | Whether every time of every graph and observer holds exactly one task and every kind states a target. |

`supervision.json` holds `rows` and `split`. There is one row per number of a target, in the order of the `target` table. A row holds `graph`, `kind`, the key of its task and, where one is stated, its grouping, under the names the schema states, `time` unless the model is `none`, `y`, `weight` where the kind states one and `known` where the target states one; rows of paths add `tau` and `series`. `split` holds `indices`, the rows of every split by name, and `series`, a list with one entry per series of `name`, `mean` and `sd` over the rows of the splits trained on.

`truth/graph_NNNN/` holds one graph as arrays of facts, `graphspec.truth.Facts`, with `graph.json` giving its name, the keys of its entities, its unit, under `inner` the keys of its held nodes and under `observers` the names of the observers of the bank, which views and their errors name.

| Arrays | Hold |
|---|---|
| `kinds`, `entity_token`, `entity_held` | Per entity, its kind (kinds of node, then kinds of set edge), the hash of kind and key, and the encodings of the graph it holds in the blocks of the held graphs, zero where it holds none. |
| `obs_entity`, `obs_since`, `obs_until`, `obs_seq`, `obs_values`, `obs_x` | Per fact about a node or a set edge: the entity, the interval, the `seq` stated (0 where the rule reads none), the values as stated, one column per column of the layout and missing where the kind has no such column, and the values encoded with the entity's `entity_held`. |
| `set_edge`, `set_member`, `set_role`, `set_values` | Per member of a set edge, ordered by set edge, member and role: the set edge, the node, the position of its role in the roles of the kind, and its values as stated, one column per member column of any kind of set edge. |
| `inner_owner`, `inner_kind`, `inner_values` | Per held node, ordered by kind that holds, then node that holds, then as stated: the node that holds it, the position of its held kind, its values as stated, one column per held column of any kind. |
| `within_source`, `within_target`, `within_code`, `within_values` | Per held edge: its two held nodes, the position of its relation among the held relations of the kind, its values as stated, one column per held edge column of any kind. |
| `fact_source`, `fact_target`, `fact_code`, `fact_present`, `fact_since`, `fact_until`, `fact_seq`, `fact_values` | Per fact about an edge: its ends, its relation, whether it states presence or absence, the interval, the `seq`, and the values as stated. |
| `obs_known_row`, `obs_known_time`, `obs_replaced_row`, `obs_replaced_time`, and the same four under `fact_` | The facts whose `known` differs from `since`, and those that were replaced, each with the time. |
| `obs_seen_row`, `obs_seen_observer`, `fact_seen_row`, `fact_seen_observer` | One row per fact and observer it names; a fact without a row is seen by all. |
| `task_time`, `task_observer`, `task_grouping`, `task_key`, `task_kind`, `task_split`, `task_weight`, `task_values`, `task_x`, `task_signals` | Per task, in order of time: `task_kind` and `task_split` are positions in the kinds of the schema and in the splits of the manifest; `task_observer` and `task_split` are -1 where none is named; `task_weight` is missing where the kind states none. |
| `member_task`, `member_name`, `member_place`, `member_source`, `member_target`, `member_code` | Per member of a reference: its task, the position of its reference in the schema, its place within the reference, its node or the first node of its edge, and for an edge the second node and the relation, -1 for a node. Members are ordered by task, then by the order in which the kind states its references, then by place. |
| `target_task`, `target_row`, `target_known`, `target_y` | Per number of a target, ordered by task and row: its task, its row in the supervision, the time from which it is known and the number. |
| `graph_values` | The attributes of the graph as stated. |

A time before every time is stored as minus two to the power 62, and the absence of an end or of a time from which a target is known as two to the power 62. `Facts.stated(registry, splits, observers)` gives a graph back as it was stated: keys for entities, names for relations, kinds of task, splits and observers, times as stated, values in their stated form, every member of a reference as its first key, its relation and its second key, under `sets` every member of a set edge as the key of the set edge, the key of the node, the role and its values, and under `held` every held node as the key of its holder, its held kind, its key and values, and every held edge as the key of the holder, the keys of its two held nodes, its relation and values.

**Views**

Steps are not stored. The operations read a view: the facts of a graph shown at chosen times, as the tasks of each time saw them. `graphspec.sets.stored(facts, registry, storage)` applies the form of storage the pipeline states for every kind of set edge; `graphspec.view.sequences(facts, registry, steps, looking)` gives the sequences of the graph it returns and `graphspec.view.view(facts, registry, sequence, several)` shows one, applying the rule of every edge about a set edge; the pipeline chooses them by the parameters of its spec, and the statement of the fit by what each split may see.

| Parameter | Choices | Meaning |
|---|---|---|
| `time/steps` | registered in `graphspec.STEPS`; shipped: `tasks`, `changes` | `tasks`: one step per time at which a task sits, shown as the tasks of that time saw it. `changes`: one step per time at which a fact starts, ends, becomes known or is replaced, or at which a task sits. Under both, the tasks of each observer form a sequence of their own, and so do the tasks that name none. |
| `time/several` | registered in `graphspec.SEVERAL`; shipped: `first`, `last` | Which of several facts about one node at one step the view shows, in the order of the time they hold from, then of the rule for ties. |
| `sets/<kind>/storage`, per kind of set edge | the forms of `graphspec.STORAGE` offered for the kind; no default | The form of storage of the set edges of the kind. |

A choice of steps gives, for one graph, one sequence per observer: the observer, the times it shows, the time of looking at each, its tasks and the step of each. A step may hold no task and may hold several. A choice that leaves a task that holds a target without a step stops with an error.

What the tasks of a split may see is a list of rules, registered in `graphspec.SEES`; a rule is a function of the facts, the registry and the sequence that says which facts about nodes and about edges may be seen. A name registered twice stops with an error.

| Rule | Keeps from the tasks of the split |
|---|---|
| `without_targets` | Every fact about an edge that a task of the same split which holds a target names in a reference. |
| `without_scored` | Every fact about a node that only tasks of the splits scored on name, and every fact about an edge of such a node. It is stated for splits trained on. |

A split for which rules are stated is shown in a sequence of its own, per observer: the steps of the observer, the facts its rules leave, the targets of its own tasks, and as tasks its own, those of the splits trained on and those that name no split. The splits without rules share one sequence, which holds their tasks, those of the splits trained on and those that name no split.

A view never shows a time later than the time of looking at it. The facts about edges that hold at one step are listed in the order of the rule for ties, and facts the rule leaves equal by the time they hold from. A step of a view shows the facts as its own tasks saw them; a later task reading earlier steps does not see there what became known afterwards.

A view is a `graphspec.truth.Graph`: spans over its steps (`obs_start`, `obs_stop`, `fact_start`, `fact_stop`, `fact_rank` with the patches `order_step`, `order_position`, `order_span`, and the absent facts as `removal_*`); `entity_held`, as in the bank; its tasks with `task_step`, the step of each, `task_key`, `task_kind`, `task_values`, `task_x` and `task_signals`; the members of their references as in the bank, `member_task` numbering the tasks of the view; its targets as `label_task` and `label_row`; and `period`, the group of every step. `period` holds the grouping the tasks of the step state where the schema names one, and otherwise numbers the steps; groups are runs of equal values and need not increase. Where the schema names a grouping, every step holds a task and the tasks of one step state one group. A view may show no edge at all, as where the schema states no relation or the rules of a split hide every edge. A view needs at least one step, values of both ends of a present edge at every step it is shown, where they lack an error names the edge, its relation, the end without a value, its kind and the observer by name, where an edge about a set edge follows its rule, and, for the tasks of a kind that states `absent` as `stop`, values of every node a reference names at the step of the task. A bank keeps its views beside its variants, under `views/<digest of the parameters of time, the forms of storage and what the splits may see>/graph_NNNN/`, with `index.json` naming for each the graph it shows, its observer and its split, -1 for the sequence the splits without rules share.

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

Each of the seven stages, and each of the two shipped forms of storage, states what it changes among `time`, `observations` and `facts`, and whether it may remove a referenced node, as `references`; `graphspec.transform.CHANGES` holds the statements. The schema is checked after a stage for what it changes alone: a group for every step; the width of node values and the kinds of node; relations the schema states, between kinds the relation joins. A stage that changes observations, does not state `references` and leaves a node that a reference names without its observation at the step of the task stops with an error naming the stage and the reference.

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

A model receives the references of the tasks of a batch as a mapping from the name of each reference to its members: `row`, the task of the batch each belongs to, and, unless the reference is the graph itself, `entity` and `index`, the node and its observation in the batch, -1 where the node is not observed. An edge gives its two nodes. A set edge gives its node where the form of storage makes one, and is not observed where it makes none. `tuning.model_spec.reads` states what each family reads, and `tuning.model_spec.readable` stops a fit of a kind of task whose references the family cannot read, naming the reference.

| Family | Reads |
|---|---|
| `snapshot` without a context of earlier periods | References of every sort, any number of them, absent nodes kept. Each reference is read through a projection of its own name. |
| every other family, and `snapshot` with a context of earlier periods | One to three references of the sort `node`, in the order the kind states them, observed at the time of the task; a kind that states fewer than three gives its last reference for the places left. |

The families `signal` and `affinity` are offered where `single` holds, `affinity` where a kind of task is seen as an event.
