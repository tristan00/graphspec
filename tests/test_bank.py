import shutil

import numpy as np
import pyarrow as pa

from graph_tuning_infra import database as db
from tuning.gather import neighbor_table

from .graphs import WIDTH, vector
from .support import config, equal


def neighbors(arrays, entity, cutoff, floor, count, spec, stream):
    from tuning.data import NEIGHBOURS, sampling

    mode, decay = sampling(spec)
    ticks = np.ascontiguousarray(np.asarray(arrays['events'])[:, 2])
    table, sizes = neighbor_table(np.asarray(arrays['neighbor_ids']), np.asarray(arrays['neighbor_offsets']), ticks, np.array([entity]), np.array([cutoff]),
                                  floor, count, mode, decay, stream, NEIGHBOURS)
    return table[0, :sizes[0]]


def database(path):
    stage = db.Stage(path.with_name(path.name + '.staging'))
    kinds = dict(customer=range(0, 4), policy=range(4, 9))
    units = {('graph', None, None): pa.table(dict(id=pa.array([0], pa.int32()), name=pa.array(['book'], pa.string())))}
    for kind, members in kinds.items():
        members = list(members)
        units[(kind, None, None)] = pa.table(dict(id=pa.array(members, pa.int64()), graph=pa.array([0] * len(members), pa.int32()),
                                                  entity=pa.array(members, pa.int32()), key=pa.array([kind + str(m) for m in members], pa.string())))
    rows, observed, holds, labels, targets = [], {kind: ([], []) for kind in kinds}, [], [], []
    for step in range(6):
        for position, entity in enumerate(entity for entity in range(9) if (entity, step) != (8, 1)):
            rows.append((len(rows), position, step, vector(entity, step % 3)))
            kind = 'customer' if entity < 4 else 'policy'
            observed[kind][0].append(entity)
            observed[kind][1].append(len(rows) - 1)
        for sequence, policy in enumerate(policy for policy in range(4, 9) if (policy, step) != (8, 1)):
            holds.append((policy % 4, policy, sequence, step))
    for step in (2, 5):
        for customer in range(4):
            targets.append((len(targets), float(customer % 2 + step), 'validation' if customer == 3 else 'train', step))
            labels.append((len(labels), customer))
    units[('observation', None, None)] = pa.table(dict(id=pa.array([o[0] for o in rows], pa.int64()), seq=pa.array([o[1] for o in rows], pa.int32()),
        step=pa.array([o[2] for o in rows], pa.int32()),
        x=pa.FixedSizeListArray.from_arrays(pa.array(np.stack([o[3] for o in rows]).reshape(-1)), 2 * WIDTH)))
    for kind, (sources, destinations) in observed.items():
        units[('observed', kind, 'observation')] = pa.table({'from': pa.array(sources, pa.int64()), 'to': pa.array(destinations, pa.int64())})
    units[('edge_holds', 'customer', 'policy')] = pa.table({'from': pa.array([h[0] for h in holds], pa.int64()), 'to': pa.array([h[1] for h in holds], pa.int64()),
        'seq': pa.array([h[2] for h in holds], pa.int32()), 'step': pa.array([h[3] for h in holds], pa.int32()), 'present': pa.array([True] * len(holds))})
    units[('label', None, None)] = pa.table(dict(id=pa.array([t[0] for t in targets], pa.int64()), y=pa.array([t[1] for t in targets], pa.float64()),
        split=pa.array([t[2] for t in targets], pa.string()), step=pa.array([t[3] for t in targets], pa.int32())))
    units[('label_of', 'label', 'customer')] = pa.table({'from': pa.array([l[0] for l in labels], pa.int64()), 'to': pa.array([l[1] for l in labels], pa.int64())})
    stage.write(units)
    stage.close()
    store, connection = db.connect(path, read_only=False)
    for statement in ('CREATE NODE TABLE graph(id INT32 PRIMARY KEY, name STRING)',
                      'CREATE NODE TABLE customer(id INT64 PRIMARY KEY, graph INT32, entity INT32, key STRING)',
                      'CREATE NODE TABLE policy(id INT64 PRIMARY KEY, graph INT32, entity INT32, key STRING)',
                      'CREATE NODE TABLE observation(id INT64 PRIMARY KEY, seq INT32, step INT32, x FLOAT[%d])' % (2 * WIDTH),
                      'CREATE NODE TABLE label(id INT64 PRIMARY KEY, y DOUBLE, split STRING, step INT32)',
                      'CREATE REL TABLE observed(FROM customer TO observation, FROM policy TO observation)',
                      'CREATE REL TABLE label_of(FROM label TO customer)',
                      'CREATE REL TABLE edge_holds(FROM customer TO policy, seq INT32, step INT32, present BOOLEAN)'):
        connection.execute(statement)
    db.copy(connection, stage)
    connection.close()
    store.close()
    shutil.rmtree(stage.directory)


def test_bank(book):
    import torch

    from tuning.batching import pack_batch
    from tuning.data import Bank, pack
    from tuning.turns import blocks as turn_blocks

    bank = Bank(book)
    assert bank.manifest['rows'] == 8 and bank.manifest['validation_rows'] == 2
    assert bank.registry.extras['protected'] == ['customer']
    views = ({'view': 'snapshot', 'snapshot/spatial': 'normalised'}, {'view': 'mixed'}, {'view': 'snapshot', 'snapshot/temporal': 'coupling_edges'},
             {'view': 'event_graph'}, {'view': 'tree'}, {'view': 'sequence', 'sequence/operator': 'mixer'}, {'view': 'walks'})
    for values in views:
        cfg = config(bank.registry, values, {'time/history': 3, 'time/context': 1})
        b = pack(bank, bank.blocks(0, 3, cfg)[-1], cfg, 'cpu')
        assert b['predictions'] == 3 and int(b['selected'].sum()) == 3, values
    cfg = config(bank.registry, {'view': 'snapshot', 'snapshot/temporal': 'coupling_edges'}, {'time/history': 1, 'time/context': 1})
    extended = turn_blocks(bank, bank.blocks(0, 1, cfg), cfg)
    batch, parts = pack_batch(bank, extended, cfg, 'cpu'), [pack(bank, block, cfg, 'cpu') for block in extended]
    assert (parts[0]['step'] < 0).any() and all(len(part['x']) == part['observations'] for part in parts)
    assert torch.equal(batch['x'][batch['context_index']], torch.cat([part['x'][part['context_index']] for part in parts]))
    assert torch.equal(batch['x'][batch['temporal_edges']], torch.cat([part['x'][part['temporal_edges']] for part in parts], 1))
    cfg = config(bank.registry, {'view': 'tree'}, {'time/history': 1, 'time/context': 1, 'time/periods': 'every', 'time/period_length': 2})
    assert [block[1:3] for block in turn_blocks(bank, bank.blocks(0, 1, cfg), cfg)] == [(0, 3), (2, 6)]
    arrays, spec = bank.arrays(0, cfg), cfg['graph']
    memory = config(bank.registry, {'view': 'memory'}, {'time/history': 5, 'time/context': 1, 'time/periods': 'every', 'time/period_length': 2, 'neighbors/order': 'uniform'})
    assert bank.variant(cfg)[0] == bank.variant(memory)[0]
    ids = arrays['neighbor_ids'][arrays['neighbor_offsets'][0]:arrays['neighbor_offsets'][1]]
    window = ids[:np.searchsorted(arrays['events'][ids, 2], 12)]
    assert len(window) == 2 and equal(neighbors(arrays, 0, 12, 0, 1, spec, 7), window[-1:])
    assert equal(neighbors(arrays, 0, 12, 0, 4, dict(spec, **{'neighbors/order': 'uniform'}), 7), window)
    events = np.zeros((4, 9), np.int64)
    events[:, 1], events[:, 2] = 1, [0, 2, 4, 6]
    toy = dict(neighbor_ids=np.arange(4), neighbor_offsets=np.array([0, 4, 4]), events=events)
    for order in ('uniform', 'decay'):
        settings = dict(spec, **{'neighbors/order': order, 'neighbors/decay': 0.5})
        for stream in range(16):
            wide, same_floor, narrow = (neighbors(toy, 0, 8, floor, 2, settings, stream) for floor in (0, -4, 1))
            assert len(wide) == 2 and equal(wide, same_floor) and set(wide[wide >= 1].tolist()) <= set(narrow.tolist())
        chosen = neighbors(arrays, 0, 12, 0, 1, settings, 7)
        assert len(chosen) == 1 and np.isin(chosen, window).all() and equal(chosen, neighbors(arrays, 0, 12, 0, 1, settings, 7))
