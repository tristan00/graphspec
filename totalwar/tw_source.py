import logging
import time


LOG = logging.getLogger(__name__)


def open_corpus(window):
    started = time.perf_counter()
    from advisor.reward_data import LiveCorpus
    from decisions.store import DecisionStore

    store = DecisionStore(readonly=True)
    store.con.execute('BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY')
    corpus = LiveCorpus(store, window, float('inf'))
    LOG.info('tw corpus open exit seconds=%.3f window=%d rows=%d', time.perf_counter() - started, window, len(corpus.rows))
    return store, corpus
