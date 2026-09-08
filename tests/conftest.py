import os
import threading

BUDGET_S = 20


def pytest_configure(config):
    def expire():
        print("suite exceeded the %d second budget" % BUDGET_S, flush=True)
        os._exit(124)

    config._budget = threading.Timer(BUDGET_S, expire)
    config._budget.daemon = True
    config._budget.start()


def pytest_unconfigure(config):
    config._budget.cancel()
