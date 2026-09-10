import os
import threading

BUDGET_S = 120


def pytest_configure(config):
    capture = config.pluginmanager.getplugin('capturemanager')

    def expire():
        if capture is not None:
            capture.suspend_global_capture(in_=True)
        os.write(2, b'suite exceeded the %d second budget\n' % BUDGET_S)
        os._exit(124)

    config._budget = threading.Timer(BUDGET_S, expire)
    config._budget.daemon = True
    config._budget.start()


def pytest_unconfigure(config):
    config._budget.cancel()
