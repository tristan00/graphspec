import json
import logging
import sys
import time
from datetime import datetime, timezone

LOG = logging.getLogger(__name__)


class JsonLog(logging.Formatter):
    def format(self, record):
        return json.dumps(
            dict(
                event="log",
                utc=datetime.now(timezone.utc).isoformat(),
                logger=record.name,
                message=record.getMessage(),
            )
        )


def configure(level=logging.INFO, json_lines=False):
    started = time.perf_counter()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonLog() if json_lines else logging.Formatter("%(asctime)s %(levelname)s %(process)d %(message)s")
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)
    LOG.debug("configure exit json_lines=%s seconds=%.4f", json_lines, time.perf_counter() - started)


def emit(event, **fields):
    print(json.dumps(dict(event=event, utc=datetime.now(timezone.utc).isoformat(), **fields)), flush=True)
