import json
import logging
import sys
from datetime import datetime, timezone


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
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonLog() if json_lines else logging.Formatter("%(asctime)s %(levelname)s %(process)d %(message)s")
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)


def emit(event, **fields):
    print(json.dumps(dict(event=event, utc=datetime.now(timezone.utc).isoformat(), **fields)), flush=True)
