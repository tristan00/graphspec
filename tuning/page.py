import json
import logging
import time
from pathlib import Path

LOG = logging.getLogger(__name__)
TEMPLATE = Path(__file__).parent / "web" / "panel.html"


def write(path, result):
    started = time.perf_counter()
    text = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*PANEL*/null", json.dumps(result, separators=(",", ":")).replace("</", "<\\/")
    )
    Path(path).write_text(text, encoding="utf-8")
    LOG.info("page exit seconds=%.3f bytes=%d", time.perf_counter() - started, len(text))
