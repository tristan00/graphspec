import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .log import emit
from .timing import timed


class HttpError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


@timed("serve.server")
def server(port, page, routes, assets=None):
    page = Path(page)
    static = {"/": (page, "text/html; charset=utf-8"), "/index.html": (page, "text/html; charset=utf-8")}
    if assets is not None:
        static.update(assets)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parts = urlsplit(self.path)
            path, query = parts.path, {key: values[-1] for key, values in parse_qs(parts.query).items()}
            try:
                if path in static:
                    resource, content_type = static[path]
                    body = Path(resource).read_bytes()
                elif path in routes:
                    body, content_type = (
                        json.dumps(routes[path](query), allow_nan=False).encode("utf-8"),
                        "application/json; charset=utf-8",
                    )
                elif path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                    return
                else:
                    self.send_error(404)
                    return
            except HttpError as error:
                self.send_json_error(error.status, str(error))
                return
            except Exception as error:
                emit("serve_error", path=path, error=repr(error))
                self.send_json_error(500, "Route failed: " + type(error).__name__)
                return
            self.send_body(200, body, content_type)

        def send_json_error(self, status, message):
            self.send_body(
                status, json.dumps(dict(error=message)).encode("utf-8"), "application/json; charset=utf-8"
            )

        def send_body(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


@timed("serve.serve")
def serve(port, page, routes, assets=None, **fields):
    with server(port, page, routes, assets) as instance:
        emit("serve_enter", url="http://127.0.0.1:%d" % instance.server_address[1], **fields)
        instance.serve_forever()
