"""Tiny threaded static server for the through-wall WiFi-CSI sensing demo.

Adapted from examples/three.js/server/serve-demo.py. Serves the
`examples/through-wall/` page so a browser can fetch index.html, then the
page connects directly to the LIVE sensing-server WebSocket at
ws://localhost:8765/ws/sensing (NOT proxied through here).

Why a threaded server (not `python -m http.server`)?
The stdlib SimpleHTTPServer is single-threaded; a browser opens several
parallel connections (HTML + the three.js CDN tags fetch in parallel),
the first eats the worker, the rest can stall. ThreadingHTTPServer fixes it.

IMPORTANT: this serves on port 8080 — port 8765 is taken by the
sensing-server's WebSocket. They are two different processes.

Usage:
    # 1) start the REAL sensing-server (separate terminal):
    #      cd v2
    #      cargo build -p wifi-densepose-sensing-server
    #      ./target/debug/sensing-server.exe --ws-port 8765 --udp-port 5005
    # 2) start this static server:
    python examples/through-wall/serve.py
    # 3) open:
    #      http://localhost:8080/examples/through-wall/index.html

Override the WS endpoint with a query param, e.g.:
    http://localhost:8080/examples/through-wall/index.html?ws=ws://192.168.1.20:8765/ws/sensing
"""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import os
import sys

PORT = int(os.environ.get("PORT", 8080))

# Serve from the repo root regardless of where this script is launched.
# This file lives at examples/through-wall/serve.py — two levels deep.
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Aggressive no-cache so the browser ALWAYS fetches the latest
        # index.html after edits, even on a soft refresh.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("[serve] " + (fmt % args) + "\n")


PAGE = "examples/through-wall/index.html"

with ThreadingHTTPServer(("127.0.0.1", PORT), NoCacheHandler) as srv:
    print(f"serving {os.getcwd()} on http://127.0.0.1:{PORT}/")
    print(f"  open  http://localhost:{PORT}/{PAGE}")
    print("")
    print("  The page connects to the LIVE sensing-server at")
    print("  ws://localhost:8765/ws/sensing (start it first — see README.md).")
    print("  Override with ?ws=ws://HOST:PORT/ws/sensing")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
