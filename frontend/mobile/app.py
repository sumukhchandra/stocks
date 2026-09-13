"""
NSE Stocks Mobile — Lightweight Standalone Server for Render / Cloud Web Services.
Uses standard Python library (zero dependencies, instant build).
"""
import http.server
import os
import socketserver

PORT = int(os.environ.get("PORT", 10000))
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Enable CORS and caching headers for PWA & API interoperability
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), CustomHandler) as httpd:
        print(f"✅ NSE Stocks Mobile Terminal serving on http://0.0.0.0:{PORT} from {DIRECTORY}")
        httpd.serve_forever()
