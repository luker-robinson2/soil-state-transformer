import http.server
import threading

from geosoil.corpus import common

DATA = bytes(range(256)) * 4000


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        rng = self.headers.get("Range")
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        body = DATA[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_fetch_resumes_a_partial_download_and_writes_sidecar(tmp_path):
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    dest = tmp_path / "f.bin"
    (tmp_path / "f.bin.part").write_bytes(DATA[:300_000])          # left over from an interrupted run
    common.fetch(f"http://127.0.0.1:{srv.server_port}/f.bin", dest)
    srv.shutdown()
    assert dest.read_bytes() == DATA
    assert (tmp_path / "f.bin.json").exists()
