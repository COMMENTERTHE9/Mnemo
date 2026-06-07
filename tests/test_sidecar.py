"""Golden-line tests for the Mnemo sidecar (wire contract v1).

Spawns the server as a real subprocess in BINARY mode so we can assert the
exact framing (LF only, UTF-8, one JSON value per line) the contract promises,
and that stdout carries protocol lines ONLY.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    """Train a tiny production reader once (few epochs — shape, not quality)."""
    from mnemo.sidecar.narrate import train_and_save
    out = tmp_path_factory.mktemp("weights") / "reader.pt"
    info = train_and_save(corpus_dir=str(CORPUS), out_path=str(out),
                          epochs=2, seed=0)
    assert len(info["weights_hash"]) == 12
    return str(out)


class Server:
    """Subprocess wrapper: binary stdio, one JSON request -> one JSON line."""

    def __init__(self, weights):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "mnemo.sidecar", "serve",
             "--weights", weights, "--corpus", str(CORPUS)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=str(ROOT), env=env)
        self.raw_lines = []
        self.hello_raw = self._readline()
        self.hello = json.loads(self.hello_raw)

    def _readline(self) -> bytes:
        line = self.proc.stdout.readline()
        if line == b"":
            err = self.proc.stderr.read().decode("utf-8", "replace")
            raise AssertionError(f"server produced no output. stderr:\n{err}")
        self.raw_lines.append(line)
        return line

    def rpc(self, obj):
        self.proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
        self.proc.stdin.flush()
        raw = self._readline()
        return raw, json.loads(raw)

    def send_raw(self, text: str):
        self.proc.stdin.write(text.encode("utf-8"))
        self.proc.stdin.flush()
        raw = self._readline()
        return raw, json.loads(raw)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


@pytest.fixture
def server(weights):
    s = Server(weights)
    yield s
    s.close()


RECEIPT_KEYS = {"loud_db", "mot", "act", "t0", "t1"}


def _assert_receipts(lines):
    for ln in lines:
        assert "text" in ln and isinstance(ln["text"], str) and ln["text"]
        assert RECEIPT_KEYS <= set(ln), f"missing receipt keys in {ln}"
        for k in ("loud_db", "mot", "t0", "t1"):
            assert isinstance(ln[k], (int, float))
        assert ln["act"] is None or isinstance(ln["act"], str)


# (a) hello line first and valid -------------------------------------------------
def test_hello_line_first_and_valid(server):
    h = server.hello
    assert h["hello"] == "mnemo-sidecar"
    assert h["protocol"] == "v1"
    assert h["methods"] == ["list", "summary", "describe", "peaks"]
    assert isinstance(h["weights_hash"], str) and len(h["weights_hash"]) == 12
    assert h["n_videos"] >= 1


# (b) each method returns contract-shaped JSON ----------------------------------
def test_list_shape(server):
    _, r = server.rpc({"id": 1, "method": "list"})
    assert r["id"] == 1 and r["ok"] is True and r["method"] == "list"
    assert r["videos"] and isinstance(r["videos"], list)
    v = r["videos"][0]
    assert {"id", "duration", "n_scenes", "n_segments"} <= set(v)


def test_summary_shape(server):
    vid = server.rpc({"id": 0, "method": "list"})[1]["videos"][0]["id"]
    _, r = server.rpc({"id": 2, "method": "summary", "params": {"video": vid}})
    assert r["ok"] is True and r["source"] == "gauge"
    assert len(r["lines"]) <= 16
    _assert_receipts(r["lines"])


def test_describe_shape(server):
    vid = server.rpc({"id": 0, "method": "list"})[1]["videos"][0]["id"]
    _, r = server.rpc({"id": 3, "method": "describe", "params": {"video": vid}})
    assert r["ok"] is True and r["source"] == "model"
    assert len(r["lines"]) <= 12
    assert "truncated" in r and "hint" in r
    _assert_receipts(r["lines"])


def test_describe_window(server):
    vid = server.rpc({"id": 0, "method": "list"})[1]["videos"][0]["id"]
    _, r = server.rpc({"id": 31, "method": "describe",
                       "params": {"video": vid, "t0": 0, "t1": 20}})
    assert r["ok"] is True and r["window"] == [0.0, 20.0]
    for ln in r["lines"]:
        assert ln["t1"] > 0 and ln["t0"] < 20


def test_peaks_shape(server):
    vid = server.rpc({"id": 0, "method": "list"})[1]["videos"][0]["id"]
    _, r = server.rpc({"id": 4, "method": "peaks",
                       "params": {"video": vid, "metric": "loudness", "k": 3}})
    assert r["ok"] is True and r["metric"] == "loudness"
    assert r["k"] <= 10 and len(r["lines"]) <= r["k"]
    _assert_receipts(r["lines"])
    # peaks are sorted descending by the metric value
    vals = [ln["value"] for ln in r["lines"]]
    assert vals == sorted(vals, reverse=True)


def test_peaks_k_capped_at_10(server):
    vid = server.rpc({"id": 0, "method": "list"})[1]["videos"][0]["id"]
    _, r = server.rpc({"id": 41, "method": "peaks",
                       "params": {"video": vid, "metric": "motion", "k": 999}})
    assert r["k"] == 10


# (c) error kinds ---------------------------------------------------------------
def test_unknown_video_kind(server):
    _, r = server.rpc({"id": 5, "method": "summary",
                       "params": {"video": "nope_does_not_exist"}})
    assert r["id"] == 5 and r["ok"] is False and r["kind"] == "unknown-video"
    assert "message" in r


def test_bad_params_kinds(server):
    # unknown method
    _, r = server.rpc({"id": 6, "method": "frobnicate"})
    assert r["ok"] is False and r["kind"] == "bad-params"
    # missing required param
    _, r = server.rpc({"id": 7, "method": "summary", "params": {}})
    assert r["ok"] is False and r["kind"] == "bad-params"
    # bad metric
    vid = server.rpc({"id": 0, "method": "list"})[1]["videos"][0]["id"]
    _, r = server.rpc({"id": 8, "method": "peaks",
                       "params": {"video": vid, "metric": "color"}})
    assert r["ok"] is False and r["kind"] == "bad-params"


def test_malformed_json_is_bad_params_id_null(server):
    raw, r = server.send_raw("this is not json\n")
    assert r["id"] is None and r["ok"] is False and r["kind"] == "bad-params"


# (d) caps + truncated flag fire on a long video --------------------------------
def test_caps_and_truncated_on_long_video(server):
    vids = server.rpc({"id": 0, "method": "list"})[1]["videos"]
    long = max(vids, key=lambda v: v["n_segments"])
    assert long["n_segments"] > 12, "need a video with >12 segments for this test"
    _, r = server.rpc({"id": 9, "method": "describe",
                       "params": {"video": long["id"]}})
    assert len(r["lines"]) == 12
    assert r["truncated"] is True
    assert isinstance(r["hint"], str) and "segments" in r["hint"]


# (e) stdout contains ONLY protocol lines + (f) every line parses and ends in LF -
def test_stdout_is_protocol_only_and_lf_framed(server):
    server.rpc({"id": 10, "method": "list"})
    server.rpc({"id": 11, "method": "summary",
                "params": {"video": server.hello and
                           server.rpc({"id": 0, "method": "list"})[1]["videos"][0]["id"]}})
    for raw in server.raw_lines:
        assert raw.endswith(b"\n")          # LF-terminated
        assert not raw.endswith(b"\r\n")    # no CRLF translation
        json.loads(raw)                     # every stdout line is valid JSON
