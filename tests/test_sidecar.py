"""Golden-line tests for the Mnemo sidecar — written FROM Wire Contract v1
(server.py docstring), not from the implementation. KERN parses against the
contract, so these assert the contract's exact shapes:

  success  : top-level keys == {id, ok, result}, result nested
  error    : top-level keys == {id, ok, kind, message}
  lines    : a JSON array; per-line "source"; video_id naming

Spawns the server as a real subprocess in BINARY mode to also verify the framing
(LF only, UTF-8, one JSON value per line) and that stdout is protocol-ONLY.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"

SUCCESS_KEYS = {"id", "ok", "result"}
ERROR_KEYS = {"id", "ok", "kind", "message"}
LINE_KEYS = {"t0", "t1", "text", "source", "loud_db", "mot", "act"}
PEAK_KEYS = {"t0", "t1", "value", "text"}


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
        return json.loads(self._readline())

    def result(self, obj):
        """Send a request expected to succeed; assert envelope; return result."""
        r = self.rpc(obj)
        assert set(r) == SUCCESS_KEYS, f"top-level keys {set(r)} != {SUCCESS_KEYS}"
        assert r["ok"] is True
        return r["result"]

    def send_raw(self, text: str):
        self.proc.stdin.write(text.encode("utf-8"))
        self.proc.stdin.flush()
        return json.loads(self._readline())

    def first_video(self):
        return self.result({"id": 0, "method": "list"})["videos"][0]["id"]

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


def _assert_lines(lines, expected_source):
    assert isinstance(lines, list)                      # lines is a JSON array
    for ln in lines:
        assert set(ln) == LINE_KEYS, f"line keys {set(ln)} != {LINE_KEYS}"
        assert ln["source"] == expected_source          # per-line source
        assert isinstance(ln["text"], str) and ln["text"]
        for k in ("t0", "t1", "loud_db", "mot"):
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
    res = server.result({"id": 1, "method": "list"})
    assert set(res) == {"videos"}                       # NOTHING else in result
    assert res["videos"] and isinstance(res["videos"], list)
    for v in res["videos"]:
        assert set(v) == {"id", "duration_s", "scenes", "segments"}


def test_summary_shape(server):
    vid = server.first_video()
    res = server.result({"id": 2, "method": "summary", "params": {"video": vid}})
    assert set(res) == {"video_id", "duration_s", "lines", "truncated"}
    assert res["video_id"] == vid                       # video_id naming
    assert len(res["lines"]) <= 16
    _assert_lines(res["lines"], "gauge")                # summary -> gauge


def test_describe_shape(server):
    vid = server.first_video()
    res = server.result({"id": 3, "method": "describe", "params": {"video": vid}})
    # short video (<=12 segments): no "hint" key when not truncated
    assert set(res) == {"video_id", "lines", "truncated"}
    assert res["video_id"] == vid
    assert res["truncated"] is False
    assert len(res["lines"]) <= 12
    _assert_lines(res["lines"], "model")                # describe -> model


def test_describe_window(server):
    vid = server.first_video()
    res = server.result({"id": 31, "method": "describe",
                         "params": {"video": vid, "t0": 0, "t1": 20}})
    assert "window" not in res                           # extra field dropped
    for ln in res["lines"]:
        assert ln["t1"] > 0 and ln["t0"] < 20


def test_peaks_shape(server):
    vid = server.first_video()
    res = server.result({"id": 4, "method": "peaks",
                         "params": {"video": vid, "slot": "loudness", "k": 3}})
    assert set(res) == {"video_id", "slot", "peaks"}
    assert res["slot"] == "loudness"
    assert isinstance(res["peaks"], list) and len(res["peaks"]) <= 10
    for p in res["peaks"]:
        assert set(p) == PEAK_KEYS                       # smaller {t0,t1,value,text}
    vals = [p["value"] for p in res["peaks"]]
    assert vals == sorted(vals, reverse=True)            # sorted desc by value


def test_peaks_k_capped_at_10(server):
    vids = server.result({"id": 0, "method": "list"})["videos"]
    long = max(vids, key=lambda v: v["segments"])
    res = server.result({"id": 41, "method": "peaks",
                         "params": {"video": long["id"], "slot": "motion",
                                    "k": 999}})
    assert len(res["peaks"]) == 10                       # k<=10 enforced


def test_peaks_accepts_legacy_metric_key(server):
    vid = server.first_video()
    res = server.result({"id": 42, "method": "peaks",
                         "params": {"video": vid, "metric": "importance"}})
    assert res["slot"] == "importance"


# (c) error kinds (top-level shape exact) ---------------------------------------
def _assert_error(r, kind):
    assert set(r) == ERROR_KEYS, f"error keys {set(r)} != {ERROR_KEYS}"
    assert r["ok"] is False and r["kind"] == kind


def test_unknown_video_kind(server):
    r = server.rpc({"id": 5, "method": "summary",
                    "params": {"video": "nope_does_not_exist"}})
    _assert_error(r, "unknown-video")
    assert r["id"] == 5


def test_bad_params_kinds(server):
    _assert_error(server.rpc({"id": 6, "method": "frobnicate"}), "bad-params")
    _assert_error(server.rpc({"id": 7, "method": "summary", "params": {}}),
                  "bad-params")
    vid = server.first_video()
    _assert_error(server.rpc({"id": 8, "method": "peaks",
                              "params": {"video": vid, "slot": "color"}}),
                  "bad-params")


def test_malformed_json_is_bad_params_id_null(server):
    r = server.send_raw("this is not json\n")
    _assert_error(r, "bad-params")
    assert r["id"] is None


# (d) caps + truncated flag (+ hint ONLY when truncated) -------------------------
def test_caps_and_truncated_on_long_video(server):
    vids = server.result({"id": 0, "method": "list"})["videos"]
    long = max(vids, key=lambda v: v["segments"])
    assert long["segments"] > 12, "need a video with >12 segments for this test"
    res = server.result({"id": 9, "method": "describe",
                         "params": {"video": long["id"]}})
    assert len(res["lines"]) == 12
    assert res["truncated"] is True
    assert set(res) == {"video_id", "lines", "truncated", "hint"}
    assert isinstance(res["hint"], str) and "segments" in res["hint"]


# (e) stdout is protocol-only + (f) every line parses and ends in LF ------------
def test_stdout_is_protocol_only_and_lf_framed(server):
    server.result({"id": 10, "method": "list"})
    server.result({"id": 11, "method": "summary",
                   "params": {"video": server.first_video()}})
    for raw in server.raw_lines:
        assert raw.endswith(b"\n")          # LF-terminated
        assert not raw.endswith(b"\r\n")    # no CRLF translation
        json.loads(raw)                     # every stdout line is valid JSON
