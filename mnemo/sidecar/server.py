"""Mnemo sidecar — stdio JSON-lines protocol (the KERN seam). **No torch here**;
all model work is behind narrate.Engine.

WIRE CONTRACT v1
================
Transport: newline-delimited JSON over stdio. UTF-8, LF only, one compact
(no-whitespace) JSON value per line. stdout carries PROTOCOL ONLY; every log /
diagnostic goes to stderr. On Windows, run with PYTHONUNBUFFERED=1 (or trust the
explicit per-line flush below) and note that this module writes raw UTF-8 bytes
to stdout.buffer so newline translation can never corrupt the framing.

Handshake (emitted once, before any request is read):
  {"hello":"mnemo-sidecar","protocol":"v1","weights_hash":"<12hex>",
   "n_videos":<int>,"methods":["list","summary","describe","peaks"]}

Request line:
  {"id":<any>,"method":"list|summary|describe|peaks","params":{...}}

Success response line:
  {"id":<echo>,"ok":true,"method":"<m>", ...result...}

Error response line:
  {"id":<echo|null>,"ok":false,"kind":"unknown-video|bad-params|internal",
   "message":"<text>"}
A malformed JSON line yields kind "bad-params" with id:null. The loop never
crashes on a bad request.

Methods
-------
list      params: {}                       -> {"videos":[{id,duration,n_scenes,
                                                          n_segments}],"n_videos"}
summary   params: {"video"}                -> {"video","source":"gauge",
                                              "lines"[<=16],"truncated"}
describe  params: {"video", t0?, t1?}      -> {"video","source":"model",
                                              "window","lines"[<=12],
                                              "truncated","hint"}
peaks     params: {"video","metric",k?}    -> {"video","metric","k"(<=10),
                                              "lines"}
metric in {loudness, motion, importance}.

Receipt (on EVERY narration line): loud_db, mot, t0, t1 (floats), act (string or
null when no action). describe lines are learned-reader text; summary/peaks lines
are deterministic gauge-template text.
"""
from __future__ import annotations

import json
import sys
import traceback

from mnemo.sidecar.narrate import Engine, UnknownVideo, BadParams

METHODS = ["list", "summary", "describe", "peaks"]


# ── byte-exact framing (LF only, UTF-8; no text-mode newline translation) ────
def _emit(obj: dict, out_bytes) -> None:
    data = (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    out_bytes.write(data.encode("utf-8"))
    out_bytes.flush()


def _err(rid, kind: str, message: str, out_bytes) -> None:
    _emit({"id": rid, "ok": False, "kind": kind, "message": message}, out_bytes)


def _need_str(params: dict, key: str) -> str:
    val = params.get(key)
    if not isinstance(val, str) or not val:
        raise BadParams(f"missing/invalid string param {key!r}")
    return val


def _opt_num(params: dict, key: str):
    if key not in params or params[key] is None:
        return None
    val = params[key]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise BadParams(f"param {key!r} must be a number")
    return float(val)


def _dispatch(engine: Engine, req: dict) -> tuple[str, dict]:
    """Return (method, result-without-id/ok). Raises BadParams/UnknownVideo."""
    if not isinstance(req, dict):
        raise BadParams("request must be a JSON object")
    method = req.get("method")
    params = req.get("params") or {}
    if not isinstance(params, dict):
        raise BadParams("params must be an object")

    if method == "list":
        return method, engine.list_videos()
    if method == "summary":
        return method, engine.summary(_need_str(params, "video"))
    if method == "describe":
        return method, engine.describe(
            _need_str(params, "video"),
            _opt_num(params, "t0"), _opt_num(params, "t1"))
    if method == "peaks":
        return method, engine.peaks(
            _need_str(params, "video"), _need_str(params, "metric"),
            params.get("k", 5))
    raise BadParams(f"unknown method: {method!r}")


def serve(corpus_dir: str, weights_path: str) -> int:
    """Run the stdio JSON-lines loop. Loads once, never trains."""
    sys.stdin.reconfigure(encoding="utf-8")  # deterministic decoding on Windows
    out_bytes = sys.stdout.buffer

    print(f"[sidecar] loading corpus={corpus_dir!r} weights={weights_path!r}",
          file=sys.stderr, flush=True)
    engine = Engine.load(corpus_dir, weights_path)
    print(f"[sidecar] ready: {len(engine.videos)} videos, "
          f"weights_hash={engine.weights_hash}", file=sys.stderr, flush=True)

    _emit({"hello": "mnemo-sidecar", "protocol": "v1",
           "weights_hash": engine.weights_hash,
           "n_videos": len(engine.videos), "methods": METHODS}, out_bytes)

    while True:
        raw = sys.stdin.readline()  # readline (not iteration) -> no read-ahead
        if raw == "":               # EOF
            break
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except (ValueError, TypeError):
            _err(None, "bad-params", "malformed JSON line", out_bytes)
            continue
        rid = req.get("id") if isinstance(req, dict) else None
        try:
            method, result = _dispatch(engine, req)
            _emit({"id": rid, "ok": True, "method": method, **result}, out_bytes)
        except BadParams as e:
            _err(rid, "bad-params", str(e), out_bytes)
        except UnknownVideo as e:
            _err(rid, "unknown-video", str(e), out_bytes)
        except Exception as e:  # never crash the loop
            traceback.print_exc(file=sys.stderr)
            _err(rid, "internal", f"{type(e).__name__}: {e}", out_bytes)
    return 0


# ── human driver (TASK 3): friendly REPL, same engine, no JSON required ───────
_HELP = """commands:
  list
  summary <video_id>
  describe <video_id> [t0 t1]
  peaks <video_id> <loudness|motion|importance> [k]
  help | quit"""


def _fmt_receipt(ln: dict) -> str:
    act = ln["act"] if ln["act"] is not None else "-"
    extra = f" {ln['value']:.3f}" if "value" in ln else ""
    return (f"[{ln['t0']:>7.1f}-{ln['t1']:<7.1f}s  loud={ln['loud_db']:>6.1f}dB  "
            f"mot={ln['mot']:.3f}  act={act}]{extra}")


def _print_lines(result: dict) -> None:
    lines = result.get("lines", [])
    src = result.get("source")
    if src:
        print(f"  (source: {src})")
    for ln in lines:
        print(f"  {ln['text']}")
        print(f"      {_fmt_receipt(ln)}")
    if result.get("truncated"):
        print(f"  ... {result.get('hint') or 'output truncated'}")
    if not lines:
        print("  (no lines)")


def repl(corpus_dir: str, weights_path: str) -> int:
    print(f"[sidecar repl] loading corpus={corpus_dir!r} weights={weights_path!r}",
          file=sys.stderr, flush=True)
    engine = Engine.load(corpus_dir, weights_path)
    print(f"mnemo sidecar repl — {len(engine.videos)} videos, "
          f"weights {engine.weights_hash}")
    print(_HELP)
    while True:
        try:
            line = input("mnemo> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        try:
            if cmd in ("quit", "exit", "q"):
                return 0
            if cmd in ("help", "?", "h"):
                print(_HELP)
            elif cmd == "list":
                res = engine.list_videos()
                for v in res["videos"]:
                    print(f"  {v['id']}  dur={v['duration']:.1f}s  "
                          f"scenes={v['n_scenes']}  segments={v['n_segments']}")
            elif cmd == "summary" and len(parts) >= 2:
                _print_lines(engine.summary(parts[1]))
            elif cmd == "describe" and len(parts) >= 2:
                t0 = float(parts[2]) if len(parts) >= 3 else None
                t1 = float(parts[3]) if len(parts) >= 4 else None
                _print_lines(engine.describe(parts[1], t0, t1))
            elif cmd == "peaks" and len(parts) >= 3:
                k = int(parts[3]) if len(parts) >= 4 else 5
                _print_lines(engine.peaks(parts[1], parts[2], k))
            else:
                print("  ? unrecognized — type 'help'")
        except (UnknownVideo, BadParams) as e:
            print(f"  error: {e}")
        except ValueError as e:
            print(f"  error: bad number ({e})")
