"""Console entry for the Mnemo sidecar.

  python -m mnemo.sidecar [serve] [--corpus DIR] [--weights PATH]
  python -m mnemo.sidecar train    [--corpus DIR] [--out PATH] [--epochs N] [--seed N]
  python -m mnemo.sidecar repl     [--corpus DIR] [--weights PATH]

Default subcommand is `serve`. For Windows stdio sanity, prefer
`set PYTHONUNBUFFERED=1` before spawning (the server also flushes every line).
"""
from __future__ import annotations

import argparse
import sys

from mnemo.sidecar.narrate import DEFAULT_CORPUS, DEFAULT_WEIGHTS

_SUBCOMMANDS = {"serve", "train", "repl"}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _SUBCOMMANDS:
        argv = ["serve"] + argv  # default subcommand
    cmd, opts = argv[0], argv[1:]

    p = argparse.ArgumentParser(prog="mnemo.sidecar", description=cmd)
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    if cmd == "train":
        p.add_argument("--out", default=DEFAULT_WEIGHTS)
        p.add_argument("--epochs", type=int, default=120)
        p.add_argument("--seed", type=int, default=0)
    else:
        p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    args = p.parse_args(opts)

    if cmd == "train":
        from mnemo.sidecar.narrate import train_and_save
        info = train_and_save(corpus_dir=args.corpus, out_path=args.out,
                              epochs=args.epochs, seed=args.seed)
        # plain stdout is fine here — `train` is not the protocol surface.
        print(f"trained reader: {info['n_examples']} examples from "
              f"{info['n_videos']} videos -> {info['path']}")
        print(f"weights_hash={info['weights_hash']}")
        return 0
    if cmd == "repl":
        from mnemo.sidecar.server import repl
        return repl(args.corpus, args.weights)
    from mnemo.sidecar.server import serve
    return serve(args.corpus, args.weights)


if __name__ == "__main__":
    raise SystemExit(main())
