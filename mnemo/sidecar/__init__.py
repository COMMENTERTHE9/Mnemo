"""Mnemo sidecar — a stdio JSON-lines server exposing Mnemo's perception trees
and learned reader to KERN (the wire contract is documented in server.py).

  python -m mnemo.sidecar train   # train the production reader -> weights/
  python -m mnemo.sidecar         # serve (stdio JSON-lines, the KERN seam)
  python -m mnemo.sidecar repl    # human driver (talk to it from PowerShell)

torch is confined to narrate.py (model load + generation); server.py and the
repl driver speak only the wire protocol.
"""
