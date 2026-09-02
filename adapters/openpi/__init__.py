# SPDX-License-Identifier: MIT
"""STATUS: UNVERIFIED ROADMAP ADAPTER. Imports and unit tests pass without lerobot/openpi installed; end-to-end behaviour has NOT been validated against a live policy server. See docs/ARCHITECTURE.md sec 12.

`adapters.openpi` -- lictor as a websocket proxy in front of openpi's `WebsocketPolicyServer`
(see `lictor_proxy.py`). Nothing here imports `openpi`, `openpi_client`, `websockets` or `msgpack`
at import time.
"""
