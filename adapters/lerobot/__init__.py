# SPDX-License-Identifier: MIT
"""STATUS: UNVERIFIED ROADMAP ADAPTER. Imports and unit tests pass without lerobot/openpi installed; end-to-end behaviour has NOT been validated against a live policy server. See docs/ARCHITECTURE.md sec 12.

`adapters.lerobot` -- lictor as a LeRobot v0.6 ProcessorStep (see `lictor_lerobot.py`).

This package is named after the framework it adapts to; it never shadows the upstream `lerobot`
distribution because every import in this repository is absolute (`import lerobot.processor`
resolves the installed package, `adapters.lerobot` resolves this one).
"""
