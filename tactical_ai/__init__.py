"""Tactical AI package.

Keep top-level imports lightweight so utility modules can be inspected even
when optional RL dependencies have not yet been installed.
"""

__all__ = ["OpponentStyleFilter"]

from .bayes import OpponentStyleFilter
