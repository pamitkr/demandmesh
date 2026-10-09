"""Process-wide logging configuration.

Logs are single-line ``key=value`` friendly records written to stdout, which is
what container platforms (Docker, Kubernetes, Cloud Run) collect by default.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger once; later calls only adjust the level."""
    global _CONFIGURED
    resolved = (level or os.getenv("DM_LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S"))
        root.handlers[:] = [handler]
        _CONFIGURED = True
    root.setLevel(getattr(logging, resolved, logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
