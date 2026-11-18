"""Atomic run status plus an OS-released lock, safe across interrupted workers."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..storage import write


def now():
    return datetime.now(timezone.utc).isoformat()


def status(directory, state, **details):
    write(
        Path(directory) / "status.json",
        {"state": state, "updated_at_utc": now(), **details},
    )


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / "run.lock").open("a+b")
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    import os

    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise ValueError("This run is already active")
    try:
        yield
    finally:
        handle.close()
