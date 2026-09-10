"""Guarded writes for derived artifacts — a subset build cannot clobber a panel.

This exists because of a specific, avoidable failure. ``build_event_factors.py``
accepts ``--symbols AAPL,MSFT,XOM`` for smoke tests. It then wrote its result to
the same path the full build uses, so a three-symbol smoke test replaced a
23-million-row production panel with a few thousand rows. Nothing errored: the
file was valid parquet, the correct shape, and simply wrong. It was found days
later by a routine artifact sweep.

The lesson is not "be careful with flags". It is that **a partial build should
not be able to name a production path at all**, and that a full rebuild which
suddenly produces a fraction of the previous rows is far more likely to be a bug
than a real change.

Two mechanisms, both cheap:

- :func:`resolve_artifact_path` redirects any subset build into a ``scratch/``
  sibling directory. The production path is unreachable without declaring the
  build complete.
- :func:`write_artifact` refuses to shrink an existing artifact past
  :data:`SHRINK_TOLERANCE` unless the caller passes ``allow_shrink=True``, and
  writes through a temp file so an interrupted run leaves the old artifact
  intact rather than a truncated one.

Deliberately not a general-purpose IO layer: it wraps ``to_parquet`` for the
handful of scripts that publish panels, and nothing in ``core`` computation calls
it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "SCRATCH_DIRNAME",
    "SHRINK_TOLERANCE",
    "ArtifactShrinkError",
    "resolve_artifact_path",
    "streamed_artifact",
    "write_artifact",
]

# Subset builds land here instead of alongside the production panels.
SCRATCH_DIRNAME = "scratch"

# A rebuild may legitimately lose rows (a vendor drops history, a filter
# tightens). Losing more than this fraction is a bug until proven otherwise.
SHRINK_TOLERANCE = 0.10


class ArtifactShrinkError(RuntimeError):
    """A write would have destroyed a materially larger existing artifact."""


def resolve_artifact_path(path: Path, *, subset: bool) -> Path:
    """
    Return the path a build may legally write to.

    Args:
        path: The production artifact path the script would use for a full build.
        subset: True when the run covers only some symbols/dates (``--symbols``,
            ``--limit``, a smoke test).

    Returns:
        ``path`` for a full build; a ``scratch/`` sibling for a subset build.
        The scratch directory is created on demand.
    """
    if not subset:
        return path
    scratch = path.parent / SCRATCH_DIRNAME
    scratch.mkdir(parents=True, exist_ok=True)
    redirected = scratch / path.name
    logger.warning(
        "Subset build: writing %s instead of %s. Partial results never overwrite "
        "a production panel.",
        redirected,
        path,
    )
    return redirected


def _check_shrink(new_rows: int, previous: int | None, name: str, allow_shrink: bool) -> None:
    """Raise if replacing ``previous`` rows with ``new_rows`` looks like a bug."""
    if allow_shrink or previous is None or previous <= 0:
        return
    retained = new_rows / previous
    if retained < 1.0 - SHRINK_TOLERANCE:
        raise ArtifactShrinkError(
            f"{name}: refusing to replace {previous:,} rows with {new_rows:,} "
            f"({retained:.1%} retained, tolerance {1 - SHRINK_TOLERANCE:.0%}). "
            "If this shrink is intended, pass allow_shrink=True; if this is a "
            "partial run, pass subset=True."
        )


def _log_write(name: str, new_rows: int, previous: int | None) -> None:
    if previous is None or previous <= 0:
        logger.info("Wrote %s: %s rows (new artifact)", name, f"{new_rows:,}")
    else:
        logger.info(
            "Wrote %s: %s rows (was %s, %+.1f%%)",
            name,
            f"{new_rows:,}",
            f"{previous:,}",
            100.0 * (new_rows / previous - 1.0),
        )


@contextmanager
def streamed_artifact(
    path: Path,
    *,
    subset: bool = False,
    allow_shrink: bool = False,
    label: str | None = None,
) -> Iterator[Path]:
    """
    Guard a batched/streamed parquet build that cannot buffer the whole frame.

    Panels too large to hold in memory are written batch by batch with a
    ``pq.ParquetWriter``, so :func:`write_artifact` does not apply. This yields a
    temp path to stream into, then applies the same subset redirect and shrink
    guard before publishing it over the production path.

    Args:
        path: Production artifact path.
        subset: True when this run covers only part of the universe.
        allow_shrink: Permit a large row-count drop.
        label: Name used in log lines.

    Yields:
        The temp path to write batches into.

    Raises:
        ArtifactShrinkError: If the finished temp artifact is materially smaller
            than the one it would replace. The production artifact is untouched.
    """
    target = resolve_artifact_path(path, subset=subset)
    name = label or target.name
    previous = _existing_rows(target)
    temp = target.with_suffix(target.suffix + ".tmp")

    try:
        yield temp
        written = _existing_rows(temp) or 0
        if written == 0:
            logger.warning("%s: build produced no rows; leaving existing artifact in place", name)
            return
        _check_shrink(written, previous, name, allow_shrink)
        os.replace(temp, target)
        _log_write(name, written, previous)
    finally:
        if temp.exists():
            temp.unlink()


def _existing_rows(path: Path) -> int | None:
    """Row count of an existing parquet artifact, or None if absent/unreadable."""
    if not path.exists():
        return None
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:  # noqa: BLE001 - a corrupt prior artifact must not block a rebuild
        logger.warning("Could not read row count from existing %s", path)
        return None


def write_artifact(
    frame: pd.DataFrame,
    path: Path,
    *,
    subset: bool = False,
    allow_shrink: bool = False,
    label: str | None = None,
) -> Path:
    """
    Write a derived panel, refusing the writes that have historically been bugs.

    Args:
        frame: Panel to persist.
        path: Production artifact path (redirected when ``subset`` is True).
        subset: True when this run covers only part of the universe.
        allow_shrink: Permit a large row-count drop. Pass this only when the
            shrink is the intended outcome, and say why in the caller.
        label: Name used in log lines; defaults to the file name.

    Returns:
        The path actually written.

    Raises:
        ArtifactShrinkError: If the new frame is materially smaller than the
            artifact it would replace and ``allow_shrink`` is False.
    """
    target = resolve_artifact_path(path, subset=subset)
    name = label or target.name

    previous = _existing_rows(target)
    _check_shrink(len(frame), previous, name, allow_shrink)

    # Temp-then-rename: an interrupted write leaves the previous artifact intact
    # rather than a half-written file that still parses as parquet.
    temp = target.with_suffix(target.suffix + ".tmp")
    frame.to_parquet(temp)
    os.replace(temp, target)

    _log_write(name, len(frame), previous)
    return target
