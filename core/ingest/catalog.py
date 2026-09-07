"""Loading and validating a vendor's declarative endpoint manifest.

A vendor's surface is data, not code: ``config/vendors/{vendor}.json`` lists
every endpoint with its partitioning, point-in-time status, pagination and wave.
Onboarding a new vendor is then a manifest plus a transport adapter, rather than
another fetch script per dataset — which is the whole reason this package exists.

The manifest is validated on load rather than trusted, because a typo in a
``pit_status`` is the kind of error that surfaces months later as a leaked
backtest rather than as an exception.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

from core.exceptions import ConfigError
from core.ingest.spec import EndpointSpec, Partition, Payload, Subject

logger = logging.getLogger(__name__)

VENDOR_CONFIG_DIR = Path("config/vendors")


def load_manifest(vendor: str, config_dir: Path = VENDOR_CONFIG_DIR) -> dict[str, Any]:
    """
    Read a vendor manifest from disk.

    Args:
        vendor: Manifest stem, e.g. ``"fmp"``.
        config_dir: Directory holding vendor manifests.

    Returns:
        Decoded manifest.

    Raises:
        ConfigError: If the manifest is missing or malformed.
    """
    path = config_dir / f"{vendor}.json"
    if not path.is_file():
        raise ConfigError(f"no manifest for vendor {vendor!r} at {path}")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    if "endpoints" not in manifest:
        raise ConfigError(f"{path} has no 'endpoints' list")
    return manifest


def build_specs(manifest: dict[str, Any]) -> list[EndpointSpec]:
    """
    Turn manifest entries into validated specs.

    Args:
        manifest: Decoded manifest from :func:`load_manifest`.

    Returns:
        Specs in manifest order.

    Raises:
        ConfigError: On an unknown partition, payload, or duplicate name.
    """
    specs: list[EndpointSpec] = []
    seen: set[str] = set()
    for entry in manifest["endpoints"]:
        name = entry.get("name")
        if name in seen:
            raise ConfigError(f"duplicate endpoint name {name!r} in manifest")
        seen.add(name)
        try:
            partition = Partition(entry["partition"])
            payload = Payload(entry.get("payload", "json"))
            subject = Subject(entry.get("subject", "request_key"))
        except (KeyError, ValueError) as exc:
            raise ConfigError(f"{name}: bad partition/payload: {exc}") from exc
        try:
            specs.append(
                EndpointSpec(
                    name=name,
                    endpoint=entry["endpoint"],
                    partition=partition,
                    pit_status=entry["pit_status"],
                    params=entry.get("params", {}),
                    date_columns=tuple(entry.get("date_columns", ())),
                    primary_date=entry.get("primary_date"),
                    payload=payload,
                    subject=subject,
                    batch_size=entry.get("batch_size", 100),
                    priority=entry.get("priority", 3),
                    derivable=entry.get("derivable", False),
                    cadence=entry.get("cadence", "daily"),
                    paginate=entry.get("paginate", False),
                    page_size=entry.get("page_size", 100),
                    max_pages=entry.get("max_pages", 200),
                    date_chunk_years=entry.get("date_chunk_years"),
                    history_start=entry.get("history_start", "1985-01-01"),
                    notes=entry.get("notes", ""),
                )
            )
        except (KeyError, ValueError) as exc:
            raise ConfigError(f"{name}: invalid spec: {exc!r}") from exc
    return specs


def select_specs(
    specs: Iterable[EndpointSpec],
    waves: Optional[set[int]] = None,
    names: Optional[set[str]] = None,
    include_derivable: bool = False,
) -> list[EndpointSpec]:
    """
    Filter specs for a run, sorted so lower waves execute first.

    Args:
        specs: All known specs.
        waves: Priority values to include; None means every wave.
        names: Explicit endpoint names; overrides ``waves`` when given.
        include_derivable: Include endpoints the repo could compute itself.

    Returns:
        Filtered, wave-ordered specs.
    """
    selected = []
    for spec in specs:
        if names is not None:
            if spec.name in names:
                selected.append(spec)
            continue
        if spec.derivable and not include_derivable:
            continue
        if waves is not None and spec.priority not in waves:
            continue
        selected.append(spec)
    return sorted(selected, key=lambda s: (s.priority, s.name))
