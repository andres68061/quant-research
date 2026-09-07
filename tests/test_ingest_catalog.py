"""
Unit tests for core.ingest.catalog.

A vendor's surface is data, not code, which only works if the data is validated
on load: a mistyped ``pit_status`` in the manifest would otherwise surface months
later as a leaked backtest rather than as an exception at startup. The first test
here is a guard on the shipped manifest itself — ``config/vendors/fmp.json``, 175
endpoints — so a hand-edit that breaks it fails in CI rather than at hour six of
a backfill.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.exceptions import ConfigError
from core.ingest.catalog import build_specs, load_manifest, select_specs
from core.ingest.spec import PIT_STATUSES, EndpointSpec, Partition, Payload

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_CONFIG_DIR = REPO_ROOT / "config" / "vendors"
FMP_MANIFEST = VENDOR_CONFIG_DIR / "fmp.json"

needs_fmp_manifest = pytest.mark.skipif(
    not FMP_MANIFEST.is_file(), reason="config/vendors/fmp.json not present"
)


def _entry(**overrides: Any) -> dict[str, Any]:
    """One minimal valid manifest entry with the given fields overridden."""
    entry: dict[str, Any] = {
        "name": "ratios",
        "endpoint": "ratios",
        "partition": "per_symbol",
        "pit_status": "point_in_time",
    }
    entry.update(overrides)
    return entry


def _manifest(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"vendor": "test", "endpoints": list(entries)}


def _spec(**overrides: Any) -> EndpointSpec:
    fields: dict[str, Any] = {
        "name": "ratios",
        "endpoint": "ratios",
        "partition": Partition.PER_SYMBOL,
        "pit_status": "point_in_time",
    }
    fields.update(overrides)
    return EndpointSpec(**fields)


# --------------------------------------------------------------------------
# The shipped manifest
# --------------------------------------------------------------------------


@needs_fmp_manifest
def test_build_specs_accepts_the_shipped_fmp_manifest() -> None:
    manifest = load_manifest("fmp", VENDOR_CONFIG_DIR)

    specs = build_specs(manifest)

    assert len(specs) == len(manifest["endpoints"])
    assert specs
    assert all(isinstance(spec, EndpointSpec) for spec in specs)
    assert all(spec.pit_status in PIT_STATUSES for spec in specs)
    assert all(isinstance(spec.partition, Partition) for spec in specs)
    assert all(isinstance(spec.payload, Payload) for spec in specs)
    assert len({spec.name for spec in specs}) == len(specs)
    assert all("/" not in spec.name for spec in specs)


@needs_fmp_manifest
def test_shipped_fmp_manifest_declares_the_transport_essentials() -> None:
    manifest = load_manifest("fmp", VENDOR_CONFIG_DIR)

    assert manifest["vendor"] == "fmp"
    assert manifest["base_url"].startswith("https://")
    assert manifest["rate_limit_per_minute"] > 0


@needs_fmp_manifest
def test_select_specs_on_the_shipped_manifest_returns_wave_ordered_specs() -> None:
    specs = build_specs(load_manifest("fmp", VENDOR_CONFIG_DIR))

    selected = select_specs(specs)

    priorities = [spec.priority for spec in selected]
    assert priorities == sorted(priorities)
    assert all(not spec.derivable for spec in selected)
    assert len(selected) <= len(specs)


# --------------------------------------------------------------------------
# load_manifest
# --------------------------------------------------------------------------


def test_load_manifest_reads_a_manifest_from_disk(tmp_path: Path) -> None:
    (tmp_path / "acme.json").write_text(json.dumps(_manifest(_entry())))

    manifest = load_manifest("acme", tmp_path)

    assert manifest["vendor"] == "test"
    assert len(manifest["endpoints"]) == 1


def test_load_manifest_raises_config_error_for_a_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no manifest for vendor"):
        load_manifest("nope", tmp_path)


def test_load_manifest_raises_config_error_for_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "acme.json").write_text("{not json,}")

    with pytest.raises(ConfigError, match="not valid JSON"):
        load_manifest("acme", tmp_path)


def test_load_manifest_raises_config_error_when_endpoints_are_missing(tmp_path: Path) -> None:
    (tmp_path / "acme.json").write_text(json.dumps({"vendor": "acme"}))

    with pytest.raises(ConfigError, match="no 'endpoints' list"):
        load_manifest("acme", tmp_path)


# --------------------------------------------------------------------------
# build_specs
# --------------------------------------------------------------------------


def test_build_specs_maps_every_declared_field() -> None:
    manifest = _manifest(
        _entry(
            name="historical_eod",
            endpoint="historical-price-eod/full",
            partition="per_symbol",
            pit_status="point_in_time",
            params={"serietype": "line"},
            date_columns=["date"],
            primary_date="date",
            payload="json",
            batch_size=50,
            priority=1,
            derivable=False,
            cadence="weekly",
            paginate=True,
            page_size=1000,
            max_pages=2000,
            date_chunk_years=5,
            history_start="1990-01-01",
            notes="capped at 5,000 bars per call",
        )
    )

    spec = build_specs(manifest)[0]

    assert spec.name == "historical_eod"
    assert spec.endpoint == "historical-price-eod/full"
    assert spec.partition is Partition.PER_SYMBOL
    assert spec.params == {"serietype": "line"}
    assert spec.date_columns == ("date",)
    assert spec.primary_date == "date"
    assert spec.payload is Payload.JSON
    assert spec.batch_size == 50
    assert spec.priority == 1
    assert spec.cadence == "weekly"
    assert spec.paginate is True
    assert spec.page_size == 1000
    assert spec.max_pages == 2000
    assert spec.date_chunk_years == 5
    assert spec.history_start == "1990-01-01"
    assert "5,000 bars" in spec.notes


def test_build_specs_applies_defaults_for_omitted_fields() -> None:
    spec = build_specs(_manifest(_entry()))[0]

    assert spec.params == {}
    assert spec.date_columns == ()
    assert spec.primary_date is None
    assert spec.payload is Payload.JSON
    assert spec.batch_size == 100
    assert spec.priority == 3
    assert spec.derivable is False
    assert spec.cadence == "daily"
    assert spec.paginate is False
    assert spec.date_chunk_years is None
    assert spec.notes == ""


def test_build_specs_returns_no_specs_for_an_empty_manifest() -> None:
    assert build_specs(_manifest()) == []


def test_build_specs_raises_config_error_for_a_bad_pit_status() -> None:
    manifest = _manifest(_entry(pit_status="pointintime"))

    with pytest.raises(ConfigError, match="invalid spec"):
        build_specs(manifest)


def test_build_specs_raises_config_error_for_a_duplicate_name() -> None:
    manifest = _manifest(_entry(name="ratios"), _entry(name="ratios", endpoint="ratios-ttm"))

    with pytest.raises(ConfigError, match="duplicate endpoint name"):
        build_specs(manifest)


def test_build_specs_raises_config_error_for_an_unknown_partition() -> None:
    manifest = _manifest(_entry(partition="per_planet"))

    with pytest.raises(ConfigError, match="bad partition/payload"):
        build_specs(manifest)


def test_build_specs_raises_config_error_for_an_unknown_payload() -> None:
    manifest = _manifest(_entry(payload="protobuf"))

    with pytest.raises(ConfigError, match="bad partition/payload"):
        build_specs(manifest)


# Regression: a manifest entry missing 'endpoint' or 'pit_status' used to escape as
# a raw KeyError, giving the operator no indication of which entry was at fault.
@pytest.mark.parametrize("missing_field", ["endpoint", "pit_status"])
def test_build_specs_raises_config_error_for_a_missing_required_field(
    missing_field: str,
) -> None:
    entry = _entry()
    entry.pop(missing_field)

    with pytest.raises(ConfigError):
        build_specs(_manifest(entry))


def test_build_specs_raises_config_error_for_a_missing_partition() -> None:
    entry = _entry()
    entry.pop("partition")

    with pytest.raises(ConfigError, match="bad partition/payload"):
        build_specs(_manifest(entry))


def test_build_specs_raises_config_error_for_a_name_with_a_path_separator() -> None:
    manifest = _manifest(_entry(name="ratios/ttm"))

    with pytest.raises(ConfigError, match="invalid spec"):
        build_specs(manifest)


# --------------------------------------------------------------------------
# select_specs
# --------------------------------------------------------------------------


@pytest.fixture()
def catalog() -> list[EndpointSpec]:
    return [
        _spec(name="quote", priority=1),
        _spec(name="cik_list", priority=1, partition=Partition.GLOBAL),
        _spec(name="ratios", priority=2),
        _spec(name="rsi", priority=1, derivable=True),
        _spec(name="sma", priority=3, derivable=True),
    ]


def test_select_specs_filters_by_wave(catalog: list[EndpointSpec]) -> None:
    selected = select_specs(catalog, waves={1})

    assert [spec.name for spec in selected] == ["cik_list", "quote"]


def test_select_specs_accepts_several_waves(catalog: list[EndpointSpec]) -> None:
    selected = select_specs(catalog, waves={1, 2})

    assert [spec.name for spec in selected] == ["cik_list", "quote", "ratios"]


def test_select_specs_with_no_filters_returns_every_non_derivable_spec(
    catalog: list[EndpointSpec],
) -> None:
    selected = select_specs(catalog)

    assert [spec.name for spec in selected] == ["cik_list", "quote", "ratios"]


def test_select_specs_excludes_derivable_endpoints_by_default(
    catalog: list[EndpointSpec],
) -> None:
    assert "rsi" not in {spec.name for spec in select_specs(catalog)}
    assert "rsi" not in {spec.name for spec in select_specs(catalog, waves={1})}


def test_select_specs_includes_derivable_endpoints_when_asked(
    catalog: list[EndpointSpec],
) -> None:
    selected = select_specs(catalog, include_derivable=True)

    assert [spec.name for spec in selected] == ["cik_list", "quote", "rsi", "ratios", "sma"]


def test_select_specs_filters_by_explicit_names(catalog: list[EndpointSpec]) -> None:
    selected = select_specs(catalog, names={"ratios", "quote"})

    assert [spec.name for spec in selected] == ["quote", "ratios"]


def test_select_specs_names_override_wave_and_derivable_filters(
    catalog: list[EndpointSpec],
) -> None:
    selected = select_specs(catalog, waves={2}, names={"rsi"})

    assert [spec.name for spec in selected] == ["rsi"]


def test_select_specs_with_unknown_names_returns_nothing(catalog: list[EndpointSpec]) -> None:
    assert select_specs(catalog, names={"does_not_exist"}) == []


def test_select_specs_sorts_by_wave_then_name(catalog: list[EndpointSpec]) -> None:
    selected = select_specs(catalog, include_derivable=True)

    assert [(spec.priority, spec.name) for spec in selected] == sorted(
        (spec.priority, spec.name) for spec in catalog
    )


def test_select_specs_on_an_empty_catalog_returns_nothing() -> None:
    assert select_specs([]) == []
