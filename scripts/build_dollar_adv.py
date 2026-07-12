#!/usr/bin/env python3
"""
Build the trailing dollar-ADV panel from FMP raw volume × adj_close.

Output: data/factors/dollar_adv_21d.parquet (wide panel).

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/build_dollar_adv.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.liquidity import DEFAULT_ADV_PATH, DEFAULT_RAW_DIR, build_dollar_adv_panel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_dollar_adv")


def main() -> None:
    panel = build_dollar_adv_panel(ROOT / DEFAULT_RAW_DIR)
    out = ROOT / DEFAULT_ADV_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    latest = panel.iloc[-1].dropna()
    logger.info(
        "Wrote %s shape=%s; latest cross-section median ADV=$%.1fM (n=%d)",
        out,
        panel.shape,
        float(latest.median()) / 1e6 if len(latest) else 0.0,
        len(latest),
    )


if __name__ == "__main__":
    main()
