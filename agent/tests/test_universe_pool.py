from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest

from src.factors.universe_pool import UniversePool


_ZOO_ENGINE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "skills"
    / "multi-factor"
    / "zoo_signal_engine.py"
)


def _load_zoo_signal_engine():
    spec = spec_from_file_location("zoo_signal_engine_under_test", _ZOO_ENGINE_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ZooSignalEngine


def _panel() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2026-01-01", periods=3)
    close = pd.DataFrame(
        {
            "CAP": [10, 10, 10],
            "PRICE": [99, 99, 100],
            "LOW": [5, 5, 5],
            "OK_A": [20, 20, 20],
            "OK_B": [30, 30, 30],
        },
        index=dates,
    )
    amount = pd.DataFrame(
        {
            "CAP": [900, 900, 900],
            "PRICE": [800, 800, 800],
            "LOW": [1, 1, 1],
            "OK_A": [300, 300, 300],
            "OK_B": [200, 200, 200],
        },
        index=dates,
    )
    market_cap = pd.DataFrame(
        {
            "CAP": [300_000_000_000, 300_000_000_000, 300_000_000_000],
            "PRICE": [100_000_000_000] * 3,
            "LOW": [10_000_000_000] * 3,
            "OK_A": [20_000_000_000] * 3,
            "OK_B": [30_000_000_000] * 3,
        },
        index=dates,
    )
    return {"close": close, "amount": amount, "total_market_cap": market_cap}


def test_universe_pool_applies_exclusive_limits_before_activity_ranking() -> None:
    pool = UniversePool(
        max_market_cap=300_000_000_000,
        max_price=100,
        activity_window=2,
        pool_size=2,
    )

    eligible = pool.eligibility(_panel())

    assert eligible.iloc[0].sum() == 0  # full two-day window not available yet
    assert eligible.iloc[-1].to_dict() == {
        "CAP": False,
        "PRICE": False,
        "LOW": False,
        "OK_A": True,
        "OK_B": True,
    }


def test_universe_pool_mask_removes_excluded_names_from_every_aligned_panel() -> None:
    pool = UniversePool(activity_window=2, pool_size=2)

    masked, eligible = pool.mask_panel(_panel())

    assert pd.isna(masked["close"].loc[masked["close"].index[-1], "CAP"])
    assert masked["amount"].loc[masked["amount"].index[-1], "OK_A"] == 300
    assert eligible.loc[eligible.index[-1], ["OK_A", "OK_B"]].all()
    assert not eligible.loc[eligible.index[-1], ["CAP", "PRICE", "LOW"]].any()


def test_universe_pool_requires_market_cap_panel_when_cap_filter_enabled() -> None:
    panel = _panel()
    del panel["total_market_cap"]

    with pytest.raises(ValueError, match="total_market_cap"):
        UniversePool(activity_window=2).eligibility(panel)


def test_universe_pool_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="pool_size"):
        UniversePool(pool_size=0)
    with pytest.raises(ValueError, match="activity_window"):
        UniversePool(activity_window=0)
    with pytest.raises(ValueError, match="activity_metric"):
        UniversePool(activity_metric="not-a-metric")  # type: ignore[arg-type]


def test_zoo_signal_engine_rejects_invalid_universe_pool() -> None:
    ZooSignalEngine = _load_zoo_signal_engine()

    with pytest.raises(ValueError, match="mask_panel"):
        ZooSignalEngine.from_zoo(["test_factor"], universe_pool=object())


def test_zoo_signal_engine_only_ranks_names_inside_universe_pool() -> None:
    ZooSignalEngine = _load_zoo_signal_engine()
    panel = _panel()
    # LOW has the highest factor and would win without the universe filter, but
    # it ranks last on activity and pool_size=2 removes it.
    factor = panel["close"].copy()
    factor.loc[:, :] = 0.0
    factor["LOW"] = 10.0
    factor["OK_A"] = 2.0
    factor["OK_B"] = 1.0
    panel["factor"] = factor

    class _Registry:
        def compute(self, alpha_id: str, supplied_panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
            assert alpha_id == "test_factor"
            return supplied_panel["factor"]

    engine = ZooSignalEngine.from_zoo(
        ["test_factor"],
        top_n=1,
        universe_pool=UniversePool(activity_window=2, pool_size=2),
        registry=_Registry(),
    )

    positions = engine.compute_signal(panel)

    assert positions.iloc[-1].to_dict() == {
        "CAP": 0.0,
        "PRICE": 0.0,
        "LOW": 0.0,
        "OK_A": 1.0,
        "OK_B": 0.0,
    }


def test_universe_pool_price_fallback_uses_close_when_unadjusted_missing() -> None:
    pool = UniversePool(
        max_market_cap=None,
        max_price=25,
        activity_window=2,
        pool_size=10,
    )

    eligible = pool.eligibility(_panel())

    assert not eligible.iloc[-1]["OK_B"]
    assert eligible.iloc[-1]["OK_A"]


def test_universe_pool_can_disable_cap_and_price_filters() -> None:
    pool = UniversePool(
        max_market_cap=None,
        max_price=None,
        activity_window=2,
        pool_size=1,
    )

    eligible = pool.eligibility(_panel())

    assert eligible.iloc[-1].sum() == 1
    assert eligible.iloc[-1]["CAP"]  # highest activity survives when unbounded
