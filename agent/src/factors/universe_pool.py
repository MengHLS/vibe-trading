"""Dynamic tradable-universe selection for cross-sectional strategies.

The pool is rebuilt independently on every observation date:

1. Exclude names whose total market cap is at or above ``max_market_cap``.
2. Exclude names whose price is at or above ``max_price``.
3. Rank the remaining names by trailing average activity.
4. Keep at most ``pool_size`` names.

All inputs are wide panels (``index=date``, ``columns=instrument``). Market
cap must be expressed in CNY. Activity is normally trading amount in CNY, but
ranking is unit-invariant as long as every column uses the same unit.

The threshold rules are intentionally strict (``<``): a 3000bn CNY cap or a
100 CNY price is excluded, matching the usual "3000亿以上" / "100元以上"
interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

ActivityMetric = Literal["amount", "volume", "turnover"]

_ACTIVITY_COLUMNS: dict[str, str] = {
    "amount": "amount",
    "volume": "volume",
    "turnover": "turnover",
}


@dataclass(frozen=True, slots=True)
class UniversePool:
    """Rules for a date-by-date liquid stock universe.

    Attributes:
        max_market_cap: Exclusive total-market-cap ceiling in CNY. ``None``
            disables the constraint. Defaults to 3000bn CNY (3000亿元).
        max_price: Exclusive price ceiling. ``None`` disables the constraint.
            Defaults to 100 CNY.
        activity_metric: Column used to rank activity. ``amount`` means
            trading value, ``volume`` means traded shares, and ``turnover``
            means turnover rate.
        activity_window: Trailing trading-day window for average activity.
        pool_size: Maximum number of names retained per date.
        price_column: Price panel to test against ``max_price``. When omitted,
            ``unadjusted_close`` is preferred and ``close`` is used as a
            fallback. Use an unadjusted price for an absolute price filter.
        market_cap_column: Total-market-cap panel in CNY.
    """

    max_market_cap: float | None = 300_000_000_000.0
    max_price: float | None = 100.0
    activity_metric: ActivityMetric = "amount"
    activity_window: int = 20
    pool_size: int = 500
    price_column: str | None = None
    market_cap_column: str = "total_market_cap"

    def __post_init__(self) -> None:
        if self.max_market_cap is not None and self.max_market_cap <= 0:
            raise ValueError("max_market_cap must be positive or None")
        if self.max_price is not None and self.max_price <= 0:
            raise ValueError("max_price must be positive or None")
        if self.activity_metric not in _ACTIVITY_COLUMNS:
            raise ValueError(
                f"unsupported activity_metric {self.activity_metric!r}; "
                f"expected one of {sorted(_ACTIVITY_COLUMNS)}"
            )
        if self.activity_window < 1:
            raise ValueError("activity_window must be at least 1")
        if self.pool_size < 1:
            raise ValueError("pool_size must be at least 1")

    def eligibility(self, panel: dict[str, Any]) -> pd.DataFrame:
        """Return a boolean eligibility mask with the shape of ``close``.

        A full ``activity_window`` of history is required, so recently listed
        names are excluded until enough observations exist. Names with missing
        cap, price, or activity are excluded rather than filled.

        Args:
            panel: Wide-panel mapping containing ``close``, the configured
                activity column, and market cap when that filter is enabled.

        Returns:
            Boolean DataFrame where True means the name may be scored on that
            date. The top ``pool_size`` names are selected after all filters.

        Raises:
            ValueError: If a required panel or configured column is missing.
        """
        ref = panel.get("close")
        if ref is None:
            raise ValueError("panel must contain a 'close' DataFrame")

        activity = self._activity_panel(panel)
        average_activity = activity.rolling(
            window=self.activity_window,
            min_periods=self.activity_window,
        ).mean()
        eligible = average_activity.notna() & (average_activity > 0)

        if self.max_market_cap is not None:
            market_cap = panel.get(self.market_cap_column)
            if market_cap is None:
                raise ValueError(
                    f"panel must contain {self.market_cap_column!r} when "
                    "max_market_cap is configured"
                )
            market_cap = market_cap.reindex(
                index=ref.index, columns=ref.columns
            )
            eligible &= market_cap.notna() & (market_cap < self.max_market_cap)

        if self.max_price is not None:
            price = self._price_panel(panel)
            eligible &= price.notna() & (price < self.max_price)

        # Ranking only among names that passed every eligibility filter.
        ranks = average_activity.where(eligible).rank(
            axis=1,
            method="first",
            ascending=False,
            na_option="keep",
        )
        return (eligible & ranks.le(self.pool_size)).fillna(False)

    def mask_panel(
        self, panel: dict[str, Any]
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Mask all aligned panel values outside the selected pool.

        Masking before alpha evaluation prevents excluded names from affecting
        cross-sectional operators such as rank, z-score, or neutralization.

        Args:
            panel: Wide-panel mapping used to calculate eligibility.

        Returns:
            ``(masked_panel, eligible)``. DataFrames aligned with ``close`` are
            masked; non-DataFrame metadata is copied through unchanged.
        """
        eligible = self.eligibility(panel)
        ref = panel["close"]
        masked: dict[str, Any] = {}
        for key, value in panel.items():
            if not isinstance(value, pd.DataFrame):
                masked[key] = value
                continue
            if value.index.equals(ref.index) and set(ref.columns).issubset(value.columns):
                aligned = value.reindex(index=ref.index, columns=ref.columns)
                masked[key] = aligned.where(eligible)
            else:
                masked[key] = value
        return masked, eligible

    def _price_panel(self, panel: dict[str, Any]) -> pd.DataFrame:
        ref = panel["close"]
        if self.price_column is not None:
            price = panel.get(self.price_column)
            if price is None:
                raise ValueError(
                    f"price_column {self.price_column!r} is not present in panel"
                )
        else:
            price = panel.get("unadjusted_close", ref)
        return price.reindex(index=ref.index, columns=ref.columns)

    def _activity_panel(self, panel: dict[str, Any]) -> pd.DataFrame:
        ref = panel["close"]
        column = _ACTIVITY_COLUMNS[self.activity_metric]
        activity = panel.get(column)
        if activity is None:
            raise ValueError(
                f"panel must contain {column!r} for activity_metric="
                f"{self.activity_metric!r}"
            )
        return activity.reindex(index=ref.index, columns=ref.columns)
