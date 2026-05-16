"""
Data loading utilities for SPX/SPXW option chains via the Massive API.

Authenticates via the MASSIVE_TOKEN environment variable.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Optional, List

logger = logging.getLogger(__name__)


def _get_client():
    """Instantiate a Massive REST client using the MASSIVE_TOKEN env var."""
    from massive import RESTClient
    api_key = os.getenv("MASSIVE_TOKEN", "")
    if not api_key:
        raise EnvironmentError(
            "MASSIVE_TOKEN environment variable is not set. "
            "Please export MASSIVE_TOKEN=<your_api_key>."
        )
    return RESTClient(api_key=api_key)


def fetch_option_chain(
    ticker: str,
    trade_date: str,
    expiry_date: str,
    option_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch an option chain snapshot for a given ticker, trade date, and expiry.

    Parameters
    ----------
    ticker : str
        Option ticker root, e.g. 'SPX' or 'SPXW'.
    trade_date : str
        Trade date in 'YYYY-MM-DD' format.
    expiry_date : str
        Expiry date in 'YYYY-MM-DD' format.
    option_type : str or None
        'call', 'put', or None for both.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [date, expiry, strike, type, price, iv, spot].
    """
    client = _get_client()
    records = []

    try:
        contracts = client.list_options_chain(
            underlying_asset=ticker,
            expiration_date=expiry_date,
            as_of=trade_date,
        )
        for c in contracts:
            opt_type = getattr(c, "contract_type", "").lower()
            if option_type and opt_type != option_type.lower():
                continue
            price = getattr(c, "last_price", None) or getattr(c, "close", None)
            if price is None:
                # Try mid from bid/ask
                bid = getattr(c, "bid", None)
                ask = getattr(c, "ask", None)
                if bid is not None and ask is not None:
                    price = (bid + ask) / 2.0
            records.append({
                "date": trade_date,
                "expiry": expiry_date,
                "strike": float(getattr(c, "strike_price", 0)),
                "type": opt_type,
                "price": float(price) if price is not None else np.nan,
                "iv": float(getattr(c, "implied_volatility", np.nan) or np.nan),
                "spot": float(getattr(c, "underlying_price", np.nan) or np.nan),
            })
    except Exception as e:
        logger.warning(f"Massive API call failed for {ticker} {trade_date}: {e}")

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["date", "expiry", "strike", "type", "price", "iv", "spot"])

    df = df.dropna(subset=["strike", "price"])
    df = df[df["price"] > 0]
    df = df[df["strike"] > 0]
    df = df.sort_values(["type", "strike"]).reset_index(drop=True)
    return df


def fetch_spx_spot(trade_date: str) -> Optional[float]:
    """
    Fetch the SPX spot price for a given trade date.

    Parameters
    ----------
    trade_date : str
        Trade date in 'YYYY-MM-DD' format.

    Returns
    -------
    float or None
        SPX spot price, or None if unavailable.
    """
    client = _get_client()
    try:
        aggs = []
        for a in client.list_aggs(
            ticker="SPX",
            multiplier=1,
            timespan="day",
            from_=trade_date,
            to=trade_date,
            limit=10,
        ):
            aggs.append(a)
        if aggs:
            return float(aggs[-1].close)
    except Exception as e:
        logger.warning(f"Failed to fetch SPX spot for {trade_date}: {e}")
    return None


def fetch_daily_option_chains(
    ticker: str,
    start_date: str,
    end_date: str,
    target_expiry_fn,
    trading_days: Optional[List[str]] = None,
) -> dict:
    """
    Fetch daily option chains for a date range, selecting the target expiry
    for each date using a provided function.

    Parameters
    ----------
    ticker : str
        Option ticker root.
    start_date : str
        Start date 'YYYY-MM-DD'.
    end_date : str
        End date 'YYYY-MM-DD'.
    target_expiry_fn : callable
        Function(trade_date: str) -> expiry_date: str.
    trading_days : list of str or None
        List of trading days. If None, uses all calendar days.

    Returns
    -------
    dict
        Mapping from trade_date (str) to pd.DataFrame option chain.
    """
    if trading_days is None:
        dates = pd.date_range(start_date, end_date, freq="B")
        trading_days = [d.strftime("%Y-%m-%d") for d in dates]

    chains = {}
    for td in trading_days:
        expiry = target_expiry_fn(td)
        if expiry is None:
            continue
        df = fetch_option_chain(ticker, td, expiry)
        if not df.empty:
            chains[td] = df
            logger.info(f"Loaded {len(df)} options for {td}, expiry {expiry}")
        else:
            logger.warning(f"No data for {td}, expiry {expiry}")
    return chains
