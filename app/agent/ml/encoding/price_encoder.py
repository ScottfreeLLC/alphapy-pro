"""
High-level PriceEncoder class for encoding OHLCV data into text token sequences.

Supports daily and intraday timeframes with fractal markers (boy/eoy, bom/eom, bod/eod).
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .encoder import encode_price, encode_price_df


# Fractal markers
YEARLY_MARKERS = ("boy", "eoy")
MONTHLY_MARKERS = ("bom", "eom")
DAILY_MARKERS = ("bod", "eod")

# Token pattern: letter(s) + digit(s), e.g. H3, P1, R0, V2
_TOKEN_RE = re.compile(r"^([A-Z]+)(\d+)$")

# Token type mapping
_TOKEN_TYPES = {
    "H": "pivot",
    "L": "pivot",
    "T": "pivot",
    "P": "net",
    "N": "net",
    "Z": "net",
    "R": "range",
    "V": "volume",
}

_DIRECTION_MAP = {
    "H": 1,    # pivot high
    "L": -1,   # pivot low
    "T": 0,    # tied
    "P": 1,    # positive net
    "N": -1,   # negative net
    "Z": 0,    # zero net
}


class PriceEncoder:
    """Encode OHLCV bars into text token sequences for NLP models."""

    def __init__(self, period: int = 20):
        self.period = period

    def encode_bars(self, df: pd.DataFrame, timeframe: str = "1d") -> str:
        """Encode a full DataFrame of OHLCV bars into a token string.

        Parameters
        ----------
        df : DataFrame with open, high, low, close, volume columns.
        timeframe : "1d" for daily, "5min" for intraday.
        """
        return encode_price(df, p=self.period)

    def encode_bars_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode bars and return DataFrame with encoding columns + numeric features."""
        return encode_price_df(df, p=self.period)

    def encode_rolling_window(
        self, df: pd.DataFrame, window: int = 20
    ) -> List[str]:
        """Encode sliding windows of N bars, returning a list of token strings.

        Each window is independently encoded (pivots restart per window).
        """
        results = []
        for start in range(len(df) - window + 1):
            chunk = df.iloc[start : start + window].reset_index(drop=True)
            results.append(encode_price(chunk, p=self.period))
        return results

    def encode_session(self, df: pd.DataFrame) -> str:
        """Encode a single intraday session, wrapped with bod/eod markers."""
        return encode_price(df, p=self.period, intraday=DAILY_MARKERS)

    def encode_grouped(
        self,
        df: pd.DataFrame,
        timeframe: str = "1d",
        group_col: Optional[str] = None,
    ) -> str:
        """Encode bars grouped by time period with fractal markers.

        For daily bars: groups by year (boy/eoy), then month (bom/eom).
        For intraday: groups by date (bod/eod).
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index("date")
            elif "timestamp" in df.columns:
                df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index)

        if timeframe == "1d":
            return self._encode_daily_grouped(df)
        else:
            return self._encode_intraday_grouped(df)

    def _encode_daily_grouped(self, df: pd.DataFrame) -> str:
        """Group daily bars by year > month with fractal markers."""
        parts = []
        for year, year_df in df.groupby(df.index.year):
            year_parts = ["boy"]
            for month, month_df in year_df.groupby(year_df.index.month):
                month_data = month_df.reset_index(drop=True)
                encoded = encode_price(month_data, p=self.period)
                year_parts.append(f"bom {encoded} eom")
            year_parts.append("eoy")
            parts.append(" ".join(year_parts))
        return " ".join(parts)

    def _encode_intraday_grouped(self, df: pd.DataFrame) -> str:
        """Group intraday bars by date with bod/eod markers."""
        parts = []
        for date, day_df in df.groupby(df.index.date):
            day_data = day_df.reset_index(drop=True)
            encoded = encode_price(day_data, p=self.period)
            parts.append(f"bod {encoded} eod")
        return " ".join(parts)

    @staticmethod
    def tokens_to_list(encoded_str: str) -> List[str]:
        """Split an encoded string into individual tokens."""
        return encoded_str.split()

    @staticmethod
    def parse_token(token: str) -> Optional[Dict]:
        """Parse a single token into its components.

        Returns
        -------
        dict with keys: type, direction, magnitude, raw
        Or None for marker tokens (boy, eoy, bom, eom, bod, eod).
        """
        if token in ("boy", "eoy", "bom", "eom", "bod", "eod"):
            return {"type": "marker", "value": token, "raw": token}

        match = _TOKEN_RE.match(token)
        if not match:
            return None

        prefix = match.group(1)
        magnitude = int(match.group(2))

        token_type = _TOKEN_TYPES.get(prefix)
        direction = _DIRECTION_MAP.get(prefix, 0)

        if token_type is None:
            return None

        return {
            "type": token_type,
            "direction": direction,
            "magnitude": magnitude,
            "raw": token,
        }

    @staticmethod
    def parse_bar_token(bar_token: str) -> Optional[Dict]:
        """Parse a composite bar token (e.g. 'H3P1R0V2') into components.

        Returns dict with pivot, net, range, volume sub-dicts.
        """
        # Split composite token into individual tokens using regex
        parts = re.findall(r"[A-Z]\d+", bar_token)
        if len(parts) != 4:
            return None

        result = {}
        for part in parts:
            parsed = PriceEncoder.parse_token(part)
            if parsed and parsed["type"] != "marker":
                result[parsed["type"]] = parsed
        return result if len(result) == 4 else None

    @staticmethod
    def similarity(seq_a: str, seq_b: str, n: int = 2) -> float:
        """Compute n-gram overlap similarity between two encoded sequences.

        Returns value in [0, 1] where 1 means identical n-gram distributions.
        """
        tokens_a = seq_a.split()
        tokens_b = seq_b.split()

        # Filter out markers
        markers = {"boy", "eoy", "bom", "eom", "bod", "eod"}
        tokens_a = [t for t in tokens_a if t not in markers]
        tokens_b = [t for t in tokens_b if t not in markers]

        if len(tokens_a) < n or len(tokens_b) < n:
            return 0.0

        ngrams_a = Counter(
            tuple(tokens_a[i : i + n]) for i in range(len(tokens_a) - n + 1)
        )
        ngrams_b = Counter(
            tuple(tokens_b[i : i + n]) for i in range(len(tokens_b) - n + 1)
        )

        intersection = sum((ngrams_a & ngrams_b).values())
        union = sum((ngrams_a | ngrams_b).values())

        return intersection / union if union > 0 else 0.0

    def get_last_n_encoded(self, df: pd.DataFrame, n: int = 5) -> str:
        """Encode full DataFrame but return only the last N bar tokens."""
        encoded = encode_price(df, p=self.period)
        tokens = encoded.split()
        return " ".join(tokens[-n:]) if len(tokens) >= n else encoded
