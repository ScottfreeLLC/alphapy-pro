"""
Pattern library for encoded price sequences.

Defines bullish, bearish, and continuation patterns as regex-based rules
that match against encoded bar token strings.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .encoder import encode_price_df


@dataclass
class PatternMatch:
    name: str
    pattern_type: str  # "bullish", "bearish", "continuation"
    bars_matched: int
    start_idx: int
    end_idx: int
    description: str


# ---------------------------------------------------------------------------
# Pattern definitions — regex patterns against space-separated bar tokens
#
# Each bar token is like H3P1R0V2 (composite of pivot+net+range+volume).
# Patterns match sequences of these composite tokens.
# ---------------------------------------------------------------------------

_PATTERNS = {
    # --- Bullish ---
    "bullish_reversal": {
        "type": "bullish",
        # High pivot low strength (L5+) followed by positive net + expanding range + high volume
        "regex": r"L[5-9]\d*N\d[RV]\d\s+\S*P[12]R[12]V[12]",
        "description": "Deep pivot low followed by strong bullish bar with expanding range and volume",
        "min_bars": 2,
    },
    "bullish_hammer": {
        "type": "bullish",
        # Pivot low + small net + large range (long wick) + high volume
        "regex": r"L[3-9]\d*[PNZ][01]R[12]V[12]",
        "description": "Pivot low with large range (hammer-like) on high volume",
        "min_bars": 1,
    },
    "bullish_accumulation": {
        "type": "bullish",
        # 3+ bars near pivot low with increasing volume
        "regex": r"(?:L\d+[NZ]\dR0V[01]\s+){2,}L\d+P\dR\dV[12]",
        "description": "Tight range near pivot low with volume surge on breakout",
        "min_bars": 3,
    },
    "bullish_momentum": {
        "type": "bullish",
        # 2+ consecutive positive bars with expanding range
        "regex": r"(?:\S*P[12]R[12]V\d\s+){2,}",
        "description": "Consecutive strong positive bars with expanding range",
        "min_bars": 2,
    },

    # --- Bearish ---
    "bearish_reversal": {
        "type": "bearish",
        # High pivot high followed by negative net + expanding range + volume
        "regex": r"H[5-9]\d*P\d[RV]\d\s+\S*N[12]R[12]V[12]",
        "description": "Pivot high followed by strong bearish bar with expanding range and volume",
        "min_bars": 2,
    },
    "bearish_shooting_star": {
        "type": "bearish",
        # Pivot high + small net + large range + high volume
        "regex": r"H[3-9]\d*[PNZ][01]R[12]V[12]",
        "description": "Pivot high with large range (shooting star) on high volume",
        "min_bars": 1,
    },
    "bearish_distribution": {
        "type": "bearish",
        # Bars near pivot high with volume surge on breakdown
        "regex": r"(?:H\d+[PZ]\dR0V[01]\s+){2,}H\d+N\dR\dV[12]",
        "description": "Tight range near pivot high with volume surge on breakdown",
        "min_bars": 3,
    },
    "bearish_momentum": {
        "type": "bearish",
        # 2+ consecutive negative bars with expanding range
        "regex": r"(?:\S*N[12]R[12]V\d\s+){2,}",
        "description": "Consecutive strong negative bars with expanding range",
        "min_bars": 2,
    },

    # --- Continuation ---
    "trend_continuation_up": {
        "type": "continuation",
        # Strong up, pause (small net), then strong up again
        "regex": r"\S*P[12]R\dV\d\s+\S*[PZ][01]R0V[01]\s+\S*P[12]R[12]V\d",
        "description": "Bullish bar, pause, then another bullish bar (flag continuation)",
        "min_bars": 3,
    },
    "trend_continuation_down": {
        "type": "continuation",
        # Strong down, pause, then strong down again
        "regex": r"\S*N[12]R\dV\d\s+\S*[NZ][01]R0V[01]\s+\S*N[12]R[12]V\d",
        "description": "Bearish bar, pause, then another bearish bar (flag continuation)",
        "min_bars": 3,
    },
    "volume_climax_reversal": {
        "type": "bullish",
        # Extreme volume on a negative bar followed by positive bar
        "regex": r"\S*N\dR\dV2\s+\S*P\dR\dV\d",
        "description": "Volume climax on selling followed by buying (capitulation reversal)",
        "min_bars": 2,
    },
}


def find_patterns(
    encoded_str: str,
    pattern_types: Optional[List[str]] = None,
) -> List[PatternMatch]:
    """Scan an encoded token string for known patterns.

    Parameters
    ----------
    encoded_str : str
        Space-separated encoded bar tokens.
    pattern_types : list of str, optional
        Filter to specific types: "bullish", "bearish", "continuation".
        If None, all patterns are checked.

    Returns
    -------
    list of PatternMatch
    """
    matches = []
    tokens = encoded_str.split()
    # Filter out markers
    markers = {"boy", "eoy", "bom", "eom", "bod", "eod"}
    tokens = [t for t in tokens if t not in markers]
    clean_str = " ".join(tokens)

    for name, spec in _PATTERNS.items():
        if pattern_types and spec["type"] not in pattern_types:
            continue

        for m in re.finditer(spec["regex"], clean_str):
            matched_text = m.group()
            bar_count = len(matched_text.split())
            # Find bar index by counting spaces before match start
            prefix = clean_str[: m.start()]
            start_idx = len(prefix.split()) - 1 if prefix.strip() else 0
            if not prefix.strip():
                start_idx = 0

            matches.append(
                PatternMatch(
                    name=name,
                    pattern_type=spec["type"],
                    bars_matched=bar_count,
                    start_idx=start_idx,
                    end_idx=start_idx + bar_count - 1,
                    description=spec["description"],
                )
            )

    return matches


def find_patterns_in_df(
    df: pd.DataFrame,
    p: int = 20,
    pattern_types: Optional[List[str]] = None,
) -> List[PatternMatch]:
    """Encode a DataFrame and find patterns in the resulting tokens."""
    from .encoder import encode_price
    encoded = encode_price(df, p=p)
    return find_patterns(encoded, pattern_types=pattern_types)


def get_pattern_catalog() -> List[Dict]:
    """Return the full pattern catalog with metadata."""
    return [
        {
            "name": name,
            "type": spec["type"],
            "description": spec["description"],
            "min_bars": spec["min_bars"],
        }
        for name, spec in _PATTERNS.items()
    ]


def last_bar_signal(df: pd.DataFrame, p: int = 20, lookback: int = 5) -> Optional[str]:
    """Check if the last few bars match any pattern; return signal direction or None.

    Returns "long", "short", or None.
    """
    if len(df) < lookback:
        return None

    tail = df.iloc[-lookback:].reset_index(drop=True)
    from .encoder import encode_price
    encoded = encode_price(tail, p=p)
    matches = find_patterns(encoded)

    if not matches:
        return None

    # Score by type: bullish=+1, bearish=-1, continuation depends on context
    score = 0
    for m in matches:
        if m.pattern_type == "bullish":
            score += 1
        elif m.pattern_type == "bearish":
            score -= 1

    if score > 0:
        return "long"
    elif score < 0:
        return "short"
    return None
