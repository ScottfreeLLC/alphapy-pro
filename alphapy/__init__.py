"""AlphaPy: a domain-agnostic ML pipeline framework.

Trading, markets, and Alfi were split off into the private alphapy-finance
repo in v4.0.0. See CHANGELOG.md and tag v3.1.1-monolith for the pre-split
state.
"""

__version__ = "4.0.0"

import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

__all__ = [
    "__version__",
]
