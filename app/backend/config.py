"""
Configuration for AlphaPy Markets Backend
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Massive (formerly Polygon.io) Configuration
MASSIVE_API_KEY = os.getenv('MASSIVE_API_KEY', '') or os.getenv('POLYGON_API_KEY', '')
MASSIVE_BASE_URL = "https://api.massive.com"

# For compatibility with data_fetcher functions
BASE_URL = MASSIVE_BASE_URL
API_KEY = MASSIVE_API_KEY

# Analysis Configuration
WINDOW_LENGTH = int(os.getenv('WINDOW_LENGTH', '100'))
MINIMUM_STRENGTH = int(os.getenv('MINIMUM_STRENGTH', '5'))
HISTORY_DAYS = int(os.getenv('HISTORY_DAYS', '100'))

# Data Configuration
USE_LIVE_DATA = os.getenv('USE_LIVE_DATA', 'false').lower() == 'true'
RATE_LIMIT = int(os.getenv('RATE_LIMIT', '5'))

# Portfolio Configuration
PORTFOLIO_SOURCE = os.getenv('PORTFOLIO_SOURCE', 'massive')
PORTFOLIO_NAME = os.getenv('PORTFOLIO_NAME', 'nasdaq100')

# Substack Configuration
SUBSTACK_EMAIL = os.getenv('SUBSTACK_EMAIL', '')
SUBSTACK_PASSWORD = os.getenv('SUBSTACK_PASSWORD', '')
SUBSTACK_COOKIES_PATH = os.getenv('SUBSTACK_COOKIES_PATH', '')
SUBSTACK_PUBLICATION_URL = os.getenv('SUBSTACK_PUBLICATION_URL', '')

# Pattern Types to Analyze
PATTERN_FUNCTIONS = [
    'gartley_bullish',
    'gartley_bearish',
    'abcd_bullish',
    'abcd_bearish',
    'drive3_bullish',
    'drive3_bearish',
    'wolfe_bullish',
    'wolfe_bearish',
    'expansion_bullish',
    'expansion_bearish',
    'squeeze_bullish',
    'squeeze_bearish',
    'rectangle_neutral',
    'wedge_neutral'
]

# Pattern Display Names
PATTERN_DISPLAY_NAMES = {
    'gartley_bullish': 'Bullish Gartley',
    'gartley_bearish': 'Bearish Gartley',
    'abcd_bullish': 'Bullish ABCD',
    'abcd_bearish': 'Bearish ABCD',
    'drive3_bullish': 'Bullish Three-Drive',
    'drive3_bearish': 'Bearish Three-Drive',
    'wolfe_bullish': 'Bullish Wolfe Wave',
    'wolfe_bearish': 'Bearish Wolfe Wave',
    'expansion_bullish': 'Bullish Expansion',
    'expansion_bearish': 'Bearish Expansion',
    'squeeze_bullish': 'Bullish Squeeze',
    'squeeze_bearish': 'Bearish Squeeze',
    'rectangle_neutral': 'Rectangle',
    'wedge_neutral': 'Wedge'
}

# Colors for different pattern types
PATTERN_COLORS = {
    'gartley_bullish': '#00FF00',    # Green
    'gartley_bearish': '#FF0000',    # Red
    'abcd_bullish': '#00AA00',       # Dark Green
    'abcd_bearish': '#AA0000',       # Dark Red
    'drive3_bullish': '#66FF66',     # Light Green
    'drive3_bearish': '#FF6666',     # Light Red
    'wolfe_bullish': '#0066FF',      # Blue
    'wolfe_bearish': '#FF6600',      # Orange
    'expansion_bullish': '#00FFFF',  # Cyan
    'expansion_bearish': '#FF00FF',  # Magenta
    'squeeze_bullish': '#99FF99',    # Pale Green
    'squeeze_bearish': '#FF9999',    # Pale Red
    'rectangle_neutral': '#FFA500',  # Orange
    'wedge_neutral': '#800080'       # Purple
}
