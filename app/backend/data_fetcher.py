"""
Data fetching module using Massive (formerly Massive) API for Nasdaq 100 symbols and stock data
"""

import pandas as pd
import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time
import asyncio
import aiohttp

from config import *

logger = logging.getLogger(__name__)

class MassiveDataFetcher:
    """Handles fetching Nasdaq 100 symbols and stock price data using Massive"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or MASSIVE_API_KEY
        self.base_url = MASSIVE_BASE_URL
        self.symbols = []
        self.failed_symbols = []
        
        if not self.api_key:
            logger.warning("No Massive API key provided. Some features may not work.")
    
    def get_nasdaq100_symbols(self, method: str = 'massive') -> List[str]:
        """
        Get Nasdaq 100 symbols using Massive
        
        Args:
            method: 'massive' (only supported method)
            
        Returns:
            List of stock symbols
        """
        try:
            if method == 'massive':
                return self._get_symbols_massive()
            else:
                raise ValueError(f"Unknown method: {method}. Only 'massive' is supported.")
                
        except Exception as e:
            logger.error(f"Failed to get symbols using {method}: {e}")
            # Fallback to a hardcoded list of major Nasdaq 100 symbols
            return self._get_fallback_symbols()
    
    def get_all_stock_symbols(self) -> List[str]:
        """
        Get all stock symbols from Massive (from api.py)
        
        Returns:
            List of all active stock symbols
        """
        logger.info("Getting All Stock Symbols")
        url = f'{BASE_URL}/v3/reference/tickers'
        tickers = []
        params = {
            'apikey': API_KEY,
            'limit': 1000,
            'market': 'stocks',
            'active': 'true',
            'type': 'CS'
        }
        
        while True:
            response = requests.get(url, params=params)
            data = response.json()
            if response.status_code != 200:
                logger.error(f"Error fetching data: {data.get('error', 'Unknown error')}")
                break
            
            if 'results' not in data or not data['results']:
                break
            
            # Extract tickers and add them to the list
            tickers.extend([symbol['ticker'] for symbol in data['results']])
            
            # Check if there's a next_url for pagination
            if 'next_url' in data:
                url = data['next_url']
                params = {'apikey': API_KEY}  # next_url includes other params
            else:
                break
        
        logger.info(f"Found {len(tickers)} Active Symbols")
        return tickers
    
    def get_finviz_symbols(self, portfolio_name: str) -> List[str]:
        """
        Get symbols from FinViz portfolio (from api.py)
        
        Args:
            portfolio_name: Name of the FinViz portfolio
            
        Returns:
            List of symbols from the portfolio
        """
        # This would require config_sources to be loaded
        # For now, return empty list and log warning
        logger.warning(f"FinViz integration requires config_sources setup. Portfolio: {portfolio_name}")
        logger.info("To use FinViz portfolios, ensure config/sources.yml is configured with FinViz credentials")
        return []
    
    def _get_symbols_massive(self) -> List[str]:
        """Get Nasdaq 100 symbols from Massive"""
        logger.info("Fetching Nasdaq 100 symbols from Massive")
        
        if not self.api_key:
            raise ValueError("Massive API key required for this method")
        
        # Get all active stocks first, then filter for Nasdaq 100
        url = f'{self.base_url}/v3/reference/tickers'
        symbols = []
        
        params = {
            'apikey': self.api_key,
            'limit': 1000,
            'market': 'stocks',
            'active': 'true',
            'type': 'CS',  # Common stocks
            'exchange': 'XNAS'  # Nasdaq exchange
        }
        
        try:
            while True:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'results' not in data:
                    break
                
                # Extract tickers
                batch_symbols = [ticker['ticker'] for ticker in data['results']]
                symbols.extend(batch_symbols)
                
                # Check pagination
                if 'next_url' in data:
                    url = data['next_url']
                    params = {'apikey': self.api_key}  # next_url includes other params
                else:
                    break
            
            logger.info(f"Retrieved {len(symbols)} Nasdaq symbols from Massive")
            
            # For now, return the major Nasdaq 100 symbols as Massive doesn't directly provide index constituents
            # In production, you would need a separate service or data source for exact index constituents
            major_nasdaq100 = self._get_fallback_symbols()
            return [s for s in major_nasdaq100 if s in symbols]
            
        except Exception as e:
            logger.error(f"Error fetching from Massive: {e}")
            return self._get_fallback_symbols()
    
    
    
    
    def _get_fallback_symbols(self) -> List[str]:
        """Fallback list of major Nasdaq 100 symbols"""
        logger.warning("Using fallback symbol list")
        
        return [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA',
            'NFLX', 'ADBE', 'ASML', 'PEP', 'COST', 'CSCO', 'AVGO', 'TXN',
            'QCOM', 'TMUS', 'AMAT', 'INTC', 'INTU', 'AMD', 'AMGN', 'ISRG',
            'HON', 'VRTX', 'SBUX', 'GILD', 'ADP', 'ADI', 'BKNG', 'MDLZ',
            'MU', 'LRCX', 'REGN', 'KLAC', 'MELI', 'PYPL', 'SNPS', 'CDNS',
            'CRWD', 'MAR', 'MRVL', 'ORLY', 'CSX', 'FTNT', 'ADSK', 'PANW',
            'AEP', 'NXPI', 'WDAY', 'ABNB', 'ROP', 'CHTR', 'ROST', 'PAYX',
            'CTAS', 'MNST', 'AZN', 'FAST', 'ODFL', 'TTD', 'BKR', 'EXC',
            'VRSK', 'KDP', 'EA', 'DDOG', 'XEL', 'CTSH', 'TEAM', 'GEHC',
            'KHC', 'IDXX', 'CCEP', 'FANG', 'ON', 'ZS', 'WBD', 'CSGP',
            'ANSS', 'DXCM', 'BIIB', 'GFS', 'ILMN', 'MDB', 'WBA', 'ARM',
            'CDW', 'SMCI', 'MRNA', 'ALGN'
        ]
    
    def fetch_bars(
        self,
        ticker: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch bars for any timeframe using Massive.

        Args:
            ticker: Stock symbol (e.g. 'AAPL') or crypto (e.g. 'X:BTCUSD')
            timeframe: One of '1min', '5min', '15min', '1h', '4h', '1d'
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with OHLCV data
        """
        tf_map = {
            '1min': (1, 'minute'), '5min': (5, 'minute'), '15min': (15, 'minute'),
            '1h': (1, 'hour'), '4h': (4, 'hour'), '1d': (1, 'day'),
        }
        multiplier, span = tf_map.get(timeframe, (1, 'day'))
        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{span}/{start_date}/{end_date}"
        params = {
            'apikey': API_KEY,
            'adjusted': 'true',
            'limit': 5000,
        }

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json().get('results', [])
                df = pd.DataFrame(data)
                if not df.empty:
                    df.columns = [col.lower() for col in df.columns]
                    df['date'] = pd.to_datetime(df['t'], unit='ms')
                    df.drop(columns=['t'], inplace=True)
                    column_mapping = {
                        'o': 'open', 'h': 'high', 'l': 'low',
                        'c': 'close', 'v': 'volume', 'vw': 'vwap', 'n': 'n_trades',
                    }
                    df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
                return df
            else:
                logger.warning(f"Failed to fetch {timeframe} data for {ticker}: HTTP {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching {timeframe} data for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_crypto_bars(
        self,
        pair: str,
        timeframe: str = '1d',
        start_date: str = None,
        end_date: str = None,
        days_back: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch crypto bars from Massive. Crypto uses 'X:BTCUSD' format.

        Args:
            pair: Crypto pair in Massive format (e.g. 'X:BTCUSD')
            timeframe: Bar timeframe
            start_date: Optional start date
            end_date: Optional end date
            days_back: Days of history if dates not specified
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        return self.fetch_bars(pair, timeframe, start_date, end_date)

    def fetch_historical_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical data for a single ticker using Massive
        Updated to match api.py implementation exactly

        Args:
            ticker: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with OHLCV data
        """
        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {
            'apikey': API_KEY,
            'adjusted': 'true'
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json().get('results', [])
                # Convert to DataFrame
                df = pd.DataFrame(data)
                if not df.empty:
                    # Rename columns to lower case first
                    df.columns = [col.lower() for col in df.columns]
                    
                    # Convert timestamp to datetime column
                    df['date'] = pd.to_datetime(df['t'], unit='ms')
                    df.drop(columns=['t'], inplace=True)
                    
                    # Map Massive column names to standard OHLCV names
                    column_mapping = {
                        'o': 'open',
                        'h': 'high', 
                        'l': 'low',
                        'c': 'close',
                        'v': 'volume',
                        'vw': 'vwap',
                        'n': 'n_trades'
                    }
                    
                    # Only rename columns that exist
                    df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
                    
                return df
            else:
                logger.warning(f"Failed to fetch data for {ticker}: HTTP {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_stock_data(self, symbols: List[str], days_back: int = HISTORY_DAYS) -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data for multiple symbols using Massive
        Based on api.py pattern
        
        Args:
            symbols: List of stock symbols
            days_back: Number of days of historical data to fetch
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        logger.info(f"Fetching stock data for {len(symbols)} symbols over {days_back} days")
        
        # Calculate date range
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        stock_data = {}
        self.failed_symbols = []
        
        # Process symbols with rate limiting
        for i, symbol in enumerate(symbols):
            try:
                if i > 0 and i % 10 == 0:
                    logger.info(f"Processed {i}/{len(symbols)} symbols")
                
                df = self.fetch_historical_data(symbol, start_date, end_date)
                
                if not df.empty:
                    # Additional data validation
                    df = df.dropna()
                    if len(df) >= 20:  # Minimum data requirement
                        stock_data[symbol] = df
                        logger.debug(f"Successfully processed {symbol}: {len(df)} rows")
                    else:
                        self.failed_symbols.append(symbol)
                        logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                else:
                    self.failed_symbols.append(symbol)
                    logger.warning(f"No data retrieved for {symbol}")
                
                # Rate limiting - Massive free tier allows 5 requests per minute
                time.sleep(0.1)  # Be conservative
                
            except Exception as e:
                self.failed_symbols.append(symbol)
                logger.error(f"Error processing {symbol}: {e}")
        
        logger.info(f"Successfully retrieved data for {len(stock_data)} symbols")
        if self.failed_symbols:
            logger.warning(f"Failed to retrieve data for {len(self.failed_symbols)} symbols: {self.failed_symbols}")
        
        return stock_data
    
    def get_single_stock_data(self, symbol: str, days_back: int = HISTORY_DAYS) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single stock symbol
        
        Args:
            symbol: Stock symbol
            days_back: Number of days of historical data
            
        Returns:
            DataFrame with stock data or None if failed
        """
        logger.info(f"Fetching data for {symbol}")
        
        # Calculate date range
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        df = self.fetch_historical_data(symbol, start_date, end_date)
        
        if df.empty:
            logger.warning(f"No data available for {symbol}")
            return None
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        if df.empty:
            logger.warning(f"No valid data for {symbol} after cleaning")
            return None
        
        logger.info(f"Retrieved {len(df)} rows of data for {symbol}")
        return df
    
    async def update_snapshot(self, ticker: str, retries: int = 3, backoff_secs: int = 1) -> Dict:
        """
        Update a stock's snapshot from Massive (async version from api.py)
        
        Args:
            ticker: Stock symbol
            retries: Number of retry attempts
            backoff_secs: Backoff time between retries
            
        Returns:
            Dictionary with snapshot data
        """
        if not self.api_key:
            logger.error("Massive API key required for snapshots")
            return {}
        
        url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        params = {'apikey': self.api_key}
        
        attempt = 0
        while attempt < retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            ticker_info = data.get('ticker', {})
                            day_info = ticker_info.get('day', {})
                            
                            snapshot = {
                                'symbol': ticker,
                                'close': day_info.get('c', 0),
                                'high': day_info.get('h', 0),
                                'low': day_info.get('l', 0),
                                'open': day_info.get('o', 0),
                                'volume': day_info.get('v', 0),
                                'vwap': day_info.get('vw', 0),
                                'change_pct': round(ticker_info.get('todaysChangePerc', 0), 2),
                                'change': round(ticker_info.get('todaysChange', 0), 2),
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            logger.debug(f"Snapshot for {ticker}: {snapshot}")
                            return snapshot
                        else:
                            logger.error(f"Failed to fetch snapshot for {ticker}, status: {response.status}")
            
            except aiohttp.ClientError as e:
                logger.error(f"Attempt {attempt + 1} failed for {ticker}: {e}")
            
            # Increment attempt and wait before retrying
            attempt += 1
            if attempt < retries:
                await asyncio.sleep(backoff_secs * attempt)
        
        logger.error(f"Failed to fetch snapshot for {ticker} after {retries} attempts")
        return {}
    
    def refresh_symbols(self, primary_method: str = 'massive', fallback_methods: List[str] = None) -> List[str]:
        """
        Refresh the symbol list with fallback methods
        
        Args:
            primary_method: Primary method to try
            fallback_methods: List of fallback methods
            
        Returns:
            List of symbols
        """
        if fallback_methods is None:
            fallback_methods = ['all_symbols']
        
        methods_to_try = [primary_method] + fallback_methods
        
        for method in methods_to_try:
            try:
                if method == 'all_symbols':
                    symbols = self.get_all_stock_symbols()
                elif method == 'finviz':
                    symbols = self.get_finviz_symbols('nasdaq100')  # Default portfolio name
                else:
                    symbols = self.get_nasdaq100_symbols(method)
                
                if symbols:
                    self.symbols = symbols
                    logger.info(f"Successfully retrieved {len(symbols)} symbols using {method}")
                    return symbols
            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")
                continue
        
        # If all methods fail, use fallback
        logger.error("All symbol fetching methods failed, using fallback list")
        self.symbols = self._get_fallback_symbols()
        return self.symbols
    
    def load_symbols_by_source(self, source: str, portfolio_name: str = 'nasdaq100') -> List[str]:
        """
        Load symbols based on source configuration
        
        Args:
            source: Data source ('massive', 'finviz', 'alphapy')
            portfolio_name: Portfolio/group name
            
        Returns:
            List of symbols
        """
        logger.info(f"Loading symbols from source: {source}")
        
        symbols = []
        
        if source == 'massive':
            # Use get_all_stock_symbols for all Massive symbols
            symbols = self.get_all_stock_symbols()
        elif source == 'finviz':
            # Use FinViz portfolio
            symbols = self.get_finviz_symbols(portfolio_name)
        elif source == 'alphapy':
            # This would require loading from config_groups (not implemented yet)
            logger.warning("AlphaPy source requires config_groups setup")
            logger.info("Falling back to hardcoded Nasdaq 100 symbols")
            symbols = self._get_fallback_symbols()
        else:
            logger.error(f'Unknown Group Source: {source}')
            symbols = self._get_fallback_symbols()
        
        # Convert to uppercase
        symbols = [sym.upper() for sym in symbols]
        logger.info(f"Source {source}: {len(symbols)} symbols")
        
        self.symbols = symbols
        return symbols

# For backward compatibility, create an alias
DataFetcher = MassiveDataFetcher