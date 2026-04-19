"""
Pattern analysis module for detecting pivot patterns in stock data
"""

import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from pivots import (
    gartley_bullish, gartley_bearish,
    abcd_bullish, abcd_bearish,
    drive3_bullish, drive3_bearish,
    wolfe_bullish, wolfe_bearish,
    expansion_bullish, expansion_bearish,
    squeeze_bullish, squeeze_bearish,
    rectangle_neutral, wedge_neutral,
    pivothigh, pivotlow
)

from config import PATTERN_FUNCTIONS, PATTERN_DISPLAY_NAMES, WINDOW_LENGTH, MINIMUM_STRENGTH

logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """Analyzes stock data for various pivot patterns"""
    
    def __init__(self, window_length: int = WINDOW_LENGTH, minimum_strength: int = MINIMUM_STRENGTH):
        self.window_length = window_length
        self.minimum_strength = minimum_strength
        
        # Map function names to actual functions
        self.pattern_functions = {
            'gartley_bullish': gartley_bullish,
            'gartley_bearish': gartley_bearish,
            'abcd_bullish': abcd_bullish,
            'abcd_bearish': abcd_bearish,
            'drive3_bullish': drive3_bullish,
            'drive3_bearish': drive3_bearish,
            'wolfe_bullish': wolfe_bullish,
            'wolfe_bearish': wolfe_bearish,
            'expansion_bullish': expansion_bullish,
            'expansion_bearish': expansion_bearish,
            'squeeze_bullish': squeeze_bullish,
            'squeeze_bearish': squeeze_bearish,
            'rectangle_neutral': rectangle_neutral,
            'wedge_neutral': wedge_neutral
        }
    
    def analyze_stock(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a single stock for all pattern types
        
        Args:
            symbol: Stock symbol
            df: DataFrame with OHLC data
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing patterns for {symbol}")
        
        # Ensure we have enough data
        if len(df) < self.window_length:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} rows, need {self.window_length}")
            return {
                'symbol': symbol,
                'error': f'Insufficient data: {len(df)} rows, need {self.window_length}',
                'patterns': {},
                'pivot_points': {
                    'total_high_pivots': 0,
                    'total_low_pivots': 0
                },
                'summary': {
                    'total_patterns_detected': 0,
                    'bullish_patterns_count': 0,
                    'bearish_patterns_count': 0,
                    'overall_sentiment': 'neutral'
                },
                'price_info': {
                    'current_price': float(df['close'].iloc[-1]) if len(df) > 0 else 0.0,
                    'previous_close': float(df['close'].iloc[-2]) if len(df) > 1 else None,
                    'daily_change': 0.0,
                    'daily_change_pct': 0.0,
                    'high_52w': float(df['high'].max()) if len(df) > 0 else 0.0,
                    'low_52w': float(df['low'].min()) if len(df) > 0 else 0.0
                },
                'analysis_date': datetime.now().isoformat(),
                'data_rows': len(df)
            }
        
        results = {
            'symbol': symbol,
            'patterns': {},
            'pivot_points': {},
            'analysis_date': datetime.now().isoformat(),
            'data_rows': len(df),
            'price_info': {
                'current_price': float(df['close'].iloc[-1]),
                'previous_close': float(df['close'].iloc[-2]) if len(df) > 1 else None,
                'daily_change': None,
                'daily_change_pct': None,
                'high_52w': float(df['high'].max()),
                'low_52w': float(df['low'].min())
            }
        }
        
        # Calculate daily change
        if results['price_info']['previous_close']:
            results['price_info']['daily_change'] = results['price_info']['current_price'] - results['price_info']['previous_close']
            results['price_info']['daily_change_pct'] = (results['price_info']['daily_change'] / results['price_info']['previous_close']) * 100
        
        try:
            # Get pivot points first
            high_pivots = pivothigh(df, self.window_length, self.minimum_strength)
            low_pivots = pivotlow(df, self.window_length, self.minimum_strength)
            
            results['pivot_points'] = {
                'high_pivots': high_pivots,
                'low_pivots': low_pivots,
                'total_high_pivots': len(high_pivots),
                'total_low_pivots': len(low_pivots)
            }
            
            # Analyze each pattern type
            for pattern_name in PATTERN_FUNCTIONS:
                try:
                    pattern_func = self.pattern_functions[pattern_name]
                    pattern_result = pattern_func(df, self.window_length)
                    
                    if pattern_result:  # Pattern detected
                        results['patterns'][pattern_name] = {
                            'detected': True,
                            'pattern_data': pattern_result,
                            'display_name': PATTERN_DISPLAY_NAMES.get(pattern_name, pattern_name),
                            'pattern_type': self._get_pattern_type(pattern_name),
                            'sentiment': self._get_pattern_sentiment(pattern_name)
                        }
                        logger.info(f"Detected {pattern_name} pattern for {symbol}")
                    else:
                        results['patterns'][pattern_name] = {
                            'detected': False,
                            'display_name': PATTERN_DISPLAY_NAMES.get(pattern_name, pattern_name),
                            'pattern_type': self._get_pattern_type(pattern_name),
                            'sentiment': self._get_pattern_sentiment(pattern_name)
                        }
                        
                except Exception as e:
                    logger.error(f"Error analyzing {pattern_name} for {symbol}: {e}")
                    results['patterns'][pattern_name] = {
                        'detected': False,
                        'error': str(e),
                        'display_name': PATTERN_DISPLAY_NAMES.get(pattern_name, pattern_name),
                        'pattern_type': self._get_pattern_type(pattern_name),
                        'sentiment': self._get_pattern_sentiment(pattern_name)
                    }
            
            # Summary statistics
            detected_patterns = [name for name, data in results['patterns'].items() if data.get('detected', False)]
            bullish_patterns = [name for name in detected_patterns if 'bullish' in name]
            bearish_patterns = [name for name in detected_patterns if 'bearish' in name]
            neutral_patterns = [name for name in detected_patterns if 'neutral' in name]
            
            results['summary'] = {
                'total_patterns_detected': len(detected_patterns),
                'bullish_patterns_count': len(bullish_patterns),
                'bearish_patterns_count': len(bearish_patterns),
                'neutral_patterns_count': len(neutral_patterns),
                'detected_patterns': detected_patterns,
                'bullish_patterns': bullish_patterns,
                'bearish_patterns': bearish_patterns,
                'neutral_patterns': neutral_patterns,
                'overall_sentiment': self._calculate_overall_sentiment(bullish_patterns, bearish_patterns, neutral_patterns)
            }
            
        except Exception as e:
            logger.error(f"Error in pattern analysis for {symbol}: {e}")
            results['error'] = str(e)
        
        return results
    
    def analyze_multiple_stocks(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple stocks for patterns
        
        Args:
            stock_data: Dictionary mapping symbols to DataFrames
            
        Returns:
            Dictionary mapping symbols to analysis results
        """
        logger.info(f"Analyzing patterns for {len(stock_data)} stocks")
        
        results = {}
        successful_analyses = 0
        
        for symbol, df in stock_data.items():
            try:
                results[symbol] = self.analyze_stock(symbol, df)
                if 'error' not in results[symbol]:
                    successful_analyses += 1
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                results[symbol] = {
                    'symbol': symbol,
                    'error': str(e),
                    'patterns': {},
                    'pivot_points': {},
                    'analysis_date': datetime.now().isoformat()
                }
        
        logger.info(f"Successfully analyzed {successful_analyses}/{len(stock_data)} stocks")
        return results
    
    def get_pattern_summary(self, analysis_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics across all analyzed stocks
        
        Args:
            analysis_results: Results from analyze_multiple_stocks
            
        Returns:
            Summary statistics
        """
        total_stocks = len(analysis_results)
        successful_analyses = len([r for r in analysis_results.values() if 'error' not in r])
        
        # Count patterns across all stocks
        pattern_counts = {}
        bullish_stocks = []
        bearish_stocks = []
        neutral_stocks = []
        
        for symbol, result in analysis_results.items():
            if 'error' in result:
                continue
                
            summary = result.get('summary', {})
            sentiment = summary.get('overall_sentiment', 'neutral')
            
            if sentiment == 'bullish':
                bullish_stocks.append(symbol)
            elif sentiment == 'bearish':
                bearish_stocks.append(symbol)
            else:
                neutral_stocks.append(symbol)
            
            # Count individual patterns
            for pattern_name, pattern_data in result.get('patterns', {}).items():
                if pattern_data.get('detected', False):
                    pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
        
        # Find most common patterns
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'analysis_date': datetime.now().isoformat(),
            'total_stocks_analyzed': total_stocks,
            'successful_analyses': successful_analyses,
            'failed_analyses': total_stocks - successful_analyses,
            'sentiment_distribution': {
                'bullish': len(bullish_stocks),
                'bearish': len(bearish_stocks),
                'neutral': len(neutral_stocks)
            },
            'sentiment_details': {
                'bullish_stocks': bullish_stocks,
                'bearish_stocks': bearish_stocks,
                'neutral_stocks': neutral_stocks
            },
            'pattern_counts': pattern_counts,
            'most_common_patterns': sorted_patterns[:5],
            'total_patterns_detected': sum(pattern_counts.values())
        }
    
    def _get_pattern_type(self, pattern_name: str) -> str:
        """Get the base pattern type (e.g., 'gartley', 'abcd')"""
        if 'gartley' in pattern_name:
            return 'gartley'
        elif 'abcd' in pattern_name:
            return 'abcd'
        elif 'drive3' in pattern_name:
            return 'three_drive'
        elif 'wolfe' in pattern_name:
            return 'wolfe_wave'
        elif 'expansion' in pattern_name:
            return 'expansion'
        elif 'squeeze' in pattern_name:
            return 'squeeze'
        elif 'rectangle' in pattern_name:
            return 'rectangle'
        elif 'wedge' in pattern_name:
            return 'wedge'
        else:
            return 'unknown'
    
    def _get_pattern_sentiment(self, pattern_name: str) -> str:
        """Get the sentiment of the pattern (bullish/bearish)"""
        if 'bullish' in pattern_name:
            return 'bullish'
        elif 'bearish' in pattern_name:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_overall_sentiment(self, bullish_patterns: List[str], bearish_patterns: List[str], neutral_patterns: List[str] = None) -> str:
        """Calculate overall sentiment based on detected patterns"""
        bullish_count = len(bullish_patterns)
        bearish_count = len(bearish_patterns)
        neutral_count = len(neutral_patterns) if neutral_patterns else 0
        
        # If there are neutral patterns, they indicate consolidation/indecision
        if neutral_count > 0 and neutral_count >= max(bullish_count, bearish_count):
            return 'neutral'
        elif bullish_count > bearish_count:
            return 'bullish'
        elif bearish_count > bullish_count:
            return 'bearish'
        else:
            return 'neutral'
    
    def filter_stocks_by_patterns(self, analysis_results: Dict[str, Dict[str, Any]], 
                                 pattern_types: Optional[List[str]] = None,
                                 sentiment: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Filter stocks based on detected patterns
        
        Args:
            analysis_results: Results from analyze_multiple_stocks
            pattern_types: List of pattern types to filter by
            sentiment: 'bullish', 'bearish', or None
            
        Returns:
            Filtered analysis results
        """
        filtered_results = {}
        
        for symbol, result in analysis_results.items():
            if 'error' in result:
                continue
            
            should_include = True
            
            # Filter by sentiment
            if sentiment:
                stock_sentiment = result.get('summary', {}).get('overall_sentiment', 'neutral')
                if stock_sentiment != sentiment:
                    should_include = False
            
            # Filter by pattern types
            if pattern_types and should_include:
                detected_pattern_types = []
                for pattern_name, pattern_data in result.get('patterns', {}).items():
                    if pattern_data.get('detected', False):
                        pattern_type = self._get_pattern_type(pattern_name)
                        detected_pattern_types.append(pattern_type)
                
                if not any(pt in detected_pattern_types for pt in pattern_types):
                    should_include = False
            
            if should_include:
                filtered_results[symbol] = result
        
        return filtered_results