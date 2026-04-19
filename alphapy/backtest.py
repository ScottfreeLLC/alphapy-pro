################################################################################
#
# Package   : AlphaPy
# Module    : backtest
# Created   : January 2025
#
# Copyright 2025 ScottFree Analytics LLC
# Mark Conway & Robert D. Scott II
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

"""
Vectorbt-based backtesting module for AlphaPy.

This module provides vectorized backtesting using vectorbt, replacing
the event-driven portfolio simulation in portfolio.py.
"""

#
# Imports
#

from alphapy.frame import Frame
from alphapy.frame import frame_name
from alphapy.frame import write_frame
from alphapy.globals import SSEP, USEP
from alphapy.globals import Orders
from alphapy.metalabel import get_vol_ema
from alphapy.space import Space

import json
import logging
import numpy as np
import pandas as pd
import vectorbt as vbt


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function signals_to_vectorbt
#

def signals_to_vectorbt(tradelist, symbols, price_df):
    """
    Convert AlphaPy trade list to vectorbt boolean signal DataFrames.

    Parameters
    ----------
    tradelist : list
        List of trade tuples from run_system():
        [(datetime, [symbol, order_type, quantity, price]), ...]
        where order_type is one of: Orders.le, Orders.se, Orders.lx, Orders.sx,
        Orders.lh, Orders.sh
    symbols : list
        List of symbols in the trading universe.
    price_df : pandas.DataFrame
        DataFrame with DatetimeIndex and price data for all symbols.

    Returns
    -------
    entries : pandas.DataFrame
        Boolean DataFrame for long entries (index=dates, columns=symbols).
    exits : pandas.DataFrame
        Boolean DataFrame for long exits (index=dates, columns=symbols).
    short_entries : pandas.DataFrame
        Boolean DataFrame for short entries (index=dates, columns=symbols).
    short_exits : pandas.DataFrame
        Boolean DataFrame for short exits (index=dates, columns=symbols).

    """
    logger.info("Converting AlphaPy trade signals to vectorbt format")

    # Get the date index from price_df
    dates = price_df.index

    # Normalize symbols to uppercase for matching
    symbols_upper = [s.upper() for s in symbols]

    # Initialize boolean DataFrames with False values
    entries = pd.DataFrame(False, index=dates, columns=symbols_upper)
    exits = pd.DataFrame(False, index=dates, columns=symbols_upper)
    short_entries = pd.DataFrame(False, index=dates, columns=symbols_upper)
    short_exits = pd.DataFrame(False, index=dates, columns=symbols_upper)

    # Process each trade in the trade list
    for trade_date, trade_data in tradelist:
        symbol, order_type, quantity, price = trade_data
        symbol = symbol.upper()

        if symbol not in symbols_upper:
            logger.warning("Symbol %s not in trading universe", symbol)
            continue

        # Find the nearest date in the index
        try:
            # Convert trade_date to proper format if needed
            if isinstance(trade_date, str):
                trade_date = pd.to_datetime(trade_date)

            # Get the nearest date in our index
            idx = dates.get_indexer([trade_date], method='nearest')[0]
            if idx >= 0 and idx < len(dates):
                actual_date = dates[idx]
            else:
                continue
        except Exception as e:
            logger.warning("Could not map trade date %s: %s", trade_date, e)
            continue

        # Map order types to signal DataFrames
        if order_type == Orders.le:
            entries.loc[actual_date, symbol] = True
        elif order_type in (Orders.lx, Orders.lh):
            exits.loc[actual_date, symbol] = True
        elif order_type == Orders.se:
            short_entries.loc[actual_date, symbol] = True
        elif order_type in (Orders.sx, Orders.sh):
            short_exits.loc[actual_date, symbol] = True

    # Log signal counts
    logger.info("Long entries: %d", entries.sum().sum())
    logger.info("Long exits: %d", exits.sum().sum())
    logger.info("Short entries: %d", short_entries.sum().sum())
    logger.info("Short exits: %d", short_exits.sum().sum())

    return entries, exits, short_entries, short_exits


#
# Class VBTBacktester
#

class VBTBacktester:
    """
    Vectorbt-based backtester for AlphaPy systems.

    This class wraps vectorbt's Portfolio.from_signals() to provide
    a drop-in replacement for AlphaPy's event-driven backtesting.

    Parameters
    ----------
    model : alphapy.Model
        The model specifications.
    system : alphapy.System or alphapy.SystemRank
        The trading system with parameters.
    group : alphapy.Group
        The group of instruments in the portfolio.
    portfolio_specs : dict
        Portfolio configuration from market.yml.

    Attributes
    ----------
    _portfolio : vbt.Portfolio
        The vectorbt Portfolio object after running backtest.
    _price_df : pandas.DataFrame
        Price data used for backtesting.

    """

    def __init__(self, model, system, group, portfolio_specs):
        self.model = model
        self.system = system
        self.group = group
        self.portfolio_specs = portfolio_specs
        self._portfolio = None
        self._price_df = None
        self._symbols = None

    def _build_price_dataframe(self):
        """
        Build a consolidated price DataFrame from AlphaPy Frames.

        Returns
        -------
        price_df : pandas.DataFrame
            DataFrame with columns for each symbol's close price.

        """
        logger.info("Building price DataFrame for vectorbt")

        gspace = self.group.space
        symbols = self.group.members
        self._symbols = [s.upper() for s in symbols]

        fractal = self.system.fractal
        col_close = USEP.join(['close', fractal])

        price_frames = []
        for symbol in symbols:
            symbol_lower = symbol.lower()
            tspace = Space(gspace.subject, gspace.source, 'ALL')
            fname = frame_name(symbol_lower, tspace)

            if fname in Frame.frames:
                df = Frame.frames[fname].df.copy()
                # Convert to pandas if needed
                if hasattr(df, 'to_pandas'):
                    df = df.to_pandas()
                    if 'datetime' in df.columns:
                        df.set_index('datetime', inplace=True)

                if col_close in df.columns:
                    price_series = df[col_close].rename(symbol.upper())
                    price_frames.append(price_series)
                else:
                    logger.warning("Close column %s not found for %s", col_close, symbol)
            else:
                logger.warning("Frame not found for %s", symbol)

        if not price_frames:
            raise ValueError("No price data available for backtesting")

        # Concatenate all price series
        self._price_df = pd.concat(price_frames, axis=1)
        self._price_df.index = pd.to_datetime(self._price_df.index)
        self._price_df = self._price_df.sort_index()

        # Forward fill any gaps
        self._price_df = self._price_df.ffill()

        logger.info("Price DataFrame shape: %s", self._price_df.shape)
        return self._price_df

    def _calculate_stops(self):
        """
        Calculate stop loss and take profit levels based on volatility.

        Returns
        -------
        tp_stop : float or None
            Take profit stop as a fraction (e.g., 0.05 for 5%).
        sl_stop : float or None
            Stop loss as a fraction.

        """
        # Get system parameters
        profit_factor = getattr(self.system, 'profit_factor', None)
        stoploss_factor = getattr(self.system, 'stoploss_factor', None)

        # If no factors specified, return None
        if not profit_factor and not stoploss_factor:
            return None, None

        # Calculate average volatility across all symbols
        avg_vol = 0.0
        n_symbols = 0

        for symbol in self._symbols:
            if symbol in self._price_df.columns:
                close_prices = self._price_df[symbol].dropna()
                if len(close_prices) > 0:
                    vol = get_vol_ema(close_prices)
                    if len(vol) > 0:
                        avg_vol += vol.mean()
                        n_symbols += 1

        if n_symbols > 0:
            avg_vol = avg_vol / n_symbols
        else:
            avg_vol = 0.02  # Default 2% volatility

        logger.info("Average volatility for stops: %.4f", avg_vol)

        # Calculate stops as volatility multiples
        tp_stop = profit_factor * avg_vol if profit_factor else None
        sl_stop = stoploss_factor * avg_vol if stoploss_factor else None

        return tp_stop, sl_stop

    def run(self, tradelist):
        """
        Run the vectorbt backtest using the provided trade signals.

        Parameters
        ----------
        tradelist : list
            List of trade tuples from run_system().

        Returns
        -------
        self : VBTBacktester
            Returns self for method chaining.

        """
        logger.info("Running vectorbt backtest")

        # Build price data
        self._build_price_dataframe()

        # Convert signals
        entries, exits, short_entries, short_exits = signals_to_vectorbt(
            tradelist, self._symbols, self._price_df
        )

        # Map AlphaPy config to vectorbt parameters
        init_cash = self.portfolio_specs.get('capital', 100000)
        kelly_frac = self.portfolio_specs.get('kelly_frac', 0.1)
        cost_bps = self.portfolio_specs.get('cost_bps', 0.0)
        slippage = self.portfolio_specs.get('slippage', 0.0)
        direction = self.portfolio_specs.get('direction', 'both')

        # Convert cost_bps to decimal (basis points / 10000)
        fees = cost_bps / 10000.0

        # Calculate dynamic stops
        tp_stop, sl_stop = self._calculate_stops()

        # Determine direction
        direction_map = {
            'longonly': 'longonly',
            'shortonly': 'shortonly',
            'both': 'both'
        }
        vbt_direction = direction_map.get(direction, 'both')

        # Check if we have any short signals
        has_shorts = short_entries.sum().sum() > 0

        # Build the portfolio
        logger.info("Creating vectorbt portfolio with:")
        logger.info("  init_cash: %s", init_cash)
        logger.info("  kelly_frac: %s", kelly_frac)
        logger.info("  fees: %s", fees)
        logger.info("  slippage: %s", slippage)
        logger.info("  direction: %s", vbt_direction)
        logger.info("  tp_stop: %s", tp_stop)
        logger.info("  sl_stop: %s", sl_stop)
        logger.info("  has_shorts: %s", has_shorts)

        if has_shorts and vbt_direction == 'both':
            # When we have both long and short signals, use 'value' size_type
            # to avoid the position reversal limitation with 'percent'
            # Calculate position value based on initial cash and kelly fraction
            position_value = init_cash * kelly_frac

            logger.info("  size_type: value (position_value: %s)", position_value)

            self._portfolio = vbt.Portfolio.from_signals(
                close=self._price_df,
                entries=entries,
                exits=exits,
                short_entries=short_entries,
                short_exits=short_exits,
                init_cash=init_cash,
                size=position_value,
                size_type='value',
                fees=fees,
                slippage=slippage,
                tp_stop=tp_stop,
                sl_stop=sl_stop,
                freq='1D',
                accumulate=False,  # Don't accumulate positions
                upon_opposite_entry='close'  # Close existing position on opposite entry
            )
        else:
            # Long only - can use percent sizing
            logger.info("  size_type: percent")

            self._portfolio = vbt.Portfolio.from_signals(
                close=self._price_df,
                entries=entries,
                exits=exits,
                init_cash=init_cash,
                size=kelly_frac,
                size_type='percent',
                fees=fees,
                slippage=slippage,
                tp_stop=tp_stop,
                sl_stop=sl_stop,
                freq='1D'
            )

        logger.info("Backtest complete")
        return self

    def stats(self):
        """
        Get portfolio statistics.

        Returns
        -------
        stats : pandas.Series
            Portfolio statistics from vectorbt.

        """
        if self._portfolio is None:
            raise ValueError("Must run backtest first")

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._portfolio.stats()

    def to_returns_frame(self):
        """
        Get returns in AlphaPy format for compatibility.

        Returns
        -------
        returns_df : pandas.DataFrame
            Daily returns with 'date' index and 'return' column.

        """
        if self._portfolio is None:
            raise ValueError("Must run backtest first")

        returns = self._portfolio.returns()

        # If returns is multi-column, aggregate
        if isinstance(returns, pd.DataFrame):
            returns = returns.sum(axis=1)

        returns_df = pd.DataFrame({
            'return': returns.values
        }, index=returns.index)
        returns_df.index.name = 'date'

        return returns_df

    def to_positions_frame(self):
        """
        Get positions in AlphaPy format for compatibility.

        Returns
        -------
        positions_df : pandas.DataFrame
            Position values with 'date' index and symbol columns.

        """
        if self._portfolio is None:
            raise ValueError("Must run backtest first")

        # Get position values
        positions = self._portfolio.asset_value()

        # Add cash column
        cash = self._portfolio.cash()
        if isinstance(cash, pd.DataFrame):
            cash = cash.sum(axis=1)

        positions_df = positions.copy()
        positions_df['cash'] = cash

        positions_df.index.name = 'date'
        return positions_df

    def to_transactions_frame(self):
        """
        Get transactions in AlphaPy format for compatibility.

        Returns
        -------
        transactions_df : pandas.DataFrame
            Trade log with 'date' index and symbol, amount, price columns.

        """
        if self._portfolio is None:
            raise ValueError("Must run backtest first")

        # Get trade records
        trades = self._portfolio.trades.records_readable

        if trades.empty:
            return pd.DataFrame(columns=['date', 'symbol', 'amount', 'price'])

        # Format for AlphaPy compatibility
        transactions = []
        for _, trade in trades.iterrows():
            symbol = trade.get('Column', 'UNKNOWN')
            entry_date = trade.get('Entry Timestamp', None)
            exit_date = trade.get('Exit Timestamp', None)
            size = trade.get('Size', 0)
            entry_price = trade.get('Entry Price', 0)
            exit_price = trade.get('Exit Price', 0)

            # Entry transaction
            if entry_date is not None:
                transactions.append({
                    'date': entry_date,
                    'symbol': symbol,
                    'amount': size,
                    'price': entry_price
                })

            # Exit transaction
            if exit_date is not None:
                transactions.append({
                    'date': exit_date,
                    'symbol': symbol,
                    'amount': -size,
                    'price': exit_price
                })

        transactions_df = pd.DataFrame(transactions)
        if not transactions_df.empty:
            transactions_df = transactions_df.sort_values('date')
            transactions_df.set_index('date', inplace=True)

        return transactions_df

    def generate_tearsheet(self, output_path):
        """
        Generate HTML report using vectorbt.

        Parameters
        ----------
        output_path : str
            Path to save the HTML report.

        """
        if self._portfolio is None:
            raise ValueError("Must run backtest first")

        logger.info("Generating vectorbt tearsheet: %s", output_path)

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        try:
            # Get aggregated returns across all symbols
            returns = self._portfolio.returns()
            if isinstance(returns, pd.DataFrame):
                # Sum returns across all columns for total portfolio return
                returns = returns.sum(axis=1)

            # Calculate metrics
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.cummax()
            drawdown = (cum_returns - rolling_max) / rolling_max

            # Get portfolio value
            total_value = self._portfolio.value()
            if isinstance(total_value, pd.DataFrame):
                total_value = total_value.sum(axis=1)

            # Create comprehensive tearsheet
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=['Portfolio Value', 'Cumulative Returns', 'Drawdown'],
                vertical_spacing=0.08
            )

            # Portfolio value
            fig.add_trace(
                go.Scatter(x=total_value.index, y=total_value.values,
                          name='Portfolio Value', line=dict(color='blue')),
                row=1, col=1
            )

            # Cumulative returns
            fig.add_trace(
                go.Scatter(x=cum_returns.index, y=(cum_returns - 1) * 100,
                          name='Cumulative Return %', line=dict(color='green')),
                row=2, col=1
            )

            # Drawdown
            fig.add_trace(
                go.Scatter(x=drawdown.index, y=drawdown.values * 100,
                          name='Drawdown %', fill='tozeroy', line=dict(color='red')),
                row=3, col=1
            )

            # Get stats for title
            stats = self.stats()
            total_return = stats.get('Total Return [%]', 0)
            sharpe = stats.get('Sharpe Ratio', 0)
            max_dd = stats.get('Max Drawdown [%]', 0)

            fig.update_layout(
                height=900,
                title_text=f"Portfolio Performance | Return: {total_return:.2f}% | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2f}%",
                showlegend=True
            )
            fig.update_yaxes(title_text="Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Return (%)", row=2, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)

            fig.write_html(output_path)
            logger.info("Tearsheet generated successfully")

        except Exception as e:
            logger.error("Could not generate tearsheet: %s", e)
            # Write a simple HTML with stats instead
            stats = self.stats()
            html_content = f"<html><head><title>Portfolio Stats</title></head><body><h1>Portfolio Statistics</h1><pre>{stats.to_string()}</pre></body></html>"
            with open(output_path, 'w') as f:
                f.write(html_content)

        return output_path


#
# Function gen_vbt_portfolios
#

def gen_vbt_portfolios(model, system_name, portfolio_specs, group, bframe, pframe):
    """
    Generate portfolios using vectorbt backtesting.

    This is a drop-in replacement for gen_portfolios() in portfolio.py.

    Parameters
    ----------
    model : alphapy.Model
        The model specifications.
    system_name : str
        The name of the system.
    portfolio_specs : dict
        The portfolio specifications.
    group : alphapy.Group
        The group of instruments in the portfolio.
    bframe : pandas.DataFrame
        The baseline trade list from running the system.
    pframe : pandas.DataFrame
        The probability trade list from running the system.

    """
    logger.info("Generating Portfolios using vectorbt")

    # Get model specs
    run_dir = model.specs['run_dir']
    extension = model.specs['extension']
    separator = model.specs['separator']

    # Get system from registry
    from alphapy.system import System, SystemRank
    if system_name in System.systems:
        system = System.systems[system_name]
    elif system_name in SystemRank.systems:
        system = SystemRank.systems[system_name]
    else:
        raise ValueError(f"System {system_name} not found")

    # Create systems directory
    system_dir = SSEP.join([run_dir, 'systems'])

    # Process trade frames
    trade_dfs = []
    tags = []

    if not bframe.empty:
        trade_dfs.append(bframe)
        tags.append('base')

    if not pframe.empty:
        trade_dfs.append(pframe)
        tags.append('prob')

    gname = group.name
    gspace = group.space

    for tframe, tag in zip(trade_dfs, tags):
        logger.info("Processing %s trades", tag)

        # Convert DataFrame to tradelist format
        tradelist = []
        for date, row in tframe.iterrows():
            trade_data = [row['name'], row['order'], row['quantity'], row['price']]
            tradelist.append((date, trade_data))

        # Run vectorbt backtest
        backtester = VBTBacktester(model, system, group, portfolio_specs)
        backtester.run(tradelist)

        # Get and save returns
        logger.info("Recording Returns Frame")
        rspace = Space(system_name, 'returns', gspace.fractal)
        rf = backtester.to_returns_frame()
        rfname = frame_name(gname, rspace)
        write_frame(rf, system_dir, rfname, extension, separator, tag,
                    index=True, index_label='date')

        # Get and save positions
        logger.info("Recording Positions Frame")
        pspace = Space(system_name, 'positions', gspace.fractal)
        pf = backtester.to_positions_frame()
        pfname = frame_name(gname, pspace)
        write_frame(pf, system_dir, pfname, extension, separator, tag,
                    index=True, index_label='date')

        # Get and save transactions
        logger.info("Recording Transactions Frame")
        tspace = Space(system_name, 'transactions', gspace.fractal)
        tf = backtester.to_transactions_frame()
        tfname = frame_name(gname, tspace)
        write_frame(tf, system_dir, tfname, extension, separator, tag,
                    index=True, index_label='date')

        # Save full stats as JSON
        logger.info("Recording Trading Statistics")
        stats = backtester.stats()
        stats_dict = stats.to_dict()
        # Convert any non-serializable types
        for key, value in stats_dict.items():
            if isinstance(value, (np.integer, np.floating)):
                stats_dict[key] = float(value)
            elif isinstance(value, pd.Timestamp):
                stats_dict[key] = str(value)
            elif isinstance(value, pd.Timedelta):
                stats_dict[key] = str(value)

        stats_file_name = USEP.join([system_name, 'stats', tag]) + '.json'
        stats_spec = SSEP.join([system_dir, stats_file_name])
        with open(stats_spec, 'w') as f:
            json.dump(stats_dict, f, indent=2, default=str)

        # Generate vectorbt tearsheet
        logger.info("Generating Tear Sheet")
        ts_file_name = USEP.join(['vbt_tear_sheet', tag]) + '.html'
        tear_sheet_spec = SSEP.join([system_dir, ts_file_name])
        logger.info("Saving Tear Sheet to: %s", tear_sheet_spec)
        backtester.generate_tearsheet(tear_sheet_spec)

    return
