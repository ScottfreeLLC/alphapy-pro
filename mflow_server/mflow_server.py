"""
Package   : AlphaPy
Module    : mflow_server
Created   : June 2, 2024

Copyright 2024 ScottFree Analytics LLC
Mark Conway & Robert D. Scott II

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

HOW TO RUN:

> export ALPHAPY_ROOT=/Users/markconway/Projects/alphapy-root
> cd /Users/markconway/Projects/alphapy-pro/alphapy
> python mflow_server.py

SERVER CHECK
> lsof -i :8080

"""


import asyncio
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from finviz.portfolio import Portfolio
from finviz.screener import Screener
import finnhub
import json
import logging
import os
import pandas as pd
import pandas_ta as ta
import requests
import sys
import uvicorn
import websockets
import yaml


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Global Variables
#

BASE_URL = 'https://api.polygon.io'
API_KEY = os.getenv('POLYGON_API_KEY')
stock_data = {}
config_mflow = None
config_groups = None
config_sources = None


#
# Get the Market Flow Configurations
#

def get_config_files():
    global config_mflow, config_groups, config_sources
    with open('./mflow_server.yml', 'r') as file:
        config_mflow = yaml.safe_load(file)
    alphapy_root = os.getenv('ALPHAPY_ROOT') + '/config'
    path_groups = alphapy_root + '/groups.yml'
    with open(path_groups, 'r') as file:
        config_groups = yaml.safe_load(file)
    path_sources = alphapy_root + '/sources.yml'
    with open(path_sources, 'r') as file:
        config_sources = yaml.safe_load(file)


#
# Function to get all stock symbols
#

def get_all_stock_symbols(api_key):
    logger.info("Getting All Stock Symbols")
    url = f'{BASE_URL}/v3/reference/tickers'
    tickers = []
    params = {
        'apiKey': api_key,
        'limit': 1000,  # Adjust the limit to fetch more or fewer symbols per request
        'market': 'stocks',
        'active': 'true',  # Fetch only active stocks
        'type': 'CS'  # Fetch only common stocks
    }
    while True:
        response = requests.get(url, params=params)
        data = response.json()
        if response.status_code != 200:
            logger.error(f"Error fetching data: {data.get('error', 'Unknown error')}")
            break
        # Extract tickers and add them to the list
        tickers.extend([symbol['ticker'] for symbol in data['results']])
        # Check if there's a next_url for pagination
        if 'next_url' in data:
            url = data['next_url']
        else:
            break
    logger.info(f"Found {len(tickers)} Active Symbols")
    return tickers


#
# Function get_finviz_symbols
#

def get_finviz_symbols(portfolio_name):
    finviz_specs = config_sources['finviz']
    email = finviz_specs['email']
    api_key = finviz_specs['api_key']
    try:
        portfolio = Portfolio(email, api_key, portfolio_name)
        df = pd.DataFrame(portfolio.data)
        symbols = df['Ticker'].tolist()
    except:
        error_message = f"Could not find FinViz Portfolio: {portfolio_name}"
        logger.error(error_message)
        symbols = []
    return symbols


#
# Function get_finnhub_symbols
#

def get_finnhub_symbols(group_symbol):
    """
    Fetches symbols for the specified market index group from Finnhub.

    Parameters:
    - group_symbol: str, the symbol of the group (e.g., '^NDX').

    Returns:
    - list of str, symbols for the specified group.
    """

    # URL to the Google Sheet containing the group information
    url = "https://docs.google.com/spreadsheets/d/1Syr2eLielHWsorxkDEZXyc55d6bNx1M3ZeI4vdn7Qzo/export?format=csv"
    
    # Read the CSV data into a DataFrame
    df = pd.read_csv(url)
    
    # Find the group name for the specified group symbol
    group_row = df[df['symbol'] == group_symbol]
    if group_row.empty:
        logger.error(f"No group found for symbol {group_symbol}")
        return []

    # Get the constituent symbols
    finnhub_client = finnhub.Client(api_key=config_sources['finnhub']['api_key'])
    group_data = finnhub_client.indices_const(symbol=group_symbol)
    symbols = group_data.get('constituents', [])
    
    if not symbols:
        logger.info(f"No symbols found for group {group_symbol}")
    else:
        logger.info(f"Found {len(symbols)} symbols for group {group_symbol}")
    return symbols


#
# Function to fetch historical data for a single ticker
#

def fetch_historical_data(ticker, start_date, end_date, api_key):
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        'apiKey': api_key,
        'adjusted': 'true'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json().get('results', [])
        # Convert to DataFrame
        df = pd.DataFrame(data)
        if not df.empty:
            # Rename columns to lower case
            df.columns = [col.lower() for col in df.columns]
            # Convert 't' column to datetime and set as index
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('date', inplace=True)
            df.drop(columns=['t'], inplace=True)
        return df
    else:
        return pd.DataFrame()


#
# Function to update a stock's snapshot from Polygon.io
#

def update_snapshot(ticker, api_key):
    global stock_data
    # Call the API
    url = f"{BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
    params = {'apiKey': api_key}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        ticker_info = response.json().get('ticker', {})
        day_info = ticker_info.get('day', {})
        std = stock_data[ticker]
        std['close'] = day_info.get('c', 0)
        std['vw'] = day_info.get('vw', 0)
        std['vw'] = round(std['vw'], 2)
        std['vratio'] = day_info.get('v', 0) / std['avol']
        std['vratio'] = round(std['vratio'], 2)
        std['h20'] = std['close'] > std['hh']
        std['l20'] = std['close'] < std['ll']
        std['vwapd'] = std['vw'] - std['vwap']
        std['pchg'] = ticker_info.get('todaysChangePerc', 0)
        std['pchg'] = round(std['pchg'], 2)
        logger.debug(f"TICKER: {ticker}")
        logger.debug(stock_data[ticker])
        # update statistics
        pass
    else:
        logger.error(f"Failed to fetch snapshot for {ticker}")
    return


#
# Asynchronous function to handle WebSocket messages
#

async def handle_websocket(uri, api_key):
    global stock_data
    try:
        async with websockets.connect(uri) as websocket:
            # Authenticate
            await websocket.send(json.dumps({
                'action': 'auth',
                'params': api_key
            }))
            logger.info("Sent authentication request to WebSocket.")
            
            # Subscribe to all trades
            await websocket.send(json.dumps({
                'action': 'subscribe',
                'params': 'T.*'
            }))
            logger.info("Subscribed to trade messages.")
            
            # Process incoming messages
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    for trade in data:
                        if trade['ev'] == 'T':
                            ticker = trade['sym']
                            if ticker in stock_data:
                                update_snapshot(ticker, api_key)
                                logger.debug(f"Symbol: {ticker} Data: {stock_data[ticker]}")
                except websockets.ConnectionClosedOK:
                    logger.info("WebSocket connection closed normally")
                    break
                except websockets.ConnectionClosedError as e:
                    logger.error(f"WebSocket connection closed with error: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error in WebSocket handling: {e}")
                    break
    except Exception as e:
        logger.error(f"WebSocket connection failed: {e}")


#
# Function start_websocket_client
#

async def start_websocket_client(api_key):
    uri = "wss://socket.polygon.io/stocks"
    while True:
        try:
            await handle_websocket(uri, api_key)
        except websockets.ConnectionClosedOK:
            logger.info("WebSocket closed, reconnecting...")
            await asyncio.sleep(5)
        except websockets.ConnectionClosedError as e:
            logger.error(f"WebSocket error: {e}, retrying...")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Unexpected error: {e}, retrying in 10 seconds...")
            await asyncio.sleep(10)


#
# FastAPI Application
#

app = FastAPI()


#
# FastAPI Startup Event
#

@app.on_event("startup")
async def startup_event():
    global stock_data, config_mflow, config_groups, config_sources
    # Initialize Logging
    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="mflow_server.log", filemode='a', level=logging.INFO,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    # Start the pipeline
    logger.info('*'*80)
    logger.info("Market Flow Server Start")
    logger.info('*'*80)
    # Get the AlphaPy environment variable
    alphapy_root = os.environ.get('ALPHAPY_ROOT')
    if not alphapy_root:
        root_error_string = "ALPHAPY_ROOT environment variable must be set"
        logger.error(root_error_string)
        sys.exit(root_error_string)
    # Load the configuration files
    get_config_files()
    # Load the symbols
    symbols = []
    group_source = config_mflow['portfolio']['source']
    if group_source == 'alphapy':
        symbols = config_groups[config_mflow['portfolio']['name']]
    elif group_source == 'finnhub':
        symbols = get_finnhub_symbols(config_mflow['portfolio']['name'])
    elif group_source == 'finviz':
        symbols = get_finviz_symbols(config_mflow['portfolio']['name'])
    elif group_source == 'polygon':
        symbols = get_all_stock_symbols(API_KEY)
    else:
        logger.error(f'Unknown Group Source: {group_source}')
    logger.info(f"Source {group_source}: {len(symbols)}")
    # Fetch and store average volumes
    n_days = config_mflow['data']['history']
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=n_days)).strftime('%Y-%m-%d')
    previous_first_letter = None
    for sym in symbols:
        symbol = sym.upper()
        current_first_letter = symbol[0].upper()
        if current_first_letter != previous_first_letter:
            logger.info(f"Getting Historical Data for {current_first_letter} Symbols")
            previous_first_letter = current_first_letter
        # Get data for this ticker symbol
        df = fetch_historical_data(symbol, start_date, end_date, API_KEY)
        if not df.empty:
            df.reset_index(inplace=True)
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df.rename(columns={
                'o'  : 'open',
                'h'  : 'high',
                'l'  : 'low',
                'c'  : 'close',
                'v'  : 'volume',
                'vw' : 'vwap',
                'n'  : 'ntrades'
                }, inplace=True)
            df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'ntrades']]
            # VWAP
            df['vwap'] = df['vwap'].round(2)
            # Historical Volatility Ratio
            atr_length1 = 10
            df['atr1'] = ta.atr(df['high'], df['low'], df['close'], length=atr_length1)
            atr_length = 66
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_length)
            df['hv'] = df['atr1'] / df['atr']
            df['hv'] = df['hv'].round(2)
            # Average Volume
            volume_length = 30
            df['avol'] = ta.sma(df['volume'], length=volume_length).fillna(0).astype(int)
            # N-Day Highs and Lows
            window_length = 20
            df['hh'] = df['high'].rolling(window=window_length).max()
            df['ll'] = df['low'].rolling(window=window_length).min()
            # Stochastics
            fastk = 20
            slowd = 3
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=fastk, d=slowd)
            stoch.columns = ['fastk', 'slowd']
            stoch['fastk'] = stoch['fastk'].fillna(0).round(1)
            stoch['slowd'] = stoch['slowd'].fillna(0).round(1)
            df = pd.concat([df, stoch], axis=1)
            # Squeeze Indicator
            bblength = 20
            bbstd = 2.0
            bollinger_bands = ta.bbands(df['close'], length=bblength, std=bbstd)
            col_bbu = '_'.join(['BBU', str(bblength), str(bbstd)])
            col_bbm = '_'.join(['BBM', str(bblength), str(bbstd)])
            col_bbl = '_'.join(['BBL', str(bblength), str(bbstd)])
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = bollinger_bands[col_bbu], bollinger_bands[col_bbm], bollinger_bands[col_bbl]
            kclength = 20
            kcstd = 1.5
            keltner_channels = ta.kc(df['high'], df['low'], df['close'], length=kclength, scalar=kcstd)
            col_kcu = '_'.join(['KCUe', str(kclength), str(kcstd)])
            col_kcb = '_'.join(['KCBe', str(kclength), str(kcstd)])
            col_kcl = '_'.join(['KCLe', str(kclength), str(kcstd)])
            df['kc_upper'], df['kc_middle'], df['kc_lower'] = keltner_channels[col_kcu], keltner_channels[col_kcb], keltner_channels[col_kcl]
            df['squeeze'] = ((df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper']))
            # TD Sequential
            td_seq_values = ta.td_seq(df['close'], asint=True)
            td_seq_values.columns = ['sequp', 'seqdown']
            df = pd.concat([df, td_seq_values], axis=1)
            # Drop columns
            cols_drop = ['open', 'high', 'low', 'close', 'volume', 'atr1',
                         'bb_upper', 'bb_middle', 'bb_lower',
                         'kc_upper', 'kc_middle', 'kc_lower']
            df.drop(columns=cols_drop, inplace=True)
            # Add the last row to the dictionary
            last_row_dict = df.tail(1).to_dict(orient='records')[0]
            stock_data[symbol] = last_row_dict
    logger.info(f"Processed {len(stock_data)} Symbols")
    # Start the Web socket
    if config_mflow['data']['live']:
        logger.info("Server Mode: Live Data")
        websocket_task = asyncio.create_task(start_websocket_client(API_KEY))
        await websocket_task
    else:
        logger.info("Server Mode: Historical")


#
# FastAPI / Route
#

@app.get("/")
def get_default_status():
    return "Market Flow Server"


#
# FastAPI /data Route
#

@app.get("/data")
def get_data():
    return stock_data


#
# FastAPI Shutdown Event
#

@app.on_event("shutdown")
def shutdown_event():
    # Stop the pipeline
    logger.info('*'*80)
    logger.info("Market Flow Server End")
    logger.info('*'*80)


#
# Main Program
#

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
