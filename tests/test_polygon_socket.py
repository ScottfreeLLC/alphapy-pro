"""
Package   : AlphaPy
Module    : mflow_screener
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

python mflow_screener.py

"""

from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage, EquityTrade
from typing import List
from typing import Dict
from datetime import datetime
import time
import threading
import os

# docs
# https://polygon.io/docs/stocks/ws_stocks_am
# https://polygon-api-client.readthedocs.io/en/latest/WebSocket.html#

# This program connects to the Polygon WebSocket API, authenticates the
# connection, and subscribes to receive trades. Every 5 seconds, it counts
# the number of trades per symbol and stores the results in a map. The
# program then prints the map, which gives a readout of the top stocks
# traded in the past 5 seconds.

# Here's what the output looks like after running it for a couple hours:

"""
  --- Past 5 seconds ---
   Tickers seen (5s): 1697
    Trades seen (5s): 12335
    Cash traded (5s): 88,849,414.33

  --- Running Totals ---
  Total Tickers seen: 13848
   Total Trades seen: 22775838
   Total Cash traded: 178,499,702,488.96

----------------------------------------------------------------------------------------------------

Ticker         Trades (5s)         Cash (5s)           Total Trades        Total Cash
NVDA           445                 6,933,283.61        632550              18,291,747,596.36
TSLA           279                 8,144,556.76        639585              11,319,594,268.07
NVDL           277                 3,748,806.85        10451               99,902,192.88
TELL           171                 78,424.03           6154                3,710,200.38
AFRM           163                 968,984.99          224338              745,895,134.93
AAPL           134                 2,359,278.02        304572              2,932,389,741.58
QQQ            132                 5,788,859.71        246679              11,003,577,730.48
NVDS           130                 598,047.04          7846                48,854,967.44
SOXL           127                 786,026.38          189184              719,639,349.26
AMD            116                 1,549,180.08        304704              3,713,351,432.39
SPY            113                 6,628,554.14        278926              15,435,607,506.98
MSFT           109                 1,600,861.75        148047              2,396,824,971.18
SQQQ           88                  1,006,330.83        173406              2,065,760,858.90
TQQQ           83                  717,574.40          296021              2,580,097,288.27
PLUG           82                  106,542.65          31921               53,825,007.27
ITB            75                  455,902.33          23369               185,892,273.60
AVGO           71                  1,955,826.79        31586               633,629,812.65
STX            71                  273,681.77          8420                34,141,139.17
XPEV           68                  234,765.41          41284               127,781,104.54
OIH            55                  662.12              2964                65,848,514.45
XEL            54                  197,642.42          18524               103,054,857.37
XLU            53                  850,017.20          35963               291,891,266.17
ARRY           52                  164,056.54          11354               23,001,537.49
META           52                  1,457,535.82        150793              2,717,344,906.63
PLTR           52                  147,743.93          86456               396,851,801.06

Current Time: 2023-08-25 08:27:14.602075 | App Uptime: 04:49:40 | Time taken: 0.003417 seconds
"""

app_start_time = time.time()
string_map: Dict[str, int] = {}
cash_map_5s: Dict[str, float] = {}
cash_traded = float(0)

# totals
total_tickers_seen = 0
total_trades_seen = 0
total_cash_traded = 0.0

# These dictionaries will keep track of the running total of trades and cash per ticker.
total_string_map: Dict[str, int] = {}
total_cash_map: Dict[str, float] = {}

"""
def handle_msg(msgs: List[WebSocketMessage]):
    global cash_traded
    global total_tickers_seen, total_trades_seen, total_cash_traded
    print('here')
    for m in msgs:
        if isinstance(m, EquityTrade):
            # Update total trades and cash for the past 5 seconds
            if isinstance(m.symbol, str):
                string_map[m.symbol] = string_map.get(m.symbol, 0) + 1
                total_string_map[m.symbol] = total_string_map.get(m.symbol, 0) + 1

            # Update cash traded
            if isinstance(m.price, float) and isinstance(m.size, int):
                cash_value = m.price * m.size
                cash_traded += cash_value
                total_cash_map[m.symbol] = (  # type: ignore
                    total_cash_map.get(m.symbol, 0) + cash_value  # type: ignore
                )

                # Update cash for the past 5 seconds
                cash_map_5s[m.symbol] = (  # type: ignore
                    cash_map_5s.get(m.symbol, 0) + cash_value  # type: ignore
                )  # Okay!

                # Update totals
                total_tickers_seen = len(total_string_map)
                total_trades_seen += 1
                total_cash_traded += cash_value

# client = WebSocketClient("_BynHqDfXhPoQcFf8Nb6hJzC_p67_5Sf1tn5ms") # hardcoded api_key is used
client = WebSocketClient()  # POLYGON_API_KEY environment variable is used
client.subscribe("T.*")  # all trades
client.run(handle_msg)
"""


ws = WebSocketClient(api_key="_BynHqDfXhPoQcFf8Nb6hJzC_p67_5Sf1tn5ms", subscriptions=["T.*"])

def handle_msg(msg: List[WebSocketMessage]):
    for m in msg:
        print(m)

ws.run(handle_msg=handle_msg)
