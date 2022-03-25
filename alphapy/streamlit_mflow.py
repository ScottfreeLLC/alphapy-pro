################################################################################
#
# Package   : AlphaPy
# Module    : streamlit_mflow
# Created   : February 21, 2021
#
# streamlit run streamlit.py
#
# Copyright 2021 ScottFree Analytics LLC
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

from finviz.screener import Screener
import pandas as pd
import streamlit as st


def run_finviz_screener(screener):
    st.write(screener)

    filters = ['exch_nasd', 'idx_sp500']  # Shows companies in NASDAQ which are in the S&P500
    stock_list = Screener(filters=filters, table='Performance', order='price')  # Get the performance table and sort it by price ascending

    stock_df = pd.DataFrame(stock_list.data)
    st.write(stock_df)


def run_finviz_portfolio(screener):
    st.write(screener)


def run_index(screener):
    st.write(screener)
    url = f"https://docs.google.com/spreadsheets/d/1Syr2eLielHWsorxkDEZXyc55d6bNx1M3ZeI4vdn7Qzo/export?format=csv"
    df = pd.read_csv(url)
    df.loc[df['symbol'] == '^NDX', 'name'] = 'Nasdaq 100'
    st.write(df)


def run_stocks(market_type):
    st.subheader(market_type)

    symbol = st.text_input('Ticker Symbol', 'AAPL')

    screener = st.sidebar.radio("Group", ('Finviz Screener', 'Finviz Portfolio', 'Index'))

    if screener == 'Finviz Screener':
        run_finviz_screener(screener)
    elif screener == 'Finviz Portfolio':
        run_finviz_portfolio(screener)
    elif screener == 'Index':
        run_index(screener)


def run_crypto(market_type):
    st.subheader(market_type)


def run_futures(market_type):
    st.subheader(market_type)


def run_forex(market_type):
    st.subheader(market_type)


def app():
    market_type = st.sidebar.radio("Market Type", ('Stocks', 'Crypto', 'Futures', 'Forex'))

    if market_type == 'Stocks':
        run_stocks(market_type)
    elif market_type == 'Crypto':
        run_crypto(market_type)
    elif market_type == 'Futures':
        run_futures(market_type)
    elif market_type == 'Forex':
        run_forex(market_type)
