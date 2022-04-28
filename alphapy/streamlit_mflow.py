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


#
# Imports
#

from finviz.portfolio import Portfolio
from finviz.screener import Screener
from itsdangerous import json
import finnhub
import logging
import os
import pandas as pd
import requests
import streamlit as st
import sys

from alphapy_main import get_alphapy_config
from streamlit_util import alphapy_request


#
# Initialize logger
#

logger = logging.getLogger(__name__)


def run_alphapy_groups(screener, groups):
    st.write(screener)
    group = st.selectbox('Select Group', groups)
    with st.expander("View Symbols"):
        df = pd.DataFrame(groups[group]['members'])
        df.columns = ['symbol']
        df['symbol'] = df['symbol'].str.upper()
        df.sort_values(by=['symbol'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        st.write(df)
    return df


def run_finviz_screener(screener):
    st.write(screener)

    filters = ['exch_nasd', 'idx_sp500']  # Shows companies in NASDAQ which are in the S&P500
    stock_list = Screener(filters=filters, table='Performance', order='price')  # Get the performance table and sort it by price ascending

    stock_df = pd.DataFrame(stock_list.data)
    st.write(stock_df)

    return stock_df


def run_finviz_portfolio(screener):
    port_name = 'Test'
    st.write(screener)
    portfolio = Portfolio('scottfree.analytics@scottfreellc.com', 'TS7$@@6dU9Nad@i', port_name)
    if portfolio:
        df = pd.DataFrame(portfolio.data)
        st.write(df)
    else:
        error_message = f"Could not find FinViz Portfolio: {port_name}"
        st.text(error_message)
    return df


def run_market_index(screener):
    st.write(screener)
    url = f"https://docs.google.com/spreadsheets/d/1Syr2eLielHWsorxkDEZXyc55d6bNx1M3ZeI4vdn7Qzo/export?format=csv"
    df = pd.read_csv(url)
    df.loc[df['symbol'] == '^NDX', 'name'] = 'Nasdaq 100'
    st.write(df)
    return df


def app():
 
    # Get the AlphaPy environment variable

    alphapy_root = os.environ.get('ALPHAPY_ROOT')
    if not alphapy_root:
        root_error_string = "ALPHAPY_ROOT environment variable must be set"
        logger.info(root_error_string)
        sys.exit(root_error_string)
    else:
        # Read the AlphaPy configuration file
        alphapy_specs = get_alphapy_config(alphapy_root)

    text_ap = 'Market Flow'
    text_fs = 'Finviz Screener'
    text_fp = 'Finviz Portfolio'
    text_mi = 'Market Index'
    screener = st.sidebar.radio("Group", (text_ap, text_fs, text_fp, text_mi))

    st.header("Market Flow")

    server_url = alphapy_specs['mflow']['server_url']
    if screener == text_ap:
        groups = alphapy_request(server_url, 'groups')
        df = run_alphapy_groups(screener, groups)
    elif screener == text_fs:
        df = run_finviz_screener(screener)
    elif screener == text_fp:
        df = run_finviz_portfolio(screener)
    elif screener == text_mi:
        df = run_market_index(screener)

    systems = alphapy_request(server_url, 'systems')
    system = st.selectbox('Select System', systems)

    with st.expander("View Signals"):
        df = pd.DataFrame(systems[system].items(), columns=['signal', 'value'])
        df.reset_index(drop=True, inplace=True)
        st.write(df)

