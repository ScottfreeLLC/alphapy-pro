################################################################################
#
# Package   : AlphaPy
# Module    : streamlit_finviz
# Created   : February 21, 2021
#
# Copyright 2022 ScottFree Analytics LLC
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

from datetime import datetime, timedelta
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
from alphapy_requests import alphapy_request


#
# Initialize logger
#

logger = logging.getLogger(__name__)


def get_finviz_screener_groups():

    #filters = ['sh_price_u5','ta_gap_d5','ta_rsi_os30','ft=3']
    #stocks = Screener(filters=filters, order="price")

    filters = ['exch_nasd', 'idx_sp500']  # Shows companies in NASDAQ which are in the S&P500
    stock_list = Screener(filters=filters, table='Performance', order='price')  # Get the performance table and sort it by price ascending

    stock_df = pd.DataFrame(stock_list.data)
    st.write(stock_df)

 
# Get the AlphaPy environment variable

alphapy_root = os.environ.get('ALPHAPY_ROOT')
if not alphapy_root:
    root_error_string = "ALPHAPY_ROOT environment variable must be set"
    logger.info(root_error_string)
    sys.exit(root_error_string)
else:
    # Read the AlphaPy configuration file
    alphapy_specs = get_alphapy_config(alphapy_root)

projects = alphapy_request(alphapy_specs, 'projects')
project = st.sidebar.selectbox("Select Project", sorted(projects, key=str.casefold))
