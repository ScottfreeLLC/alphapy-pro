"""
Package   : AlphaPy
Module    : ask_alpha_pages
Created   : August 9, 2024

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

"""


#
# Imports
#

from datetime import datetime, timedelta
import logging
import openai
from openai import OpenAI
import os
import pandas as pd
from PIL import Image
import requests
from simpleaichat import AIChat
import streamlit as st
from streamlit_extras.app_logo import add_logo
import sys

from alphapy.alphapy_main import get_alphapy_config
import alphapy.globals as apg

#
# Function to fetch data from FastAPI server
#

def fetch_data():
    try:
        api_url = "http://0.0.0.0:8080/data"
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return {}
    return

# Function

def get_market_ask_alpha():
    return

# Function

def get_market_portfolio():
    if st.button('Get Stock Data'):
        stock_data = fetch_data()
        if stock_data:
            df = pd.DataFrame.from_dict(stock_data, orient='index')
            cols_df = ['close', 'pchg', 'vratio', 'vwapd', 'h20', 'l20',
                    'fastk', 'slowd', 'sequp', 'seqdown', 'hv', 'squeeze']
            df = df[cols_df]
            st.dataframe(df)
        else:
            st.warning("No data available.")
    else:
        st.info("Click the button to fetch stock data.")
        return


# Function

def get_market_systems():
    return

# Function

def get_market_patterns():
    return

# Function

def get_market_screener():
    return


# Function

def get_sports_ask_alpha():
    return

# Function

def get_sports_lines():
    return

# Function

def get_sports_results():
    return

# Function

def get_sports_predictions():
    return

# Function

def get_sports_summary():
    return

# Function

def get_sports_systems():
    return
