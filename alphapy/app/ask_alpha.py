"""
Package   : AlphaPy
Module    : ask_alpha
Created   : February 21, 2021

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
> streamlit run ask_alpha.py

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
# Initialize logger
#

def create_logger(name, level='DEBUG', file=None):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    # if no stream handler present, add one
    if sum([isinstance(handler, logging.StreamHandler) for handler in logger.handlers]) == 0:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d-%(name)s-%(levelname)s>>>%(message)s', "%H:%M:%S"))
        logger.addHandler(ch)
    # if a file handler is requested, check for existence then add
    if file is not None:
        if sum([isinstance(handler, logging.FileHandler) for handler in logger.handlers]) == 0:
            ch = logging.FileHandler(file, 'w')
            ch.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d-%(name)s-%(levelname)s>>>%(message)s', "%H:%M:%S"))
            logger.addHandler(ch)
    return logger

if 'logger' not in st.session_state:
    st.session_state['logger'] = create_logger(name='app', level='INFO', file='ask_alpha.log')
logger = st.session_state['logger']

#
# Application Configuration
#

def set_page_config():
    im = Image.open('logo.jpg')

    st.set_page_config(
        page_title="Scottfree Analytics",
        page_icon=im,
        layout="wide",
    )

    st.markdown("""
            <style>
                .block-container {
                        padding-top: 2rem;
                        padding-bottom: 2rem;
                        padding-left: 2rem;
                        padding-right: 2rem;
                    }
            </style>
            """, unsafe_allow_html=True)

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    return True

if 'page_config' not in st.session_state:
    st.session_state.page_config = set_page_config()

#
# Function to display the status message
#

def display_status_message(col, message, message_type='info'):
    if message_type == 'info':
        col.info(message)
    elif message_type == 'success':
        col.success(message)
    elif message_type == 'warning':
        col.warning(message)
    elif message_type == 'error':
        col.error(message)

#
# Function to handle dismissal
#

def dismiss_message():
    st.session_state.dismissed = True

if 'dismissed' not in st.session_state:
    st.session_state.dismissed = False

#
# Get the AlphaPy environment variables
#

alphapy_root = os.environ.get('ALPHAPY_ROOT')
if not alphapy_root:
    root_error_string = "ALPHAPY_ROOT environment variable must be set"
    logger.info(root_error_string)
    sys.exit(root_error_string)
else:
    # Read the AlphaPy configuration file
    alphapy_specs = get_alphapy_config(alphapy_root)

# Ask Alpha Options

st.sidebar.title(':red[α]sk :red[α]lph:red[α]')

market_string = "Markets 📈 💵 🐂 🐻 🏙 💱"
sports_string = "Sports 🏀 ⚾ 🏈 ⚽ 🏒 🎾"
topic = st.sidebar.radio(
    "Select a topic",
    [market_string, sports_string],
    horizontal=True,
    label_visibility="hidden")

st.sidebar.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)

# OpenAI API Key

col1, col2 = st.columns((2, 1))

# Function to validate the API key format

def is_valid_api_key(key):
    return key.startswith('sk-') and len(key) == 51

# Retrieve the API key from Streamlit secrets or environment variables
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    # Ask the user to input their API key if not already provided
    api_key = col1.text_input('Enter OpenAI API token:', type='password')
    
    # Validate the entered API key
    if not api_key:
        st.warning('Please enter your OpenAI API key!', icon='⚠️')
    elif not is_valid_api_key(api_key):
        st.warning('Invalid API key format. Please check and enter again.', icon='⚠️')
    else:
        st.success('API key looks good! Proceed to entering your prompt message.', icon='✅')
else:
    # Display the message if it hasn't been dismissed
    if not st.session_state.dismissed:
        display_status_message(col1, "OpenAI API key has been provided ✅. &emsp; Click the Dismiss button to the right. &emsp; :arrow_right:", "info")
        col2.button("Dismiss", on_click=dismiss_message)

# Set the OpenAI API key for the client
openai.api_key = api_key

# Markets

if topic == market_string:
    col201, col202 = st.columns((1, 3))
    common_queries = ["Latest Prices"]
    market_queries = col201.selectbox('Get', common_queries)
    prompt_text = col202.text_input('Ask AI')

# Sports

if topic == sports_string:
    col201, col202 = st.columns((1, 3))
    common_queries = ["Latest Lines"]
    sports_queries = col201.selectbox('Get', common_queries)
    prompt_text = col202.text_input('Ask AI')
    # Set the number of columns
    col21, col22, col23, col24, col25, col26 = st.columns(6)
    # Define the options for the league dropdown
    options = ["NFL", "NBA", "MLB", "NHL", "NCAAF", "NCAAB"]
    # Create a select dropdown with the options
    league_selected = col21.selectbox('Select League', options)
    # Define the options for the model dropdown
    options = ["Spread", "Moneyline", "Over/Under"]
    # Create a select dropdown with the options
    model_selected = col22.selectbox('Select Model', options)


# Generative AI

def test_openai_api_key(prompt):
    try:
        # Make a simple API request to the OpenAI API
        response = openai.chat.completions.create(model="gpt-4o",
            messages=[
                    {"role": "system", "content": "Hello"},
                    {"role": "user", "content": prompt},
                ])
        # Print the response
        response_content = response.choices[0].message.content
        st.write(response_content)
    except openai.AuthenticationError:
        print("Invalid API key. Please check your API key and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the test function

if prompt_text:
    test_openai_api_key(prompt_text)

# Function to fetch data from FastAPI server
def fetch_data():
    try:
        api_url = "http://0.0.0.0:8080/data"
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return {}

# Add a button to fetch data
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
