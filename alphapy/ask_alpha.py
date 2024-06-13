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
from simpleaichat import AIChat
import streamlit as st
from streamlit_extras.app_logo import add_logo
import sys

from alphapy.alphapy_main import get_alphapy_config
import alphapy.globals as apg


#
# Global Variables
#

dir_assets = './assets/'


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Main Program
#


# Initialize Logging

if "logging" not in st.session_state:
    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="streamlit_main.log", filemode='a', level=logging.INFO,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    logger.info('*'*80)
    logger.info("Streamlit Start")
    logger.info('*'*80)
    # Set the logging state variable
    st.session_state["logging"] = True

# Start Streamlit

#
# Application Configuration
#

path_logo = os.path.join(dir_assets, 'logo.jpg')
im = Image.open(path_logo)

st.set_page_config(
    page_title="Scottfree Analytics",
    page_icon=im,
    layout="wide",
)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
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

# Function to display the status message
def display_status_message(col, message, message_type='info'):
    if message_type == 'info':
        col.info(message)
    elif message_type == 'success':
        col.success(message)
    elif message_type == 'warning':
        col.warning(message)
    elif message_type == 'error':
        col.error(message)

# Function to handle dismissal
def dismiss_message():
    st.session_state.dismissed = True

# Initialize the dismissed state if it doesn't exist
if 'dismissed' not in st.session_state:
    st.session_state.dismissed = False

# Get the AlphaPy environment variable

alphapy_root = os.environ.get('ALPHAPY_ROOT')
if not alphapy_root:
    root_error_string = "ALPHAPY_ROOT environment variable must be set"
    logger.info(root_error_string)
    sys.exit(root_error_string)
else:
    # Read the AlphaPy configuration file
    alphapy_specs = get_alphapy_config(alphapy_root)

col1, col2, col3 = st.columns((2, 3, 2))

# Ask Alpha Options
col1.header(':red[α]sk :red[α]lph:red[α]')

market_string = "Markets 📈 💵 🐂 🐻 🏙 💱"
sports_string = "Sports 🏀 ⚾ 🏈 ⚽ 🏒 🎾"
topic = col2.radio(
    "Select a topic",
    [market_string, sports_string],
    horizontal=True,
    label_visibility="hidden")

# OpenAI API Key

col1, col2, col3 = st.columns((3, 2, 2))

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
        display_status_message(col1, "OpenAI API key has been provided ✅. Click the button to dismiss.", "info")
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

client = OpenAI()

def call_openai(prompt):
    try:
        response = client.completions.create(model='gpt-4',
        messages=[
                {"role": "system", "content": "Hello"},
                {"role": "user", "content": "When was GPT launched?"},
            ])
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"Error: {e}")

if prompt_text:
    if api_key and is_valid_api_key(api_key):
        response = call_openai(prompt_text)
        st.write(response)
    else:
        st.warning('Please provide a valid OpenAI API key.', icon='⚠️')
