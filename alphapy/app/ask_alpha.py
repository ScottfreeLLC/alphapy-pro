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
from ask_alpha_pages import get_market_ask_alpha
from ask_alpha_pages import get_market_portfolio
from ask_alpha_pages import get_market_systems
from ask_alpha_pages import get_market_patterns
from ask_alpha_pages import get_market_screener
from ask_alpha_pages import get_sports_ask_alpha
from ask_alpha_pages import get_sports_lines
from ask_alpha_pages import get_sports_results
from ask_alpha_pages import get_sports_predictions
from ask_alpha_pages import get_sports_summary
from ask_alpha_pages import get_sports_systems


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

def load_image():
    im = Image.open('logo.jpg')
    return im

def set_page_config():
    st.set_page_config(
        page_title="Scottfree Analytics",
        page_icon=st.session_state.load_image,
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

if 'load_image' not in st.session_state:
    st.session_state.load_image = load_image()
set_page_config()

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

st.sidebar.title(':red[α]sk :red[α]lph:red[α]')

market_string = "Markets 📈 💵 🐂 🐻 🏙 💱"
sports_string = "Sports 🏀 ⚾ 🏈 ⚽ 🏒 🎾"
topic = st.sidebar.radio(
    "Select a topic",
    [market_string, sports_string],
    horizontal=True,
    label_visibility="hidden")

st.sidebar.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)

#
# Market and Sports Options
#

market_option = None
sports_option = None

if topic == market_string:
    market_option = st.sidebar.radio(
        "Choose Market Option",
        ["Ask Alpha", "Portfolio", "Systems", "Patterns", "Screener"],
    )
    if market_option == "Ask Alpha":
        get_market_ask_alpha()
    elif market_option == "Portfolio":
        get_market_portfolio()
    elif market_option == "Systems":
        get_market_systems()
    elif market_option == "Patterns":
        get_market_patterns()
    elif market_option == "Screener":
        get_market_screener()
    
elif topic == sports_string:
    league_options = ["MLB", "NBA", "NCAAB", "NCAAF", "NFL", "NHL"]
    league_selected = st.sidebar.selectbox('Select League', league_options)
    sports_option = st.sidebar.radio(
        league_selected,
        ["Ask Alpha", "Lines", "Results", "Predictions", "Summary", "Systems"],
    )
    if sports_option == "Ask Alpha":
        get_sports_ask_alpha()
    elif sports_option == "Lines":
        get_sports_lines()
    elif sports_option == "Results":
        get_sports_results()
    elif sports_option == "Predictions":
        get_sports_predictions()
    elif sports_option == "Summary":
        get_sports_summary()
    elif sports_option == "Systems":
        get_sports_systems()

st.sidebar.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)

#
# Generative AI
#

if "markets" not in st.session_state:
    st.session_state["markets"] = [{"role": "assistant", "content": "Ask Alpha about Markets"}]

if "sports" not in st.session_state:
    st.session_state["sports"] = [{"role": "assistant", "content": "Ask Alpha about Sports"}]

if 'api_key' not in st.session_state:
    st.session_state['api_key'] = False

def is_valid_api_key(key):
    return key.startswith('sk-')

def test_openai_api_key(prompt):
    try:
        # Make a simple API request to the OpenAI API
        response = openai.chat.completions.create(model="gpt-4o",
            messages=[
                    {"role": "system", "content": "Hello"},
                    {"role": "user", "content": prompt},
                ])
    except openai.AuthenticationError:
        print("Invalid API key. Please check your API key and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")

if market_option == "Ask Alpha" or sports_option == "Ask Alpha":
    if not st.session_state['api_key']:
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = st.text_input('Enter OpenAI API token:', type='password')
        if not is_valid_api_key(api_key):
            st.warning('Invalid API key format. Please check and enter again.', icon='⚠️')
        else:
            st.success('API key looks good! Go ahead and Ask Alpha.', icon='✅')
            openai.api_key = api_key
            test_openai_api_key('Test Text')
            st.session_state['api_key'] = True
    if topic == market_string:
        for msg in st.session_state.markets:
            st.chat_message(msg["role"]).write(msg["content"])
        if prompt_text := st.chat_input():
            st.session_state.markets.append({"role": "user", "content": prompt_text})
            st.chat_message("user").write(prompt_text)
            response = openai.chat.completions.create(model="gpt-4o",
                messages=[
                        {"role": "system", "content": "Hello"},
                        {"role": "user", "content": prompt_text},
                    ])
            msg = response.choices[0].message.content
            st.session_state.markets.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
    elif topic == sports_string:
        for msg in st.session_state.sports:
            st.chat_message(msg["role"]).write(msg["content"])
        if prompt_text := st.chat_input():
            st.session_state.sports.append({"role": "user", "content": prompt_text})
            st.chat_message("user").write(prompt_text)
            response = openai.chat.completions.create(model="gpt-4o",
                messages=[
                        {"role": "system", "content": "Hello"},
                        {"role": "user", "content": prompt_text},
                    ])
            msg = response.choices[0].message.content
            st.session_state.sports.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
