################################################################################
#
# Package   : AlphaPy
# Module    : ask_alpha
# Created   : February 21, 2021
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
# HOW TO RUN:
#
# export ALPHAPY_ROOT=/Users/markconway/Projects/alphapy-root
# cd /Users/markconway/Projects/alphapy-3.0.0/alphapy
# streamlit run ask_alpha.py
#


#
# Imports
#

from datetime import datetime, timedelta
import logging
import openai
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
 
openai.api_key = st.secrets["OPENAI_API_KEY"]
if 'OPENAI_API_KEY' in st.secrets:
    col3.success('OpenAI API key has been provided.', icon='✅')
    openai.api_key = st.secrets['OPENAI_API_KEY']
else:
    openai.api_key = col3.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==51):
        st.warning('Please enter your credentials!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')

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

if prompt_text:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ]
    )
    answer = response.choices[0].message.content
    # Display response
    st.text_area('Response:', answer, height=350)
