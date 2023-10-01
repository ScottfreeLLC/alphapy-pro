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
from alphapy.requests_ap import alphapy_request
from alphapy.requests_ap import run_command


#
# Global Variables
#

dir_assets = './assets/'


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function run_project
#

def run_project(project):

    # Vet the model and market specifications

    project_root = '/'.join([alphapy_specs['mflow']['project_root'], project])
    model_specs, model_dict = alphapy_request(alphapy_specs, 'model_config', project_root)
    market_specs, market_dict = alphapy_request(alphapy_specs, 'market_config', project_root)

    # Determine the source of market groups

    text_ap = 'Market Flow'
    text_fp = 'Finviz Portfolio'
    text_mi = 'Market Index'
    screener = st.sidebar.radio("Group Source", (text_ap, text_fp, text_mi))

    col1, col2, col3, col4 = st.columns(4)

    if screener == text_ap:
        groups = alphapy_request(alphapy_specs, 'groups')
    elif screener == text_fp:
        groups = get_finviz_portfolios()
    elif screener == text_mi:
        groups = get_market_index_groups()

    # Select the group to test

    group_list = list(groups.keys())
    if screener == text_ap:
        group_default = market_specs['data']['target_group']
        group_list.remove(group_default)
        group_list.insert(0, group_default)
    group_text = ' '.join(['Select', screener, 'Group'])
    group = col1.selectbox(group_text, group_list)

    # Select the date range (market:data_start_date and market:data_end_date)
    # If the configuration variable market:data_history is set, then calculate the dates.

    start_date_default = market_specs['data']['data_start_date']
    end_date_default = market_specs['data']['data_end_date']
    data_history_default = market_specs['data']['data_history']
    today = datetime.now()
    if data_history_default and start_date_default and end_date_default:
        from_date = start_date_default
        to_date = end_date_default
    elif data_history_default and start_date_default and not end_date_default:
        from_date = start_date_default
        to_date = today
    elif data_history_default and not start_date_default and end_date_default:
        from_date = end_date_default - timedelta(days=data_history_default)
        to_date = end_date_default
    elif data_history_default and not start_date_default and not end_date_default:
        from_date = today - timedelta(days=data_history_default)
        to_date = today
    elif not data_history_default and start_date_default and end_date_default:
        from_date = start_date_default
        to_date = end_date_default
    elif not data_history_default and start_date_default and not end_date_default:
        from_date = start_date_default
        to_date = today
    elif not data_history_default and not start_date_default and end_date_default:
        from_date = end_date_default - timedelta(days=365)
        to_date = end_date_default
    elif not data_history_default and not start_date_default and not end_date_default:
        from_date = today - timedelta(days=365)
        to_date = today

    col2.date_input('From', from_date)
    col2.date_input('To', to_date)

    # Select the symbols

    group_container = col1.container()
    select_all = col1.checkbox("Select all")
    select_text = "Select one or more symbols:"
    if screener == text_ap:
        symbols = sorted(map(lambda x: x.upper(), groups[group].members))
    else:
        symbols = sorted(map(lambda x: x.upper(), groups[group]))
    
    if select_all:
        selected_symbols = group_container.multiselect(select_text, symbols, symbols)
    else:
        selected_symbols =  group_container.multiselect(select_text, symbols)

    # Modify any settings

    market_setting = col3.selectbox('Select Market Settings Group', market_specs.keys())
    select_dict = {}
    select_dict['data_source'] = alphapy_specs['sources'].keys()
    select_dict['subject'] = apg.SUBJECTS
    select_dict['algo'] = market_dict
    select_dict['bar_type'] = market_dict
    select_dict['fractals'] = market_dict
    select_dict['features'] = market_dict
    market_inputs = get_market_inputs(market_dict, select_dict)

    market_text = ' '.join(['View', market_setting, 'Settings'])
    with col3.expander(market_text):
        st.write(market_text)

    model_settings = col4.selectbox('Select Model Settings Group', model_specs.keys())
    model_text = ' '.join(['View', model_settings, 'Settings'])
    with col4.expander(model_text):
        st.write(model_text)

    # Run the selected action

    run_model_text = ' '.join(['Run', 'Model', project])
    run_system_text = ' '.join(['Run', 'System'])
    get_model_text = ' '.join(['Model', project, 'Results'])
    get_system_text = ' '.join(['System', 'Results'])
    select_action = col2.selectbox("Choose Action",
                        [None, run_model_text, run_system_text, get_model_text, get_system_text])

    status_ph = st.empty()
    status_ph.info("Status")

    in_progress = 'In Progress:'
    completed = 'Completed:'
    if select_action == run_model_text:
        status_text = ' '.join([in_progress, run_model_text])
        status_ph.info(status_text)
        with st.expander('View Log'):
            result = run_command(['mflow'], project_root)
        status_ph.info(' '.join([completed, select_action]))
    elif select_action == run_system_text:
        status_text = ' '.join([in_progress, run_system_text])
        status_ph.info(status_text)
        with st.expander('View Log'):
            result = run_command(['mflow'], project_root)
        status_ph.info(' '.join([completed, select_action]))
    elif select_action == get_model_text:
        st.info("Model Results Placeholder")
    elif select_action == get_system_text:
        st.info("System Results Placeholder")


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
    st.session_state["logging"] = True

# Start Streamlit

logger.info('*'*80)
logger.info("Streamlit Start")
logger.info('*'*80)

#
# Application Configuration
#

path_logo = os.path.join(dir_assets, 'logo.jpg')
print(path_logo)
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
    market_prompt = st.text_input('Ask Market Alpha:')

# Sports

if topic == sports_string:
    sports_prompt = st.text_input('Ask Sports Alpha:')
    # Set the number of columns
    col21, col22, col23, col24, col25, col26 = st.columns(6)
    # Define the options for the dropdown
    options = ["NFL", "NBA", "MLB", "NHL", "NCAAF", "NCAAB"]
    # Create a select dropdown with the options
    selected_option = col21.selectbox('Select League', options)


# Generative AI

if False:

    if "openai_model" not in st.session_state:
        gpt_model = "gpt-4-0613"
        st.session_state["openai_model"] = gpt_model

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask Alpha"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})