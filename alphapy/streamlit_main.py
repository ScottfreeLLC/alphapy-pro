################################################################################
#
# Package   : AlphaPy
# Module    : streamlit
# Created   : February 21, 2021
#
# streamlit run streamlit.py
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

import glob
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import streamlit as st
import streamlit_aflow
import streamlit_mflow
import streamlit_sflow
import streamlit_finviz
from streamlit_multipage import MultiPage
import subprocess
import sys
import time


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Main Program
#

# Initialize Logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                    filename="streamlit_main.log", filemode='a', level=logging.INFO,
                    datefmt='%m/%d/%y %H:%M:%S')
formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                              datefmt='%m/%d/%y %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# Start Streamlit

logger.info('*'*80)
logger.info("Streamlit Start")
logger.info('*'*80)

# Set Page Configuration (alternate names: setup_page, page, layout)

st.set_page_config(
    # Can be "centered" or "wide". In the future also "dashboard", etc.
	layout="wide",
    # Can be "auto", "expanded", "collapsed"
	initial_sidebar_state="auto",
    # String or None. Strings get appended with "• Streamlit".
	page_title=None,
    # String, anything supported by st.image, or None.
	page_icon=None,
)

# Set window padding

vertical_padding = 2
horizontal_padding = 2

st.markdown(f""" <style>
    .appview-container .main .block-container{{
        padding-top: {vertical_padding}rem;
        padding-right: {horizontal_padding}rem;
        padding-left: {horizontal_padding}rem;
        padding-bottom: {vertical_padding}rem;
    }} </style> """, unsafe_allow_html=True)

# Create an instance of the application
app = MultiPage()

# Display the Scottfree logo

logo = Image.open('logo.jpg')
st.sidebar.image(logo)

# Add all your applications (pages) here

app.add_page("Alpha Flow", streamlit_aflow.app)
app.add_page("Finviz Screener", streamlit_finviz.app)
app.add_page("Market Flow", streamlit_mflow.app)
app.add_page("Sport Flow", streamlit_sflow.app)

# Run the main application
app.run()
