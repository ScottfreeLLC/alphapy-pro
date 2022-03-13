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
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import requests
import streamlit as st
import streamlit_alphapy
import streamlit_mflow
import streamlit_sflow
from streamlit_multipage import MultiPage
import subprocess
import time


# Create an instance of the application
app = MultiPage()

# Display the Scottfree logo

logo = Image.open('logo.jpg')
st.sidebar.image(logo)

# Add all your applications (pages) here

app.add_page("AlphaPy AutoML", streamlit_alphapy.app)
app.add_page("AlphaPy Markets", streamlit_mflow.app)
app.add_page("AlphaPy Sports", streamlit_sflow.app)


#result = subprocess.run(['pyomo', 'solve', 'my_model.py', '--solver="cbc"'])
#st.write(result.stdout)  # Do something interesting with the result

base_url = 'http://localhost:8000/'
url_item = 'groups'
r = requests.get(base_url+url_item) # Make HTTPS call 
groups = r.json() # Decode JSON

for g in groups.items():
    print(g)
    st.sidebar.text(g)

# The main app
app.run()
