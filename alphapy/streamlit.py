################################################################################
#
# Package   : AlphaPy
# Module    : streamlit
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

import glob
import numpy as np
import pandas as pd
from pathlib import Path
import requests
import streamlit as st
import subprocess
import time

st.set_page_config(layout="wide")

# st.sidebar.image('./logo.jpg', use_column_width=True)
# st.sidebar.markdown("<h1 style='text-align: center; color: red;'>AlphaPy :chart_with_upwards_trend: Vikki :woman:</h1>", unsafe_allow_html=True)
st.sidebar.title('Vikki :woman: the AlphaPy :chart_with_upwards_trend: UI')

st.sidebar.text('groups')

base_url = 'http://localhost:8000/'
url_item = 'groups'
r = requests.get(base_url+url_item) # Make HTTPS call 
groups = r.json() # Decode JSON

for g in groups.items():
    print(g)
    st.sidebar.text(g)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
vikki_mode = st.radio('', ['Chart', 'Symbol', 'System', 'Model', 'Portfolio', 'Account'])

def do_charts():
    pass

def do_symbols():
    pass

def do_systems():
    pass

def do_models():
    pass

def do_portfolios():
    pass

def do_accounts():
    pass

if vikki_mode == 'Chart':
    st.write(vikki_mode)
elif vikki_mode == 'Symbol':
    st.write(vikki_mode)
elif vikki_mode == 'System':
    st.write(vikki_mode)
elif vikki_mode == 'Model':
    st.write(vikki_mode)
elif vikki_mode == 'Portfolio':
    st.write(vikki_mode)
elif vikki_mode == 'Account':
    st.write(vikki_mode)


"""
option = st.sidebar.selectbox(
    'What do you want to explore?',
     ('Machine Learning', 'Markets', 'Sports'))

home_directory = str(Path.home())
st.sidebar.subheader("Project Root Directory")

if st.sidebar.checkbox("Use Home Directory", True):
    root_directory = home_directory
    st.sidebar.markdown(home_directory)
else:
    root_directory = st.sidebar.text_input('Root Directory', home_directory)
    st.sidebar.markdown(root_directory)

@st.cache
def get_projects(file_name, directory):
    paths = []
    for path in Path(directory).rglob(file_name):
        paths.append(path)
    return paths

#result = subprocess.run(['pyomo', 'solve', 'my_model.py', '--solver="cbc"'])
#st.write(result.stdout)  # Do something interesting with the result

st.sidebar.subheader("Projects")
st.sidebar.markdown(get_projects('model.yml', root_directory))



st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(chart_data)

'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'
"""