################################################################################
#
# Package   : AlphaPy
# Module    : streamlit_requests
# Created   : April 22, 2022
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

import requests

from mflow_server import get_groups
from mflow_server import get_paths
from mflow_server import get_projects
from mflow_server import get_systems


#
# AlphaPy Dispatch Table
#

alphapy_dispatcher = {
     'groups'   : get_groups,
     'paths'    : get_paths,
     'projects' : get_projects,
     'systems'  : get_systems
}


#
# Function alphapy_request
#

def alphapy_request(alphapy_specs, item):
    use_server = alphapy_specs['use_server']
    if use_server:
        url = alphapy_specs['mflow']['server_url']
        r = requests.get(url+item)
        results = r.json()
    else:
        results = alphapy_dispatcher[item](alphapy_specs)
    return results


"""
st.title('Counter Example')
if 'count' not in st.session_state:
    st.session_state.count = 0
    st.session_state.last_updated = datetime.time(0,0)

def update_counter():
    st.session_state.count += st.session_state.increment_value
    st.session_state.last_updated = st.session_state.update_time

with st.form(key='my_form'):
    st.time_input(label='Enter the time', value=datetime.datetime.now().time(), key='update_time')
    st.number_input('Enter a value', value=0, step=1, key='increment_value')
    submit = st.form_submit_button(label='Update', on_click=update_counter)

st.write('Current Count = ', st.session_state.count)
st.write('Last Updated = ', st.session_state.last_updated)
"""