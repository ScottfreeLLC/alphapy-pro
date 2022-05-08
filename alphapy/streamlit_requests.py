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
import streamlit as st
import subprocess

from mflow_server import request_groups
from mflow_server import request_market_config
from mflow_server import request_model_config
from mflow_server import request_paths
from mflow_server import request_projects
from mflow_server import request_systems


#
# AlphaPy Dispatch Table
#

alphapy_dispatcher = {
     'groups'        : request_groups,
     'market_config' : request_market_config,
     'model_config'  : request_model_config,
     'paths'         : request_paths,
     'projects'      : request_projects,
     'systems'       : request_systems
}


#
# Function alphapy_request
#

def alphapy_request(alphapy_specs, item, *args):
    use_server = alphapy_specs['use_server']
    if use_server:
        url = alphapy_specs['mflow']['server_url']
        r = requests.get(url+item)
        results = r.json()
    else:
        results = alphapy_dispatcher[item](*args)
    return results


#
# Function run_subprocess
#

def run_command(cmd_with_args, cwd):
    result = subprocess.run(cmd_with_args, capture_output=True, text=True, cwd=cwd)
    try:
        result.check_returncode()
        result_text = result.stderr.replace('[', '\n[')
        st.info(result_text)
    except subprocess.CalledProcessError as e:
        error_text = result.stderr.replace('[', '\n[')
        error_split_text = error_text.split('Traceback')
        st.info(error_split_text[0])
        st.error(' '.join(['ERROR', error_split_text[1]]))
    return result
