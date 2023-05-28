################################################################################
#
# Package   : AlphaPy
# Module    : requests_ap
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

import subprocess
import requests
import streamlit as st

from alphapy.mflow_server import request_groups
from alphapy.mflow_server import request_market_config
from alphapy.mflow_server import request_model_config
from alphapy.mflow_server import request_paths
from alphapy.mflow_server import request_projects


#
# AlphaPy Dispatch Table
#

alphapy_dispatcher = {
     'groups'        : request_groups,
     'market_config' : request_market_config,
     'model_config'  : request_model_config,
     'paths'         : request_paths,
     'projects'      : request_projects
}


def get_web_content(url):
    try:
        response = requests.get(url)

        # Successful request
        if response.status_code == 200:
            print("Success!")
            return response.text

        # Page not found
        elif response.status_code == 404:
            print("Error: Page not found.")
            return None

        # Server error
        elif response.status_code >= 500:
            print("Server error.")
            return None

        # Other errors
        else:
            print(f"Unexpected status code: {response.status_code}")
            return None

    except requests.ConnectionError:
        print("Error: Failed to establish a new connection.")
        return None

    except requests.Timeout:
        print("Error: The request timed out.")
        return None

    except requests.TooManyRedirects:
        print("Error: Too many redirects.")
        return None

    except requests.RequestException as e:
        print(f"Error: An unexpected error occurred. {e}")
        return None


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
