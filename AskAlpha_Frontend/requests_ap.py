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

import logging
import requests
import streamlit as st
import subprocess

from MFlow_Backend.mflow_server import request_groups
from MFlow_Backend.mflow_server import request_market_config
from MFlow_Backend.mflow_server import request_model_config
from MFlow_Backend.mflow_server import request_paths
from MFlow_Backend.mflow_server import request_projects


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


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_web_content
#

def get_web_content(url):
    r"""Use the requests package to get data over HTTP.

    Parameters
    ----------
    url : str
        The URL for making the request over HTTP.

    Returns
    -------
    response : str
        The results returned from the request.

    """

    logger.debug(f"Connecting to {url}")
    try:
        response = requests.get(url)

        # Successful request
        if response.status_code == 200:
            logger.debug("Success!")
            return response.text

        # Page not found
        elif response.status_code == 404:
            logger.debug("Error: Page not found.")
            return None

        # Server error
        elif response.status_code >= 500:
            logger.debug("Server error.")
            return None

        # Other errors
        else:
            logger.debug(f"Unexpected status code: {response.status_code}")
            return None

    except requests.ConnectionError:
        logger.debug("Error: Failed to establish a new connection.")
        return None

    except requests.Timeout:
        logger.debug("Error: The request timed out.")
        return None

    except requests.TooManyRedirects:
        logger.debug("Error: Too many redirects.")
        return None

    except requests.RequestException as e:
        logger.debug(f"Error: An unexpected error occurred. {e}")
        return None


#
# Function alphapy_request
#

def alphapy_request(alphapy_specs, item, *args):
    r"""Make a request to an AlphaPy server.

    Parameters
    ----------
    alphapy_specs : str
        The specifications for AlphaPy.
    item : str
        The name of the AlphaPy server.
    args : str
        The specific request.

    Returns
    -------
    response : str
        The results returned from the AlphaPy server.

    """
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
    r"""Run a subprocess based on the command with arguments.

    Parameters
    ----------
    cmd_with_args : str
        The command to run as a subprocess.
    cwd: str
        The current working directory.

    Returns
    -------
    result : str
        The result returned from running the subprocess.

    """
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
