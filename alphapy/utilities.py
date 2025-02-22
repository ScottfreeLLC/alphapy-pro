################################################################################
#
# Package   : AlphaPy
# Module    : utilities
# Created   : July 11, 2013
#
# Copyright 2024 ScottFree Analytics LLC
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

from alphapy.globals import PSEP, SSEP

import argparse
from datetime import datetime, timedelta
import glob
import logging
import numpy as np
import os
import re
import requests
import subprocess


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function datetime_stamp
#

def datetime_stamp():
    r"""Returns today's datetime stamp.

    Returns
    -------
    dtstamp : str
        The valid datetime string in YYYYmmdd_hhmmss format.

    """
    d = datetime.now()
    f = "%Y%m%d_%H%M%S"
    dtstamp = d.strftime(f)
    return dtstamp


#
# Function ensure_dir
#

def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


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
# Function most_recent_file
#

def most_recent_file(directory, file_spec):
    r"""Find the most recent file in a directory.

    Parameters
    ----------
    directory : str
        Full directory specification.
    file_spec : str
        Wildcard search string for the file to locate.

    Returns
    -------
    file_name : str
        Name of the file to read, excluding the ``extension``.

    """

    # Create search path
    search_path = SSEP.join([directory, file_spec])
    # find the latest file
    file_name = max(glob.iglob(search_path), key=os.path.getctime)
    # load the model predictor
    return file_name


#
# Function np_store_data
#

def np_store_data(data, dir_name, file_name, extension, separator):
    r"""Store NumPy data in a file.

    Parameters
    ----------
    data : numpy array
        The model component to store
    dir_name : str
        Full directory specification.
    file_name : str
        Name of the file to read, excluding the ``extension``.
    extension : str
        File name extension, e.g., ``csv``.
    separator : str
        The delimiter between fields in the file.

    Returns
    -------
    None : None

    """

    output_file = PSEP.join([file_name, extension])
    output = SSEP.join([dir_name, output_file])
    logger.info("Storing output to %s", output)
    np.savetxt(output, data, delimiter=separator)


#
# Function remove_list_items
#

def remove_list_items(elements, alist):
    r"""Remove one or more items from the given list.

    Parameters
    ----------
    elements : list
        The items to remove from the list ``alist``.
    alist : list
        Any object of any type can be a list item.

    Returns
    -------
    sublist : list
        The subset of items after removal.

    Examples
    --------

    >>> test_list = ['a', 'b', 'c', test_func]
    >>> remove_list_items([test_func], test_list)  # ['a', 'b', 'c']

    """

    sublist = [x for x in alist if x not in elements]
    return sublist


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
        logger.info(result_text)
    except subprocess.CalledProcessError as e:
        error_text = result.stderr.replace('[', '\n[')
        error_split_text = error_text.split('Traceback')
        logger.info(error_split_text[0])
        logger.error(' '.join(['ERROR', error_split_text[1]]))
    return result


#
# Function split_duration
#

def split_duration(duration_str):
    r"""Subtract a number of days from a given date.

    Parameters
    ----------
    duration_str : str
        An alphanumeric string in the format of a Pandas offset alias.

    Returns
    -------
    value : int
        The duration value.
    unit : str
        The scale of the period.

    Examples
    --------

    >>> split_duration('5min')   # 5, 'min'

    """
    match = re.match(r"(\d+)(\w+)", duration_str)
    if match:
        value, unit = match.groups()
        return int(value), unit
    else:
        raise ValueError(f"Invalid duration format: {duration_str}")


#
# Function subtract_days
#

def subtract_days(date_string, ndays):
    r"""Subtract a number of days from a given date.

    Parameters
    ----------
    date_string : str
        An alphanumeric string in the format %Y-%m-%d.
    ndays : int
        Number of days to subtract.

    Returns
    -------
    new_date_string : str
        The adjusted date string in the format %Y-%m-%d.

    Examples
    --------

    >>> subtract_days('2017-11-10', 31)   # '2017-10-10'

    """

    new_date_string = None
    valid = valid_date(date_string)
    if valid:
        date_dt = datetime.strptime(date_string, "%Y-%m-%d")
        new_date = date_dt - timedelta(days=ndays)
        new_date_string = new_date.strftime("%Y-%m-%d")
    return new_date_string


#
# Function valid_date
#

def valid_date(date_string):
    r"""Determine whether or not the given string is a valid date.

    Parameters
    ----------
    date_string : str
        An alphanumeric string in the format %Y-%m-%d.

    Returns
    -------
    date_string : str
        The valid date string.

    Raises
    ------
    ValueError
        Not a valid date.

    Examples
    --------

    >>> valid_date('2016-7-1')   # datetime.datetime(2016, 7, 1, 0, 0)
    >>> valid_date('345')        # ValueError: Not a valid date

    """

    try:
        date_time = datetime.strptime(date_string, "%Y-%m-%d")
        return date_string
    except:
        message = "Not a valid date: '{0}'.".format(date_string)
        raise argparse.ArgumentTypeError(message)


#
# Function valid_name
#

def valid_name(name):
    r"""Determine whether or not the given string is a valid
    alphanumeric string.

    Parameters
    ----------
    name : str
        An alphanumeric identifier.

    Returns
    -------
    result : bool
        ``True`` if the name is valid, else ``False``.

    Examples
    --------

    >>> valid_name('alpha')   # True
    >>> valid_name('!alpha')  # False

    """

    identifier = re.compile(r"^[^\d\W]\w*\Z", re.UNICODE)
    result = re.match(identifier, name)
    return result is not None
