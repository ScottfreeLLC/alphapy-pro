################################################################################
#
# Package   : AlphaPy
# Module    : google_drive
# Created   : October 21, 2023
#
# Copyright 2023 ScottFree Analytics LLC
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
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Google Drive Dictionary
#

gdrive_dict = {
    'nb_mlb'   : '1sBuixscaAzMU8ktMi7Hsw2hZC4-3KqEc',
    'nb_nba'   : '1ws_c-EpBi3GVbtSw6qc0Ca-E1uumQfNS',
    'nb_ncaab' : '1ca9A8h8HJG5Dwuhq8xLKQO8LvnMguO-I',
    'nb_ncaaf' : '10D7-y4XAgWfroYg5MYFoYngKn3xSd7oP',
    'nb_nfl'   : '1SaQN7SSX4df9L5-b0_kyfe41HeYjwwX2',
    'nb_nhl'   : '1F_6TIjRPa4q_6BdThJOwPhtST-2Hw1oU',
    'sb_mlb'   : '1RI23LUdz_MfEq05HzRcBA6g1hAFP8RIl',
    'sb_nba'   : '11UO1s2kX_24x5ZGFcpXWRca94dt4--jb',
    'sb_ncaab' : '17w-Ck0tM0mmBzWMPGM2xWbfHcs-D8eY9',
    'sb_ncaaf' : '10wzKNHQHa9e_DejNhgnKJ_97H6ULMsFx',
    'sb_nfl'   : '1KQyBG5aqLbMYFLHH9ws0EyEHLEq166Ia',
    'sb_nhl'   : '1tSgMfgnUNz_i7sk0X71fOTmUCKhD6iSA',
}


#
# Function authenticate_google_drive
#

def authenticate_google_drive(creds):
    """
    Authenticates and returns a Google Drive object.
    """
    gauth = GoogleAuth()
    gauth.credentials = creds
    return GoogleDrive(gauth)


#
# Function upload_to_drive
#

def upload_to_drive(drive, file_path, folder_id=None):
    """
    Uploads a file to Google Drive.
    :param drive: Authenticated Google Drive object.
    :param file_path: Path to the file to be uploaded.
    :param folder_id: Google Drive folder ID where the file will be uploaded.
    """
    file = drive.CreateFile({'parents': [{'id': folder_id}]}) if folder_id else drive.CreateFile()
    file.SetContentFile(file_path)
    file.Upload()
    logger.info(f"'{file_path}' has been uploaded successfully to Google Drive.")


import json
from google.oauth2.credentials import Credentials

def get_google_credentials(temp_file_path):
    """
    Retrieve Google credentials from a temporary file.
    
    Args:
    - temp_file_path (str): Path to the temporary file containing the serialized credentials.
    
    Returns:
    - Credentials object or None if there's an error.
    """
    try:
        with open(temp_file_path, 'r') as temp_file:
            creds_json = json.load(temp_file)
        return Credentials.from_authorized_user_info(creds_json)
    except Exception as e:
        print(f"Error retrieving credentials from {temp_file_path}: {e}")
        return None

# Usage example:
temp_file_path = "path_to_temp_file"  # Replace with the actual path to your temp file
creds = get_google_credentials(temp_file_path)
