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

import json
import logging
import os

from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Global Constants
#

CREDENTIALS_FILE = './client_secrets.json'
SCOPES = ['https://www.googleapis.com/auth/drive']
TOKEN_FILE = './token.json'

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
# Function save_credentials_to_file
#

def save_credentials_to_file(creds, file_path=TOKEN_FILE):
    """Save Google credentials to a file."""
    with open(file_path, 'w') as token:
        token.write(creds.to_json())

#
# Function get_google_credentials
#

def authenticate_google():
    """Handle Google Authentication."""
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE)
        # If the credentials are not valid, delete the token file and re-authenticate
        if not creds.valid:
            os.remove(TOKEN_FILE)
            return authenticate_google()
    else:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        save_credentials_to_file(creds)
    return creds

#
# Function get_google_credentials
#

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
        logger.info(f"Error retrieving credentials from {temp_file_path}: {e}")
        return None

#
# Function authenticate_google_drive
#

def authenticate_google_drive(creds):
    """Authenticates and returns a Google Drive service object."""
    return build('drive', 'v3', credentials=creds)

#
# Function get_gfile_id
#

def get_gfile_id(drive_service, file_name, folder_id=None):
    """Retrieve the file ID of an existing file based on its name in Google Drive."""
    query = f"name='{file_name}'"
    if folder_id:
        query += f" and '{folder_id}' in parents"
    try:
        response = drive_service.files().list(q=query, spaces='drive', fields='nextPageToken, files(id, name)').execute()
        for file in response.get('files', []):
            if file.get('name') == file_name:
                return file.get('id')
    except HttpError as error:
        logger.error(f"An error occurred: {error}")
    return None

#
# Function upload_to_drive
#

def upload_to_drive(drive_service, file_path, folder_id=None):
    """Uploads a file to Google Drive."""
    file_metadata = {
        'name': os.path.basename(file_path)
    }
    if folder_id:
        file_metadata['parents'] = [folder_id]

    file_id = get_gfile_id(drive_service, file_metadata['name'], folder_id)
    media = MediaFileUpload(file_path)

    if file_id:
        verb = 'replaced'
        drive_service.files().update(fileId=file_id, media_body=media).execute()
    else:
        verb = 'created'
        drive_service.files().create(body=file_metadata, media_body=media).execute()

    logger.info(f"{file_path} has been {verb} successfully on Google Drive.")
