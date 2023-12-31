################################################################################
#
# Package   : AlphaPy
# Module    : google_utils
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

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json
import logging
import os
import pandas as pd


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Global Constants
#

CREDENTIALS_FILE = './client_secrets.json'
SCOPES = ['https://www.googleapis.com/auth/drive']


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

def save_credentials_to_file(creds, token_file):
    """Save Google credentials to a file."""

    with open(token_file, 'w') as token:
        token.write(creds.to_json())
    logger.info(f"Credentials saved to {token_file}")


#
# Function get_google_credentials
#

def authenticate_google(script_directory='.', token_file='token.json'):
    """Handle Google Authentication."""

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file)
        # If the credentials are not valid, delete the token file and re-authenticate
        if not creds.valid:
            os.remove(token_file)
            return authenticate_google(script_directory, token_file)
    else:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        save_credentials_to_file(creds, token_file)
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
# Function get_sheets_service
#

def get_sheets_service(creds):
    return build('sheets', 'v4', credentials=creds)


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
# Function find_gsheet_by_name
#

def find_gsheet_by_name(drive_service, name):
    """
    Find a Google Sheet by name.
    
    :param drive_service: The authenticated Google Drive service object.
    :param name: The name of the Google Sheet to find.
    :return: The file ID of the first Google Sheet with the given name or None.
    """

    response = drive_service.files().list(q=f"name='{name}' and mimeType='application/vnd.google-apps.spreadsheet'",
                                          spaces='drive',
                                          fields='files(id, name)').execute()
    for file in response.get('files', []):
        # Return the first file id that matches the name
        return file.get('id')
    return None


#
# Function get_sheet_id_by_name
#

def get_sheet_id_by_name(sheets_service, spreadsheet_id, sheet_name):
    """
    Retrieves the sheet ID for a given sheet name within a spreadsheet.

    :param sheets_service: The authenticated Google Sheets service object.
    :param spreadsheet_id: The ID of the spreadsheet.
    :param sheet_name: The name of the sheet.
    :return: The ID of the sheet or None if not found.
    """

    # Get metadata about all sheets in the spreadsheet
    sheet_metadata = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheets = sheet_metadata.get('sheets', '')

    # Find the ID of the sheet with the given name
    for sheet in sheets:
        if sheet['properties']['title'] == sheet_name:
            return sheet['properties']['sheetId']

    return None


#
# Function format_header_row
#

def format_header_row(sheets_service, spreadsheet_id, sheet_name):
    """
    Formats the header row of a specific sheet in a Google Sheets spreadsheet.

    :param sheets_service: The authenticated Google Sheets service object.
    :param spreadsheet_id: The ID of the spreadsheet.
    :param sheet_name: The name of the sheet to format.
    """

    # Find the ID of the sheet
    sheet_id = get_sheet_id_by_name(sheets_service, spreadsheet_id, sheet_name)
    if sheet_id is None:
        logger.info(f"Sheet named '{sheet_name}' not found in the spreadsheet.")
        return

    # Define the header text format requests
    requests = [{
        "repeatCell": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": 0,
                "endRowIndex": 1
            },
            "cell": {
                "userEnteredFormat": {
                    "textFormat": {
                        "fontSize": 10,
                        "bold": True
                    }
                }
            },
            "fields": "userEnteredFormat.textFormat(fontSize,bold)"
        }
    }]

    # Send the batchUpdate request
    body = {
        'requests': requests
    }
    response = sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=body
    ).execute()

    logger.info(f"Formatted header row for Sheet ID: {sheet_id} in '{sheet_name}'")


#
# Function apply_alternating_row_colors
#

def apply_alternating_row_colors(sheets_service, spreadsheet_id, sheet_name, start_row, end_row):
    """
    Applies alternating row colors to a specified range in a Google Sheet for better readability.

    :param sheets_service: The authenticated Google Sheets service object.
    :param spreadsheet_id: The ID of the Google Sheet.
    :param sheet_name: The name of the sheet in the spreadsheet.
    :param start_row: The starting row index for the banding (0-indexed).
    :param end_row: The ending row index for the banding (0-indexed).
    """

    # Get the sheet ID based on the sheet name
    sheet_id = get_sheet_id_by_name(sheets_service, spreadsheet_id, sheet_name)
    if sheet_id is None:
        logger.info(f"No sheet found with the name '{sheet_name}'")
        return

    # Define the request for alternating row colors
    requests = [{
        "addBanding": {
            "bandedRange": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": start_row,
                    "endRowIndex": end_row
                },
                "rowProperties": {
                    "headerColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
                    "firstBandColor": {"red": 0.88, "green": 0.96, "blue": 1.0},
                    "secondBandColor": {"red": 1.0, "green": 1.0, "blue": 1.0}
                }
            }
        }
    }]

    # Send the batchUpdate request
    body = {'requests': requests}

    try:
        response = sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=body
        ).execute()
        logger.info(f"Applied alternating row colors to '{sheet_name}' in the spreadsheet.")
    except HttpError as error:
        logger.info(f"Alternating row colors already applied: {error}")
        return    


#
# Function auto_resize_columns
#

def auto_resize_columns(sheets_service, spreadsheet_id, start_column, end_column):
    """
    Sets the column width to auto fit the content in the specified range.

    :param sheets_service: The authenticated Google Sheets service object.
    :param spreadsheet_id: The ID of the Google Sheet.
    :param start_column: The starting column index to resize.
    :param end_column: The ending column index to resize.
    """

    # Define the request to update column widths
    requests = [{
        "updateDimensionProperties": {
            "range": {
                "sheetId": 0,
                "dimension": "COLUMNS",
                "startIndex": start_column,
                "endIndex": end_column
            },
            "properties": {
                "pixelSize": 150
            },
            "fields": "pixelSize"
        }
    }]

    # Send the batchUpdate request
    body = {'requests': requests}
    response = sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=body
    ).execute()

    logger.info(f"Auto-resized columns from {start_column} to {end_column} in Sheet ID: {sheet_id}")


#
# Function apply_conditional_formatting
#

def apply_conditional_formatting(sheets_service, spreadsheet_id, sheet_name, format_dict):
    """
    Applies conditional formatting to the Google Sheet.

    :param sheets_service: The authenticated Google Sheets service object.
    :param sheet_name: The name of the sheet.
    :param sheet_id: The ID of the Google Sheet to format.
    :param format_dict: Dictionary of formatting options.
    """

    # Get the sheet ID based on the sheet name
    sheet_id = get_sheet_id_by_name(sheets_service, spreadsheet_id, sheet_name)
    if sheet_id is None:
        logger.info(f"No sheet found with the name '{sheet_name}'")
        return

    # format_dict = {'league'    : league,
    #                'target'    : target,
    #                'file_type' : file_type,
    #                'level'     : level}

    #
    # results_nb (6: 1 green, 0 red)
    # results_sb (6: 1 green, 0 red)
    #
    # over (6), won_on_points (5), won_on_spread (7)
    # predictions_nb (n6-12: 1 green, 0 red)
    # predictions_sb (n6-12: >68 green, 50-68 light green, 32-50 light red, <32 red)
    #                (13-14: >10 green, >0-10 light green, -10<0 light red, <-10 red)
    #
    # summary_nb (4-5: >55 green, 50-55 light green, 45-50 light red, <45 red)
    # summary_sb (4, 5, 7, 9, 11, 13)
    #

    # Define the conditional formatting rules

    requests = [
    {
        "addConditionalFormatRule": {
            "rule": {
                "ranges": [{
                    "sheetId": 0,  # Assumes the first sheet - you might need to adjust this depending on the actual sheet ID
                    "startColumnIndex": 0,  # Adjust if your data starts from a different column
                    "endColumnIndex": 2,  # Adjust based on the number of columns to format
                }],
                "booleanRule": {
                    "condition": {
                        "type": "NUMBER_EQ",
                        "values": [{"userEnteredValue": "1"}]
                    },
                    "format": {
                        "backgroundColor": {
                            "red": 0.0,
                            "green": 1.0,
                            "blue": 0.0
                        }
                    }
                }
            },
            "index": 0
        }
    },
    {
        "addConditionalFormatRule": {
            "rule": {
                "ranges": [{
                    "sheetId": 0,
                    "startColumnIndex": 0,
                    "endColumnIndex": 2,
                }],
                "booleanRule": {
                    "condition": {
                        "type": "NUMBER_EQ",
                        "values": [{"userEnteredValue": "0"}]
                    },
                    "format": {
                        "backgroundColor": {
                            "red": 1.0,
                            "green": 0.0,
                            "blue": 0.0
                        }
                    }
                }
            },
            "index": 1
        }
    }]
    
    # Send the batchUpdate request
    body = {
        'requests': requests
    }
    response = sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=sheet_id,
        body=body
    ).execute()
    
    logger.info(f"Applied conditional formatting to Sheet ID: {sheet_id}")


#
# Function apply_gradient_formatting
#

def apply_gradient_formatting(sheets_service, sheet_id, start_column, end_column):
    """
    Applies gradient conditional formatting to the Google Sheet based on cell values.

    :param sheets_service: The authenticated Google Sheets service object.
    :param sheet_id: The ID of the Google Sheet to format.
    :param start_column: The starting column index for applying the formatting.
    :param end_column: The ending column index for applying the formatting.
    """

    # Define the gradient conditional formatting rules
    requests = [{
        "addConditionalFormatRule": {
            "rule": {
                "gradientRule": {
                    "minpoint": {
                        "color": {"red": 1.0, "green": 0.0, "blue": 0.0},  # Red color
                        "type": "MIN"
                    },
                    "midpoint": {
                        "color": {"red": 1.0, "green": 1.0, "blue": 0.0},  # Yellow color
                        "type": "PERCENTILE",
                        "value": "50"
                    },
                    "maxpoint": {
                        "color": {"red": 0.0, "green": 1.0, "blue": 0.0},  # Green color
                        "type": "MAX"
                    }
                },
                "ranges": [{
                    "sheetId": 0,
                    "startColumnIndex": start_column,
                    "endColumnIndex": end_column,
                    "startRowIndex": 1  # Assuming the first row is headers
                }]
            }
        }
    }]
    
    # Send the batchUpdate request
    body = {'requests': requests}
    response = sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=sheet_id,
        body=body
    ).execute()

    logger.info(f"Applied gradient formatting to columns {start_column} to {end_column} in Sheet ID: {sheet_id}")


#
# Function highlight_differences
#

def highlight_differences(sheets_service, sheet_id, range_to_check, range_to_highlight):
    """
    Applies conditional formatting to highlight rows where predicted values differ from actual values.

    :param sheets_service: The authenticated Google Sheets service object.
    :param sheet_id: The ID of the Google Sheet to format.
    :param range_to_check: The A1 notation of the range where the predicted values are to check against actual values.
    :param range_to_highlight: The A1 notation of the range to highlight when there is a difference.
    """

    # Define the request for conditional formatting
    requests = [{
        "addConditionalFormatRule": {
            "rule": {
                "ranges": [{
                    "sheetId": sheet_id,
                    "startRowIndex": 1,
                    "endRowIndex": None,
                    "startColumnIndex": None,
                    "endColumnIndex": None
                }],
                "booleanRule": {
                    "condition": {
                        "type": "CUSTOM_FORMULA",
                        "values": [{
                            "userEnteredValue": f"=NOT(EXACT({range_to_check}, {range_to_highlight}))"
                        }]
                    },
                    "format": {
                        "backgroundColor": {"red": 1.0, "green": 0.9, "blue": 0.9}
                    }
                }
            },
            "index": 0
        }
    }]

    # Send the batchUpdate request
    body = {'requests': requests}
    response = sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=sheet_id,
        body=body
    ).execute()

    logger.info(f"Conditional formatting applied to highlight differences in Sheet ID: {sheet_id}")


#
# Function gsheet_format
#

def gsheet_format(creds, drive_service, csv_file_id, df, format_dict):
    """
    Updates an existing Google Sheet with the contents of a new CSV file and formats it.
    
    :param creds: The Google Credentials object.
    :param drive_service: The authenticated Google Drive service object.
    :param csv_file_id: The ID of the uploaded CSV file.
    :param df: pandas DataFrame containing the CSV data.
    :param format_dict: Dictionary of formatting options.
    """

    # Initialize the Google Sheets service object
    sheets_service = build('sheets', 'v4', credentials=creds)

    # Get the name of the CSV file
    csv_file_metadata = drive_service.files().get(fileId=csv_file_id, fields='name').execute()
    csv_file_name = csv_file_metadata.get('name')

    # Find the Google Sheet by name
    sheet_name = csv_file_name.rsplit('.', 1)[0]
    sheet_id = find_gsheet_by_name(drive_service, sheet_name)

    if not sheet_id:
        logger.info(f"No Google Sheet found with name '{sheet_name}'")
        return

    # Define the Google Sheet range to clear
    range_all = f'{sheet_name}!A:Z'

    # Clear the existing data in the Google Sheet
    sheets_service.spreadsheets().values().clear(
        spreadsheetId=sheet_id,
        range=range_all,
        body={}
    ).execute()

    # Prepare the new CSV data for uploading
    headers = [df.columns.tolist()]
    for col in df.select_dtypes(include=['datetime']):
        df[col] = df[col].dt.strftime('%Y-%m-%d')
    data = df.fillna('').values.tolist()
    values = headers + data
    body = {'values': values}

    # Write the new data to the Google Sheet
    sheets_service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range=range_all,
        valueInputOption='USER_ENTERED',
        body=body
    ).execute()

    # Apply formatting functions
    format_header_row(sheets_service, sheet_id, sheet_name)

    # Apply alternating colors
    alternating_colors = False
    if alternating_colors:
        start_row = 1
        end_row = len(df) + 1
        apply_alternating_row_colors(sheets_service, sheet_id, sheet_name, start_row, end_row)

    # Apply conditional formatting
    # apply_conditional_formatting(sheets_service, sheet_id, sheet_name, format_dict)

    logger.info(f"Google Sheet with ID {sheet_id} has been updated with the new CSV data.")


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
    return file_id
