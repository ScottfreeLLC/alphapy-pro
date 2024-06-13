=#
# Imports
#

import calendrical
import datetime
from datetime import date, datetime
import logging
import numpy as np
import os
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from PIL import Image
from plotly_calplot import calplot
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import yaml


#
# Initialize logger
#

if "logging" not in st.session_state:
    logger = logging.getLogger(__name__)
    # Initialize Logging
    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="app_bt.log", filemode='a', level=logging.INFO,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                    datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    # Start the application
    logger.info('*'*80)
    logger.info("BT App Start")
    logger.info('*'*80)
    st.session_state.logging = True


#
# Global Variables
#

dir_assets = './assets/'
dir_data = './data/'
dir_plans = dir_data + 'plans/'
dir_plan_default = 'Plan_Original/'
dir_tables = dir_data + 'tables/'
dir_scoring = './scoring/'
dir_models = ''.join([dir_scoring, 'models/'])
path_models = ''.join([dir_scoring, 'models_dict_json.txt'])

live_calendar_start_date = "2022-01-01"
live_calendar_end_date = "2023-12-31"


#
# Application Configuration
#

path_image = ''.join([dir_assets, 'BT_Beacon_Color_RGB.png'])
im = Image.open(path_image)

st.set_page_config(
    page_title="Butler/Till",
    page_icon=im,
    layout="wide",
)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


#
# Function get_config
#

def get_config():
    r"""Read the configuration file.

    Parameters
    ----------
    None

    Returns
    -------
    specs : dict
        The parameters for controlling the application.

    """

    logger.info('*'*80)
    logger.info("Configuration")

    # Read the configuration file

    full_path = '/'.join(['.', 'config.yml'])
    with open(full_path, 'r') as ymlfile:
        specs = yaml.load(ymlfile, Loader=yaml.FullLoader)

    #
    # Log the configuration parameters
    #

    logger.info('CONFIGURATION PARAMETERS:')
    for spec in specs.keys():
        logger.info('%s: %s', spec, specs[spec])

    # Configuration Specifications
    return specs

#
# Snowflake Connection
#

def connect_backend():
    # Get the initial MMM table
    table_tag = 'mmm'
    table_name = st.session_state.config_specs['tables'][table_tag]
    table_path = dir_tables + '/' + table_tag + '.csv'
    # Read the initial table
    connect_flag = False
    if connect_flag:       
        # Connect to Snowflake
        snowflake_specs = st.session_state.config_specs['snowflake']
        snowflake_specs = connect_snowflake(snowflake_specs)
        # Load the table from Snowflake
        df_table = load_table(snowflake_specs, table_name)
        # Store the table
        df_table.to_csv(table_path, index=False)
    else:
        # Load the table from the cache
        df_table = pd.read_csv(table_path)
    # Save the table in session state
    st.session_state.table_dict[table_tag] = df_table
    # Set the session state
    st.session_state.connected = True

#
# Initialize session state
#

if "connected" not in st.session_state:
    st.session_state.connected = False

if "config_specs" not in st.session_state:
    # Get the configuration parameters
    st.session_state.config_specs = get_config()
    # Create global table dictionary
    st.session_state.table_dict = {}
    # Connect (remove this later)
    connect_backend()


#
# Open the simulator logo
#

use_btsim = True
if use_btsim:
    path_image = ''.join([dir_assets, 'BT_SIM_Color_RGB.png'])
    im = Image.open(path_image)
else:
    path_image = ''.join([dir_assets, 'BT_Logo_Horz_Color_RGB.png'])
    im = Image.open(path_image)

#
# Display the simulator logo in one column and the connect button in the other
#

user_authentication = False
if user_authentication:
    col_h1, col_h2 = st.columns((1, 8))
    col_h1.image(path_image, width=120)

    connect = col_h2.button('**Connect**',
                            on_click=connect_backend,
                            help='Connect to Snowflake to run any models.',
                            key='connect_button')
else:
    st.image(path_image, width=120)

#
# Display the banner at the top of the app
#

path_image = ''.join([dir_assets, 'BT_Color_Strip.png'])
imc = Image.open(path_image)
st.image(imc, use_column_width=True)

#
# Define the Butler/Till Color Palette
#

bt_color_darkest = '#052133'
bt_color_dark_blue = '#083F61'
bt_color_blue = '#0090EB'
bt_color_orange = '#F58426'
bt_color_orange_light = '#FBB034'
bt_color_gray = '#F2F2F0'
bt_color_white = '#FFFFFF'

#
# Helper Functions
#

#
# Function highlight_columns
#

def highlight_columns(df, color, columns_to_shadow=[], columns_to_show=[]):
    highlight = lambda slice_of_df: 'background-color: %s' % color
    if len(columns_to_show) != 0:
        df = df[columns_to_show]
    df.style.applymap(highlight, subset=pd.IndexSlice[:, columns_to_shadow])
    return df

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
    f = "%Y_%m_%d_%H_%M"
    dtstamp = d.strftime(f)
    return dtstamp

#
# Function add_jitter
#

def add_jitter(arr, amount=0.01):
    """Add jitter for certain plots."""
    return arr + amount * np.random.randn(len(arr))

#
# Function extract_date_parts
#

def extract_date_parts(date_dt):
    # Convert to datetime if necessary
    if type(date_dt) == str:
        date_dt = pd.to_datetime(date_dt)
    # Extract year, month and day
    year = date_dt.year
    month = date_dt.month
    day = date_dt.day
    return year, month, day

#
# Function ensure_dir
#

def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

#
# Function get_plan_names
#

def get_plan_names():
    # Get the list of plan names
    plan_names = [pdir.split('.')[0] for pdir in os.listdir(dir_plans)]
    plan_names = sorted(plan_names)
    return plan_names

#
# Function load_plan
#

def load_plan(df_edit):
    # Reset any optimization session state
    st.session_state.optimization = False
    # Extract scenario name from state
    plan_name = st.session_state['load_plan_name']
    # Get the original scenarios
    plan_names = get_plan_names()
    # Load the plan
    load_path = dir_plans + plan_name + '/plan_defaults.csv'
    try:
        df_edit = pd.read_csv(load_path)
    except:
        st.error(f'Plan {plan_name} not found.')
    # Update table
    df_edit = update_table(df_edit)
    # Set session state
    st.session_state['df_scenario'] = df_edit
    return df_edit

#
# Function save_plan
#

def save_plan(df_edit):
    # Extract plan name from state
    plan_name = st.session_state['save_plan_name']
    # Save the plan
    if not plan_name:
        dts = datetime_stamp()
        plan_name = 'Plan_' + dts
    ensure_dir(dir_plans + plan_name)
    save_path = dir_plans + plan_name + '/plan_defaults.csv'
    df_edit.to_csv(save_path, index=False)
    # Set session state
    st.session_state['df_scenario'] = df_edit
    return df_edit

#
# Function add_channel
#

def add_channel(df_edit):
    # Extract date state
    start_date = st.session_state['start_date_form']
    end_date = st.session_state['end_date_form']
    # Define new channel data
    new_channel_data = {
        "Channel": st.session_state['channel_name_form'],
        "Start Date": pd.to_datetime(start_date),
        "End Date": pd.to_datetime(end_date),
        "Live Days": 0,
        "Dark Days": 0,
        "Min": st.session_state['budget_min'],
        "Budget": st.session_state['budget_form'],
        "Max": st.session_state['budget_max'],
        "Est Conversions": 0,
        "Est CPC": 0.00,
    }
    df_new_channel = pd.DataFrame([new_channel_data])
    # Get dataframe components
    df_organic = df_edit[df_edit["Channel"] == "Organic/Other"]
    df_total = df_edit[df_edit["Channel"] == "Total"]
    df_tail = pd.concat([df_organic, df_total], ignore_index=True)
    # Append new channel data and sort by channel name
    df_head = df_edit[(df_edit["Channel"] != "Organic/Other") & (df_edit["Channel"] != "Total")]
    df_head = pd.concat([df_head, df_new_channel], ignore_index=True)
    df_head.sort_values(by=['Channel'], inplace=True)
    # Concatenate the dataframes
    df_edit = pd.concat([df_head, df_tail], ignore_index=True)
    # Reset index
    df_edit.reset_index(drop=True, inplace=True)
    # calculate live and dark days
    df_edit = compute_live_days(df_edit)
    # Update table
    df_edit = update_table(df_edit)
    # Set session state
    st.session_state['df_scenario'] = df_edit
    return df_edit

#
# Function remove_channels
#

def remove_channels(df_edit):
    # Extract channels from state
    channels = st.session_state['channels_form']
    # Append new channel data to df_edit
    df_edit = df_edit[~df_edit["Channel"].isin(channels)]
    # Reset index
    df_edit.reset_index(drop=True, inplace=True)
    # Update table
    df_edit = update_table(df_edit)
    # Set session state
    st.session_state['df_scenario'] = df_edit
    return df_edit

#
# Function compute_live_days
#

def compute_live_days(df):
    for index, row in df.iterrows():
        row_start_date = row['Start Date']
        row_end_date = row['End Date']
        year_start, month_start, day_start = extract_date_parts(row_start_date)
        year_end, month_end, day_end = extract_date_parts(row_end_date)
        rday_first = calendrical.gdate_to_rdate(year_start, month_start, day_start)
        rday_last = calendrical.gdate_to_rdate(year_end, month_end, day_end)
        rdays = []
        for this_year in range(year_start, year_end + 1):
            holiday_dict = calendrical.set_holidays(this_year)
            rdays.extend(holiday_dict.values())
        rdays = [rday for rday in rdays if rday_first <= rday <= rday_last]
        interval_days = calendrical.subtract_dates(year_start, month_start, day_start,
                                                   year_end, month_end, day_end) + 1
        row_live_days = interval_days - len(rdays)
        row_dark_days = interval_days - row_live_days
        df.loc[index, 'Live Days'] = row_live_days
        df.loc[index, 'Dark Days'] = row_dark_days
    return df

#
# Function change_date
#

def change_date(df_edit):
    # Reset the optimization session state
    st.session_state.optimization = False
    # Extract dates
    start_date = st.session_state['start_date']
    end_date = st.session_state['end_date']
    # Set the dates
    df_edit['Start Date'] = start_date
    df_edit['End Date'] = end_date
    # Calculate live and dark days
    df_edit = compute_live_days(df_edit)
    # Set the new state
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date
    return df_edit

#
# Function update_budget_total
#

def update_budget_total(df):
    budget_cols = ['Min', 'Budget', 'Max']
    for bc in budget_cols:
        print(df[bc])
        df[bc] = df[bc].replace('[\$,]', '', regex=True).astype(int)
        total_budget = df[df["Channel"] != "Total"][bc].sum()
        df.loc[df["Channel"] == "Total", bc] = total_budget
        df[bc] = '$' + df[bc].astype(str)
    return df

#
# Function post_results
#

def post_results(df_edit, df_scenarios, df_channels, scenario_number=0):
    # Extract per-channel estimates for the specified scenario from df_channels
    channel_data = df_channels[df_channels['Scenario'] == scenario_number]
    
    # Extract total estimates for the specified scenario from df_scenarios
    est_conversions_total = df_scenarios.loc[scenario_number, 'Est Conv']
    est_cpc_total = df_scenarios.loc[scenario_number, 'Est CPC']
    total_budget = df_scenarios.loc[scenario_number, 'Budget']

    for _, row in channel_data.iterrows():
        channel = row['Channel']
        mask = df_edit['Channel'] == channel
        
        channel_budget = row['Budget']
        budget_fraction = channel_budget / total_budget if total_budget != 0 else 0
        
        est_conv = row['Est Conv'] 
        if est_conv <= 0:
            est_conv = est_conversions_total * budget_fraction
        
        est_cpc = row['Est CPC'] 
        if est_cpc <= 0 or np.isinf(est_cpc):
            est_cpc = est_cpc_total
        
        df_edit.loc[mask, 'Est Conv'] = est_conv

        if channel == 'Organic/Other':
            df_edit.loc[mask, 'Est CPC'] = 0
        else:
            df_edit.loc[mask, 'Est CPC'] = est_cpc

    # Update the "Total" row in df_edit with overall estimates
    mask_total = df_edit['Channel'] == 'Total'
    df_edit.loc[mask_total, 'Est Conv'] = est_conversions_total
    df_edit.loc[mask_total, 'Est CPC'] = est_cpc_total

    temp_table = False
    if temp_table:
        data = {
            "Channel": ["Affiliate", "Display", "Paid Search", "Paid Social", "Print", "TV/Video", "Organic/Other", "Total"],
            "Start Date": ["1/1/2023"] * 8,
            "End Date": ["1/31/2023"] * 8,
            "Live Days": [355] * 8,
            "Dark Days": [11] * 8,
            "Budget": ["$20,000", "$5,000", "$250,000", "$100,000", "$10,000", "$1,000,000", "$0", "$1,385,000"],
            "Est Conv": [1285, 946, 3445, 2060, 643, 6966, 4888, 20233],
            "Est CPC": ["$15.56", "$5.29", "$72.56", "$48.54", "$15.56", "$143.55", "$0.00", "$68.45"]
        }
        df_edit = pd.DataFrame(data)

    # Set session state
    st.session_state['df_scenario'] = df_edit

    return df_edit

#
# Function update_table
#

def update_table(df_edit):
    # Get table state and update the dataframe
    table_state = st.session_state['scenario_table']['edited_rows']
    for index, kv_dict in table_state.items():
        key = list(kv_dict.keys())[0]
        value = list(kv_dict.values())[0]
        df_edit.at[index, key] = value
    # calculate live and dark days
    df_edit = compute_live_days(df_edit)
    # update budget total
    df_edit = update_budget_total(df_edit)
    # Update the table state
    st.session_state['scenario_table']['edited_rows'] = {}
    # Return dataframe
    return df_edit

#
# Function change_state
#

def change_state(df_edit):
    # Reset from any optimization session state
    st.session_state.optimization = False
    # Update Table
    df_edit = update_table(df_edit)
    # Reset session state
    st.session_state['df_scenario'] = df_edit
    # Return dataframe
    return df_edit

#
# Function optimize
#

def optimize(df_edit, target):
    # Run Models and Generate Optimization Curve
    print('Running Models')
    if st.session_state.connected:
        # Run the Models
        df_run = df_edit[df_edit['Channel'] != 'Total'].copy()
        df_run = df_run[['Channel', 'Start Date', 'End Date', 'Budget']]
        df_scenarios, df_channels = run_models(df_run,
                                               st.session_state.model_dict,
                                               dir_models,
                                               st.session_state.table_dict['mmm'],
                                               target)
        st.session_state.df_scenarios = df_scenarios
        # Post the output
        df_edit = post_results(df_edit, df_scenarios, df_channels)
        # Set session state for optimization
        st.session_state.optimization = True
    else:
        st.error('Please click the Connect Button to run any models.')
    # Return dataframe
    return df_edit

#
# Function get_unique_dates
#

def get_unique_dates(df):
    all_dates = []
    for _, row in df.iterrows():
        date_range = pd.date_range(start=row['Start Date'], end=row['End Date'])
        all_dates.extend(date_range)
    unique_dates = pd.Series(all_dates).unique()
    return unique_dates

#
# Function get_dark_periods
#

def get_dark_periods(plan_name, channel_name, dark_path):
    # try opening dark_periods.csv
    try:
        df_dark = pd.read_csv(dark_path, parse_dates=['Start Date', 'End Date'])
        df_dark = df_dark[(df_dark['Channel'] == 'All Channels') | (df_dark['Channel'] == channel_name)]
    except FileNotFoundError:
        print(f"Creating Dark Periods File for Plan {plan_name}")
        df_dark = pd.DataFrame(columns=['Channel', 'Start Date', 'End Date'])
        ensure_dir(dir_plans + plan_name)
    # Return dataframe
    return df_dark

#
# Function add_dark_period
#

def add_dark_period(plan_name, channels):
    # Extract dates
    dp_start_date = pd.to_datetime(st.session_state['dp_start_date_form'])
    dp_end_date = pd.to_datetime(st.session_state['dp_end_date_form'])
    # get dark periods for this plan
    dark_path = dir_plans + plan_name + '/dark_periods.csv'
    df_dark = get_dark_periods(plan_name, channels, dark_path)
    # Add dark period entry to dataframe
    new_row = {'Channel': channels, 'Start Date': dp_start_date, 'End Date': dp_end_date}
    df_dark.loc[len(df_dark)] = new_row
    df_dark.reset_index(drop=True, inplace=True)
    df_dark.drop_duplicates(inplace=True)
    # Save new dataframe
    df_dark.to_csv(dark_path, index=False)
    return

#
# Function remove_dark_period
#

def remove_dark_period(df_dark):
    # Find the index of the selected period
    selected_idx = df_dark[df_dark['period'] == st.session_state['selected_dark_period']].index.item()
    # Remove the selected dark period
    df_dark = df_dark.drop(selected_idx)
    df_dark.drop(columns=['period'], inplace=True)
    df_dark.reset_index(drop=True, inplace=True)
    # Save the updated DataFrame
    df_dark.to_csv(dark_path, index=False)

#
# Function convert_df
#

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


#
# Debug session state
#

debug_session_state = False
if debug_session_state:
    print(f"Session State: {st.session_state}")

#
# Get the models
#

if "model_dict" not in st.session_state:
    # Load the models
    st.session_state.model_dict = load_models(path_models)
    logger.info(f"Model Dictionary: {st.session_state.model_dict}")
    # Define the channels that are available
    st.session_state.channels = [
        'Affiliate',
        'Display',
        'Paid Search',
        'Paid Social',
        'Print',
        'TV/Video',
        'Organic/Other'
    ]

#
# Initialize the session state variables
#

if "df_scenario" not in st.session_state:
    file_spec = ''.join([dir_plans, dir_plan_default, 'plan_defaults.csv'])
    df = pd.read_csv(file_spec)
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['End Date'] = pd.to_datetime(df['End Date'])
    # set the session state variable
    st.session_state.df_scenario = df

if "df_scenarios" not in st.session_state:
    # read in the original scenarios
    file_spec = ''.join([dir_plans, dir_plan_default, 'scenarios.csv'])
    df_scenarios = pd.read_csv(file_spec)
    df_scenarios['Scenario'] = df_scenarios['Scenario'].astype(str)
    df_scenarios['Est CPC'] = df_scenarios['Est CPC'].replace('[\$,]', '', regex=True).astype(float)
    # set the session state variable
    st.session_state.df_scenarios = df_scenarios

if "start_date" not in st.session_state:
    st.session_state.start_date = st.session_state.df_scenario['Start Date'].min()

if "end_date" not in st.session_state:
    st.session_state.end_date = st.session_state.df_scenario['End Date'].max()

if "optimization" not in st.session_state:
    st.session_state.optimization = False


#
# Start app layout
#

col1, col2 = st.columns((8, 5))

with col1:
    # Set the current table state first
    df_edit = st.session_state['df_scenario']
    # Set the number of columns
    col11, col12, col13, col14, col15, col16 = st.columns(6)
    # Add two date pickers to the first column
    start_date = col11.date_input('Start Date',
                                  format="MM/DD/YYYY",
                                  on_change=change_date,
                                  args=(df_edit,),
                                  help='Set the start date for all channels.',
                                  key='start_date')
    end_date = col12.date_input('End Date',
                                format="MM/DD/YYYY",
                                on_change=change_date,
                                args=(df_edit,),
                                help='Set the end date for all channels.',
                                key='end_date')
    # Load Scenario
    button_load_plan = col13.button('**Load Plan**',
                                    on_click=load_plan,
                                    args=(df_edit,),
                                    help='Load a saved plan from the database.',
                                    key='load_plan_button')
    load_plan_name = col13.selectbox('Plan Name', get_plan_names(),
                                     label_visibility="collapsed",
                                     key='load_plan_name',
                                     help='Select the plan to load.')
    # Add and Remove Channels
    button_add_channel = col14.button(':heavy_plus_sign: **Channels**',
                                      help='Add a channel for this plan.')
    button_remove_channel = col14.button(':heavy_minus_sign: **Channels**',
                                         help='Remove channels from this plan.')
    # Save Scenario
    button_save_plan = col15.button('**Save Plan**',
                                    on_click=save_plan,
                                    args=(df_edit,),
                                    help='Save this plan to the database.',
                                    key='save_plan_button')
    plan_dt = 'Plan_' + datetime.now().strftime('%Y%m%d')
    save_plan_name = col15.text_input('Plan Name', value=plan_dt,
                                      label_visibility="collapsed",
                                      key='save_plan_name',
                                      help='Enter a name for the plan to save.')
    # Download the table
    df_csv = convert_df(df_edit)
    col16.download_button(
        "Export Plan as CSV",
        df_csv,
        "plan_table.csv",
        "text/csv",
        key='download_plan_table'
    )
    # Check button state for adding a channel
    if button_add_channel:
        # Channels not in st.session_state.channels
        df_subset = df_edit[df_edit['Channel'] != 'Total']
        available_channels = [channel for channel in st.session_state.channels if channel not in df_subset['Channel'].unique()]
        if len(available_channels) == 0:
            st.error('All available channels have been added.')
        else:
            with st.form("form_add_channel", clear_on_submit=True):
                col141, col142, col143, col144, col145, col146 = st.columns(6)
                channel_name_form = col141.selectbox('Channel Name',
                    options=available_channels,                                  
                    key='channel_name_form')
                start_date_form = col142.date_input('Start Date',
                    format="MM/DD/YYYY", value=datetime.now(), key='start_date_form')
                end_date_form = col143.date_input('End Date',
                    format="MM/DD/YYYY", value=datetime.now(), key='end_date_form')
                budget_form = col144.number_input('Budget', min_value=1000,
                    step=1000, key='budget_form')
                budget_min = col145.number_input('Budget Min', min_value=1000,
                    step=1000, key='budget_min')
                budget_max = col146.number_input('Budget Max', min_value=1000,
                    step=1000, key='budget_max')
                st.form_submit_button("Submit", on_click=add_channel, args=(df_edit,))
    # Check button state for removing a channel
    if button_remove_channel:
        with st.form("form_remove_channel", clear_on_submit=True):
            df_subset = df_edit[df_edit['Channel'] != 'Total']
            st.multiselect('Channels to Remove', df_subset['Channel'].unique(),
                           key='channels_form')
            st.form_submit_button("Submit", on_click=remove_channels, args=(df_edit,))
    # Ensure the date columns are in the right format
    df_edit['Start Date'] = pd.to_datetime(df_edit['Start Date'])
    df_edit['End Date'] = pd.to_datetime(df_edit['End Date'])
    # Edit the table
    df_edit = st.data_editor(
        df_edit,
        column_config={
            "Channel": st.column_config.TextColumn(
                help="To add a channel, click the Add Channel Button.",
            ),
            "Start Date": st.column_config.DateColumn(
                help="Customize the specific Start Date for any Channel(s).",
                min_value=date(2020, 1, 1),
                max_value=date(2030, 12, 31),
                format="MM/DD/YYYY",
            ),
            "End Date": st.column_config.DateColumn(
                help="Customize the specific End Date for any Channel(s).",
                min_value=date(2020, 1, 1),
                max_value=date(2030, 12, 31),
                format="MM/DD/YYYY",
            ),
            "Live Days": st.column_config.NumberColumn(
                help="Live Days is automatically calculated from the date range.",
                min_value=1,
                width="small",
            ),
            "Dark Days": st.column_config.NumberColumn(
                help="Dark Days is automatically calculated from the date range.",
                min_value=1,
                width="small",
            ),
            "Min": st.column_config.NumberColumn(
                help="Enter the Minimum Planned Budget.",
                min_value=1,
                step="1",
            ),
            "Budget": st.column_config.NumberColumn(
                help="Enter the Maximum Planned Budget.",
                min_value=1,
                step="1",
            ),
            "Max": st.column_config.NumberColumn(
                help="Enter an Estimate of the Planned Budget.",
                min_value=1,
                step="1",
            ),
            "Est Conv": st.column_config.NumberColumn(
                help="Run Optimization to generate Est Conversions.",
                min_value=0,
                format="%d",
            ),
            "Est CPC": st.column_config.NumberColumn(
                help="Run Optimization to generate Estimated Cost Per Click.",
                min_value=0.00,
                format="$ %.2f",
            )
        },
        num_rows="fixed",
        disabled=['Channel', 'Live Days', 'Dark Days', 'Est Conv', 'Est CPC'],
        on_change=change_state,
        args=(df_edit,),
        use_container_width=True,
        hide_index=True,
        key="scenario_table")
    # Set the number of columns
    col17, col18 = st.columns((1, 2))
    # Run Optimization
    target = col17.selectbox('Optimization Target',
                             ['Budget', 'Est CPC', 'Est Conv'],
                             key='optimization_target')
    run_optimization = col18.button('**Run Optimization**&nbsp;&nbsp;:arrow_forward:',
                                    on_click=optimize,
                                    args=(df_edit, target),
                                    key='optimize_button',
                                    help='Run the optimization model for all scenarios.')
    # Check if the optimization has been run
    if st.session_state.optimization:
        # Plot the Optimization Curve
        fig = go.Figure()
        # Get the latest scenarios data
        df_scenarios = st.session_state.df_scenarios
        # Set x and y values
        x = df_scenarios['Est CPC']
        y = df_scenarios['Est Conv']
        text = df_scenarios['Scenario']
        # Apply jitter to Est CPC and Est Conv
        jitter_amount = 0.01
        x = add_jitter(df_scenarios['Est CPC'], jitter_amount)
        y = add_jitter(df_scenarios['Est Conv'], jitter_amount)
        # Scale 'Budget' column to suitable range for marker size (e.g., 5-15)
        scaler = MinMaxScaler(feature_range=(5, 15))
        df_scenarios['Budget'] = df_scenarios['Budget'].replace({'\$': '', ',': ''}, regex=True).astype(float)
        marker_size = 10 * df_scenarios['Budget'] / df_scenarios['Budget'].max()
        # Create a color scale
        color_scale = df_scenarios['Budget']
        # Create the figure
        fig.add_trace(go.Scatter(
            x=x, 
            y=y, 
            text=text, 
            mode='markers', 
            hovertemplate='%{text}', 
            marker=dict(
                size=marker_size,
                color=color_scale, # set color to scaled_budget
                colorscale = [
                    [0, bt_color_orange_light],
                    [0.333, bt_color_orange],
                    [0.667, bt_color_blue],
                    [1, bt_color_dark_blue],
                ],
                opacity = 0.6,
                reversescale=False, # reverse to have high values as blue
                colorbar=dict(
                    title="Budget",
                    y=1.0,
                    x=0.4,           # Position the colorbar's center at the middle of the x-axis
                    len=0.8,         # Adjust the length of the colorbar as needed
                    thickness=20,    # Make the colorbar thicker
                    orientation="h"  # Set the orientation to horizontal
                ),
            ),
            name='Scenario'
        ))
        # Set ordering
        fig.update_layout(
            height=420,
            xaxis=dict(title='Est CPC', title_standoff=50),
            yaxis=dict(title='Est Conversions'),
            legend=dict(y=1.1, x=0.5, xanchor="center", yanchor="top", orientation="h"),
            title={"text": "Optimization Plot by Budget Scenarios", "x": 0.5,
                   "xanchor": "center", "font": {"size": 22}},
            title_font=dict(color=bt_color_darkest)
        )
        # Show the plot
        st.plotly_chart(fig, use_container_width=True)

with col2:
    # Make a copy of the input dataframe
    df_ref = st.session_state['df_scenario'].copy()
    # get holidays for the entire calendar
    cal_holiday = USFederalHolidayCalendar()
    cal_holiday = cal_holiday.holidays(start=live_calendar_start_date, end=live_calendar_end_date)
    # Set the heading
    st.markdown('##### Channel Live Days and Dark Days')
    # Set the number of columns
    col21, col22 = st.columns((3, 2))
    # Add an 'All Channels' option to the select box
    selected_channel = col21.selectbox('Select Channel Calendar', ['All Channels'] + df_ref['Channel'].unique().tolist())
    # If 'All Channels' is selected, use the original logic
    if selected_channel == 'All Channels':
        min_date = pd.to_datetime(df_ref['Start Date'].min())
        max_date = pd.to_datetime(df_ref['End Date'].max())
        title_text = "All Channels Calendar"
    else:
        # Filter the DataFrame based on selected channel and get the date range
        df_ref = df_ref[df_ref['Channel'] == selected_channel]
        min_date = pd.to_datetime(df_ref['Start Date'].min())
        max_date = pd.to_datetime(df_ref['End Date'].max())
        title_text = f"{selected_channel} Calendar"
    # Add a dark period
    button_add_dark_period = col22.button(':heavy_plus_sign: **Add Dark Period**',
                                          help='Add a dark period for all partners for selected channel(s).')
    # Check button state for adding a dark period
    if button_add_dark_period:
        with st.form("form_add_dark_period", clear_on_submit=True):
            col221, col222 = st.columns(2)
            dp_start_date_form = col221.date_input('Dark Period Start',
                format="MM/DD/YYYY", value=datetime.now(), key='dp_start_date_form')
            dp_end_date_form = col222.date_input('Dark Period End',
                format="MM/DD/YYYY", value=datetime.now(), key='dp_end_date_form')
            st.form_submit_button("Submit", on_click=add_dark_period,
                                  args=(load_plan_name, selected_channel))
    # Remove a dark period
    button_remove_dark_period = col22.button(':heavy_minus_sign: **Remove Dark Period**',
                                             help='Remove a dark period for all partners for selected channel(s).')

    if button_remove_dark_period:
        # Load the current dark periods for the selected channel
        dark_path = dir_plans + load_plan_name + '/dark_periods.csv'
        df_dark = get_dark_periods(load_plan_name, selected_channel, dark_path)
        # Create a selection box with current dark periods
        if not df_dark.empty:
            df_dark['period'] = df_dark['Start Date'].astype(str) + ' to ' + df_dark['End Date'].astype(str)
            dark_period_options = df_dark['period'].tolist()
            dark_period_options.insert(0, 'Select a Dark Period to Remove')
            selected_period = st.selectbox('Select Dark Period to Remove',
                                           dark_period_options,
                                           on_change=remove_dark_period,
                                           args=(df_dark,),
                                           key='selected_dark_period')
        else:
            st.write('No dark periods available to remove.')
    # Create the calendar
    df_cal = pd.DataFrame({
        "ds": pd.date_range(live_calendar_start_date, live_calendar_end_date),
        "value": 0
        })
    df_cal['ds'] = pd.to_datetime(df_cal['ds'])
    df_cal['value'] = np.where((df_cal['ds'] >= min_date) & (df_cal['ds'] <= max_date), 1, 0)
    df_cal['value'] = np.where(df_cal['ds'].isin(cal_holiday), 2, df_cal['value'])
    # Fill in dark periods
    dark_path = dir_plans + load_plan_name + '/dark_periods.csv'
    df_dark = get_dark_periods(load_plan_name, selected_channel, dark_path)
    unique_dates = get_unique_dates(df_dark)
    df_cal['value'] = np.where(df_cal['ds'].isin(unique_dates), 3, df_cal['value'])
    # Plot the Live/Dark Day Calendar
    fig = go.Figure()
    # choosing a standard colorscale
    fig = calplot(
        df_cal,
        x="ds",
        y="value",
        years_title=True,
        space_between_plots=0.3,
        gap=0,
        colorscale = [
            [0, bt_color_gray],
            [0.333, bt_color_orange_light],
            [0.667, bt_color_blue],
            [1, bt_color_dark_blue],
        ],
        month_lines_width=2, 
        month_lines_color="#fff"
    )
    # Set layout attributes
    fig.update_layout(
        height=400,
        title={"text": title_text, "x": 0.5, "xanchor": "center",
               "font": {"size": 16}},
        title_font=dict(color=bt_color_darkest),
        margin=dict(t=60),
    )
    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
    # Show the optimization table
    if st.session_state.optimization:
        # Drop the duplicate Budget column
        df_scenarios = df_scenarios.drop(columns=['Organic/Other'])
        # Download the table
        df_csv = convert_df(df_scenarios)
        st.download_button(
            "Export Optimization Table as CSV",
            df_csv,
            "optimization_table.csv",
            "text/csv",
            key='download_optimization_table'
        )
        # Show the table
        st.dataframe(df_scenarios, hide_index=True)
