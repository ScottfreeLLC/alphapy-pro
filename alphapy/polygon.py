#
# Imports
#

from datetime import datetime
import json
import pandas as pd
import requests


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

    print(f"Connecting to {url}")
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
# Function get_polygon_data
#

def get_polygon_data(symbol, from_date, to_date, time_frame, period):
    r"""Get Polygon daily and intraday data.

    Parameters
    ----------
    symbol : str
        A valid stock symbol.
    from_date : str
        Starting date for symbol retrieval.
    to_date : str
        Ending date for symbol retrieval.
    time_frame : str
        Pandas offset alias.
    period : int
        The number of periods for the time frame.

    Returns
    -------
    df : pandas.DataFrame
        The dataframe containing the intraday data.

    """

    time_frames = ['minute', 'hour', 'day', 'week', 'month', 'quarter', 'year']
    if time_frame not in time_frames:
        print(f"Invalid time frame: {time_frame}")
        print(f"Select one of: {time_frames}")
        quit()

    # Google requires upper-case symbol, otherwise not found
    symbol = symbol.upper()

    # Initialize data frame
    df = pd.DataFrame()

    #
    # Compose the request to Polygon
    #
    # Example:
    #
    # https://api.polygon.io/v2/aggs/ticker/AAPL/range/5/minute/2023-01-09/2023-05-09
    # ?adjusted=true&sort=asc&limit=120&apiKey=_BynHqDfXhPoQcFf8Nb6hJzC_p67_5Sf1tn5ms
    #

    base_url = 'https://api.polygon.io/v2/aggs'
    ticker = '/'.join(['ticker', symbol])
    fractal = '/'.join(['range', str(period), time_frame])
    limit = '='.join(['limit', str(50000)])
    api_key = '='.join(['apiKey', '_BynHqDfXhPoQcFf8Nb6hJzC_p67_5Sf1tn5ms'])
    modifiers = '&'.join(['?adjusted=true&sort=asc', limit, api_key])

    done = False
    dfs = []
    to_date_dt = datetime.strptime(to_date, '%Y-%m-%d')

    start_date = from_date
    while not done:
        date_range = '/'.join([start_date, to_date])
        # Make the request
        url = '/'.join([base_url, ticker, fractal, date_range, modifiers])
        response = get_web_content(url)
        json_data = json.loads(response)
        # Create the data frame and rename columns
        df = pd.DataFrame(json_data['results'])
        df.drop(columns=['vw', 'n'], inplace=True)
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={
            'v': 'volume',
            'o' : 'open',
            'c' : 'close',
            'h' : 'high',
            'l' : 'low',
            't' : 'datetime'
        })
        cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        df = df[cols]
        # check the last date to see if we are done
        last_date = df['datetime'].iloc[-1]
        n_days = abs(last_date - to_date_dt).days
        if n_days <= 3:
            done = True
        else:
            df = df[df['datetime'] < last_date]
            start_date = last_date.strftime('%Y-%m-%d')
        # add the dataframe to the list of dataframes
        dfs.append(df)
    # concatenate all of the dataframes
    df = pd.concat(dfs)

    # Return the dataframe
    return df


#
# Try some examples
#

# AAPL 1-minute data
# df_appl = get_polygon_data('AAPL', '2020-01-01', '2020-12-31', 'minute', 1)
# print(df_appl)

# AAPL 5-minute data
# df_appl = get_polygon_data('AAPL', '2020-01-01', '2020-12-31', 'minute', 5)
# print(df_appl)

# AAPL 1-hour data
df_appl = get_polygon_data('HOFV', '2018-01-01', '2023-06-01', 'hour', 1)
print(df_appl)

# AAPL daily data
# df_appl = get_polygon_data('AAPL', '2020-01-01', '2020-12-31', 'day', 1)
# print(df_appl)