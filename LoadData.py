import os
import pytz
import datetime
import pandas as pd
from datetime import datetime, timezone

def selectfiletype(dataFilePath, opExt):
    """
    Load a file into a pandas DataFrame based on its extension.

    Parameters:
    - dataFilePath (str): The file path to load.
    - opExt (str): The file extension indicating the file type (e.g., '.csv', '.xlsx').

    Returns:
    - pandas.DataFrame: The loaded DataFrame.

    Raises:
    - Exception: If the file extension is not supported.
    """
    try:
        # Handle .xlsx files using the openpyxl engine
        if opExt == '.xlsx':
            opDF = pd.read_excel(dataFilePath, engine='openpyxl')

        # Handle .csv files
        elif opExt == '.csv':
            opDF = pd.read_csv(dataFilePath)

        # Handle .xls files using the xlrd engine
        elif opExt == '.xls':
            opDF = pd.read_excel(dataFilePath, engine='xlrd')

        # Raise an exception for unsupported file types
        else:
            raise Exception(f"Unable to parse file. Unsupported extension: {opExt}")

        return opDF

    except Exception as e:
        print(f"Error loading file: {dataFilePath}. Details: {e}")
        raise


def convert_time(time_str, position, Format='%Y%m%d %H:%M:%S'):
    try:
        # Split the input string by '/' and select the segment at the specified position
        part = time_str.split('/')[position]

        # Convert the ISO 8601 formatted timestamp to a datetime object
        # Replace 'Z' with '+00:00' to make it compatible with fromisoformat
        dt = datetime.fromisoformat(part.replace('Z', '+00:00'))

        # Format the datetime object to a string and then parse it again to strip microseconds etc.
        dt = datetime.strptime(dt.strftime(Format), Format)

        # Return the datetime object with UTC timezone information explicitly set
        return dt.replace(tzinfo=timezone.utc)

    except Exception as e:
        # Handle any exceptions that occur (e.g., invalid position, bad format) and print a helpful message
        print(f"Time conversion error ({position}): {e}")
        return None


def modify_datetimes(df, DateTimes, StartTime, EndTime, Format):
    cols = df.columns.tolist()

    if 'StartEnd' in df.columns:
        # Create new columns
        df['StartTime'] = df['StartEnd'].apply(lambda x: convert_time(x, position=0, Format=Format))
        df['EndTime'] = df['StartEnd'].apply(lambda x: convert_time(x, position=1, Format=Format))

        # Get the position of StartEnd column
        startend_pos = df.columns.get_loc('StartEnd')

        # Reorder columns to insert new ones after StartEnd
        cols.insert(startend_pos + 1, 'StartTime')
        cols.insert(startend_pos + 1, 'EndTime')
        df = df[cols]
        # Filter out data for datetimes within the analysis start and end datetimes
        filt = (df['StartTime'] >= StartTime) & (df['EndTime'] <= EndTime)
        df = df.loc[filt]

    else:
        for Time in DateTimes:
            if Time in cols:  # Check if the specified column exists
                df[str(Time)] = df[str(Time)].apply(lambda time: datetime.strptime(time, Format).replace(tzinfo=timezone.utc))

        # Filter out data for datetimes within the analysis start and end datetimes
        filt = (df[DateTimes[0]] >= StartTime) & (df[DateTimes[1]] <= EndTime)
        df = df.loc[filt]

    return df

def loadData(Path, DateTimes, StartTime, EndTime, Format):
    """
    Load data from a file at the specified path, ensuring the file exists, is non-empty,
    and can be loaded into a pandas DataFrame. Additionally, specified datetime columns
    are converted to datetime objects.

    Parameters:
    - Path (str): The file path to load the data from.
    - DateTimes (list): A list of column names in the DataFrame that should be converted to datetime.
      The first 2 elements should contain column names of startdatetime and enddatetimes
    - StartTime: Analysis start datetime in UTC
    - EndTime: Analysis end datetime in UTC
    - Format (str): The format of the datetime.

    Returns:
    - DataDF (pandas.DataFrame): A DataFrame containing the loaded data, or an empty DataFrame
                                 if the file is not found, is empty, or contains invalid data.
    """

    # Initialize an empty DataFrame
    if Path is not None:
        try:
            # Check if the file exists at the given path
            if not os.path.exists(Path):
                print(f"File not found: {Path}")
                DataDF = pd.DataFrame()  # Return an empty DataFrame if file is not found
            else:
                # Open and read the file to check for content
                with open(Path, 'r') as file:
                    content = file.read().strip()  # Remove leading/trailing whitespace and newlines
                # If the file is empty, return an empty DataFrame
                if not content:
                    print("The file is empty.")
                    DataDF = pd.DataFrame()
                else:
                    # File has content, attempt to load into a DataFrame
                    DataDF = selectfiletype(Path, '.csv')  # Function to handle specific file types
                    print("File successfully loaded into a DataFrame.")
        # Handle cases where the file cannot be parsed into a DataFrame
        except pd.errors.EmptyDataError:
            print("The file is invalid or contains no parsable data.")
            DataDF = pd.DataFrame()
    else:
        # If no path is provided, return an empty DataFrame
        DataDF = pd.DataFrame()

    # If the DataFrame is not empty, convert specified columns to datetime
    if not DataDF.empty:
        DataDF = modify_datetimes(df=DataDF, DateTimes=DateTimes, StartTime=StartTime, EndTime=EndTime, Format=Format)

    return DataDF
