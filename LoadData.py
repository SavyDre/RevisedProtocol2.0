from pathlib import Path as FilePath
import pandas as pd
import warnings
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
        Load CSV or Excel data and apply the existing datetime processing.

        Parameters
        ----------
        Path : str, pathlib.Path, or None
            Input file path. Missing or blank paths return an empty DataFrame.
        DateTimes : list
            Datetime column names. The first two identify the start and end.
        StartTime, EndTime : datetime
            Analysis time boundaries passed to modify_datetimes().
        Format : str
            Datetime format passed to modify_datetimes().

        Returns
        -------
        pandas.DataFrame
            Loaded and filtered data, or an empty DataFrame when the path is
            absent, the file does not exist, or the file contains no records.

        Notes
        -----
        Unsupported formats, permission errors, malformed nonempty files,
        and datetime conversion errors remain visible to the caller.
    """
    # An omitted path means no data were supplied for this input.
    if Path is None or not str(Path).strip():
        return pd.DataFrame()

    file_path = FilePath(Path).expanduser()

    if not file_path.is_file():
        print(f"File not found: {file_path}. Returning an empty DataFrame.")
        return pd.DataFrame()

    try:
        # Check file size without reading binary Excel files as text.
        if file_path.stat().st_size == 0:
            print(f"File is empty: {file_path}")
            return pd.DataFrame()

        # Use the actual extension; lowercase also accepts .CSV and .XLSX.
        extension = file_path.suffix.lower()
        DataDF = selectfiletype(file_path, extension)

    except FileNotFoundError:
        # Handle a file removed between the existence check and reading.
        print(f"File no longer exists: {file_path}")
        return pd.DataFrame()

    except pd.errors.EmptyDataError:
        # Includes CSV files containing only whitespace or no usable columns.
        print(f"File contains no readable data: {file_path}")
        return pd.DataFrame()

    # Preserve column headers when a file has headers but no records.
    if DataDF.empty:
        return DataDF

    DataDF = modify_datetimes(df=DataDF, DateTimes=DateTimes, StartTime=StartTime, EndTime=EndTime, Format=Format)
    print(f"Loaded {len(DataDF)} records within the analysis window.")
    return DataDF