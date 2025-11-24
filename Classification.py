import pytz
import os
import math
import pandas as pd
import numpy as np
import time
import itertools
import random
from datetime import datetime, timedelta
from geopy.distance import geodesic
from shapely import wkt
from shapely.geometry import Polygon


def polygonEdgesDistanceFromRefCoord(polygon_wkt, ref_point):
    """
    Calculates the distance from a reference point to each vertex of an XY polygon.

    Parameters:
    - polygon_wkt (str): WKT format of the polygon (e.g., "POLYGON((lon1 lat1, lon2 lat2, ..., lonN latN))").
    - ref_point (tuple): Reference point as (latitude, longitude).

    Returns:
    - list: Distances from the reference point to each vertex of the polygon in meters.
    """

    # Parse the polygon from WKT format
    polygon = wkt.loads(polygon_wkt)

    # Ensure the polygon is valid
    if not polygon.is_valid or polygon.is_empty:
        raise ValueError("Invalid polygon WKT format")

    # Calculate distances from reference point to each vertex of the polygon
    distances = np.array([geodesic(ref_point, (lat, lon)).meters for lon, lat in polygon.exterior.coords])

    return distances

def calc_bearing(cord1, cord2):
    """
    Calculate the bearing between two geographical coordinates.

    Parameters:
        - param cord1: Coordinates of the first point as (latitude, longitude) in degrees.
        - param cord2: Coordinates of the second point as (latitude, longitude) in degrees.
    return
        - Bearing in degrees from point1 (start) to point2 (end) (clockwise from North).
    """
    # Convert latitudes and longitudes from degrees to radians
    lat1, lon1 = map(math.radians, cord1)
    lat2, lon2 = map(math.radians, cord2)

    # Calculate the difference in longitudes
    dlon = lon2 - lon1

    # Calculate x and y components for the bearing
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    # Calculate the initial bearing in radians
    initial_bearing = math.atan2(y, x)

    # Convert bearing from radians to degrees
    bearing = math.degrees(initial_bearing)

    # Normalize the bearing to a range of 0 to 360 degrees
    return (bearing + 360) % 360


def polygon_intersection_extent(reference_coord, ASE_polygon, subASE_polygon, BoundingZASE_min, BoundingZASE_max,
                                BoundingZsubASE_min, BoundingZsubASE_max, ASE_bufferDistance=0, subASE_bufferDistance=0):
    """
    Evaluate the degree of overlap between two polygons formed by distances from a reference coordinate.

    Parameters:
    - reference_coord (tuple): The reference coordinate as (latitude, longitude).
    - ASE_polygon (str): WKT format string representing polygon ASE.
    - subASE_polygon (str): WKT format string representing polygon subASE.
    - ASE_bufferDistance (float): Buffer to apply to ASE distances (in meters).
    - subASE_bufferDistance (float): Buffer to apply to subASE distances (in meters).
    - BoundingZASE_min (float): Minimum height boundary for the ASE polygon.
    - BoundingZASE_max (float): Maximum height boundary for the ASE polygon.
    - BoundingZsubASE_min (float): Minimum height boundary for the subASE polygon.
    - BoundingZsubASE_max (float): Maximum height boundary for the subASE polygon

    Returns:
    - float: Degree of overlap (intersection area / union area).
    """
    try:
        # Parse WKT polygons
        ASE_polygon = wkt.loads(ASE_polygon)
        subASE_polygon = wkt.loads(subASE_polygon)

        # Calculate distances and bearings for ASE
        ASE_distances = [geodesic(reference_coord, (lat, lon)).meters + float(ASE_bufferDistance) for lat, lon in list(ASE_polygon.exterior.coords)]
        ASE_bearings = [math.radians(calc_bearing(cord1=reference_coord, cord2=(lat, lon))) for lat, lon in list(ASE_polygon.exterior.coords)]

        # Calculate distances and bearings for subASE
        subASE_distances = [geodesic(reference_coord, (lat, lon)).meters + float(subASE_bufferDistance) for lat, lon in list(subASE_polygon.exterior.coords)]
        subASE_bearings = [math.radians(calc_bearing(cord1=reference_coord, cord2=(lat, lon))) for lat, lon in list(subASE_polygon.exterior.coords)]

        # Convert distances and bearings to Cartesian coordinates
        coords_cartesian_ASE = [(r * math.sin(theta), r * math.cos(theta)) for r, theta in zip(ASE_distances, ASE_bearings)]
        coords_cartesian_subASE = [(r * math.sin(theta), r * math.cos(theta)) for r, theta in zip(subASE_distances, subASE_bearings)]

        # Create polygons using the calculated Cartesian coordinates
        polygon_cartesian_ASE = Polygon(coords_cartesian_ASE)
        polygon_cartesian_subASE = Polygon(coords_cartesian_subASE)

        # Calculate intersection and union areas
        polygons_intersect = polygon_cartesian_ASE.intersection(polygon_cartesian_subASE)

        # Check if the height ranges overlap
        height_overlap = (BoundingZsubASE_min >= BoundingZASE_min) and (BoundingZsubASE_max <= BoundingZASE_max)
        if polygons_intersect and height_overlap:
            # Calculate intersection and union areas
            intersection_area = polygons_intersect.area
            union_area = polygon_cartesian_ASE.union(polygon_cartesian_subASE).area
            # Calculate degree of overlap
            overlap_degree = (intersection_area / union_area) if union_area > 0 else 0.0
            return overlap_degree

        return 0.0  # Return 0 if no intersection or height bounds don't overlap

    except Exception as e:
        print(f"Error: {e}")
        return 0.0


def extractSpatialSubsetData(DataDF, DataDF_PolygonCoord, DataDF_BoundingZMin, DataDF_BoundingZMax, DataDF_localization,
                             ASE_PolygonCoord_Dict, ASE_textual_mapper, overLappingFrac=0, ASE_bufferDistance=0,
                             subASE_bufferDistance=0):
    """
    Filters a DataFrame to return rows that fall within a specified spatial bounding area.

    Parameters:
    - DataDF (DataFrame): The input DataFrame containing spatial data to be filtered.
    - DataDF_PolygonCoord (str): Column name in DataDF containing WKT format polygons.
    - DataDF_BoundingZMin (str): Column name in DataDF specifying minimum height of each entry.
    - DataDF_BoundingZMax (str): Column name in DataDF specifying maximum height of each entry.
    - DataDF_localization (str): Column name in DataDF for identifiers used if polygon data is absent.
    - ASE_PolygonCoord_Dict (dict): Dictionary with "BoundingXY" for the polygon (WKT format) and "BoundingZ" for height bounds.
    - ASE_textual_mapper (dict): Mapping of DataDF_localization identifiers to spatial properties (polygon and height).
    For example, ASE_PolygonCoord_Dict = {"IT-1": {"BoundingXY": "POLYGON((lat_1 lon_1, ..., lat_n lon_n))",
                                         "BoundingZ": ["MinimumZ", "MaximumZ"], "ReferenceCoord": ["lat", "lon"]}}
    - overLappingFrac (float): Minimum intersection fraction required for filtering (default is 0).
    - ASE_bufferDistance (float): The buffer distance for the ASE polygon (meters)
    - subASE_bufferDistance (float): The buffer distance for the subASE polygon (meters)
    -
    Returns:
    - DataFrame: A filtered subset of DataDF where rows intersect the ASE bounding area by the specified fraction.
    """
    # Ensure DataDF is a copy
    DataDF = DataDF.copy(deep=True)

    # Retrieve bounding polygon and height range from ASE_PolygonCoord_Dict
    ASE_BoundingXY = ASE_PolygonCoord_Dict["BoundingXY"]
    ASE_ReferenceCoord = ASE_PolygonCoord_Dict["ReferenceCoord"]
    ASE_BoundingZMinimum = ASE_PolygonCoord_Dict["BoundingZ"][0]
    ASE_BoundingZMaximum = ASE_PolygonCoord_Dict["BoundingZ"][1]

    # Case 1: If DataDF contains polygon coordinates, apply spatial filtering based on polygons
    if DataDF_PolygonCoord in DataDF.columns:
        # Calculate the intersection extent for each row's polygon with the ASE bounding polygon
        DataDF["ASE"] = DataDF.apply(lambda row: polygon_intersection_extent(
                                     reference_coord=ASE_ReferenceCoord,
                                     ASE_polygon=ASE_BoundingXY,
                                     BoundingZASE_min=ASE_BoundingZMinimum,
                                     BoundingZASE_max=ASE_BoundingZMaximum,
                                     subASE_polygon=row[DataDF_PolygonCoord],
                                     BoundingZsubASE_min=row[DataDF_BoundingZMin],
                                     BoundingZsubASE_max=row[DataDF_BoundingZMax],
                                     ASE_bufferDistance=ASE_bufferDistance,
                                     subASE_bufferDistance=subASE_bufferDistance),
                                     axis=1)


        # Filter rows where intersection extent is above the specified threshold
        Data4thisASE = DataDF.loc[DataDF["ASE"] > overLappingFrac]
        # Drop the temporary 'ASE' column used for filtering
        Data4thisASE = Data4thisASE.drop("ASE", axis=1)

    # Case 2: If DataDF contains localization data, use textual mapping for spatial filtering
    elif DataDF_localization in DataDF.columns:
        # Calculate the intersection extent using the textual mapper's bounding polygons and heights
        DataDF["ASE"] = DataDF.apply(lambda row: polygon_intersection_extent(
                                     reference_coord=ASE_ReferenceCoord,
                                     ASE_polygon=ASE_BoundingXY,
                                     BoundingZASE_min=ASE_BoundingZMinimum,
                                     BoundingZASE_max=ASE_BoundingZMaximum,
                                     subASE_polygon=ASE_textual_mapper[row[DataDF_localization]]["BoundingXY"],
                                     BoundingZsubASE_min=ASE_textual_mapper[row[DataDF_localization]]["BoundingZ"][0],
                                     BoundingZsubASE_max=ASE_textual_mapper[row[DataDF_localization]]["BoundingZ"][1],
                                     ASE_bufferDistance=ASE_bufferDistance,
                                     subASE_bufferDistance=subASE_bufferDistance),
                                     axis=1)

        # Filter rows based on intersection extent threshold
        Data4thisASE = DataDF.loc[DataDF["ASE"] > overLappingFrac]
        # Remove the 'ASE' column used for filtering
        Data4thisASE = Data4thisASE.drop("ASE", axis=1)

    # Case 3: If no spatial data is present, return a copy of the entire DataFrame without filtering
    else:
        Data4thisASE = DataDF.copy()

    return Data4thisASE

def convert_timezone(dt, from_tz, to_tz='UTC'):
    """
    Converts a datetime object from one timezone to another.

    Parameters:
    - dt (datetime): The datetime to be converted.
    - from_tz (str): The name of the original timezone of the input datetime (e.g., 'America/New_York').
    - to_tz (str): The target timezone to convert to. Defaults to 'UTC'.

    Returns:
    - datetime: The converted datetime in the target timezone.
    """
    # Create timezone objects based on the input strings
    from_timezone = pytz.timezone(from_tz)
    to_timezone = pytz.timezone(to_tz)

    # Localize the datetime to 'from_tz' timezone if it is naive (lacks timezone info).
    if dt.tzinfo is None:
        # Localize the naive datetime to 'from_tz'
        localized_dt = from_timezone.localize(dt)
    else:
        # Adjust dt to 'from_tz' if it already has timezone info
        localized_dt = dt.astimezone(from_timezone)

    # Convert the datetime to the target timezone, taking daylight savings into account
    target_dt = localized_dt.astimezone(to_timezone)

    return target_dt


def roundDatetime(dt, freq):
    """
    Rounds up a datetime object to the nearest time unit associated with the specified frequency.
    Parameters
        - dt : The datetime to be rounded.
        - freq : Frequency string (e.g., 's', 'min', '5min', '10min', '15min', '30min', 'h', 'D', etc).
    Returns
        - datetime: The rounded-up datetime.
    """

    # Map frequency strings to seconds
    step = pd.Timedelta(freq).total_seconds()

    # Convert datetime to timestamp (seconds since epoch)
    ts = dt.timestamp()

    # Divide by step and round to nearest bin
    rounded_ts = round(ts / step) * step

    # Return rounded datetime
    return datetime.utcfromtimestamp(rounded_ts)


def timerange2Datetimes(time_range, from_tz, to_tz='UTC', format="%Y-%m-%dT%H:%M:%SZ"):
    """
    Parses a time range, converts each datetime to a specified timezone, and calculates duration in seconds.

    Parameters:
    - time_range (str): A string in the format "start_time/end_time", with each time as "yyyy-mm-ddThh:mm:ssZ".
    - from_tz (str): The timezone of the input times.
    - to_tz (str): The target timezone to convert to. Defaults to 'UTC'.
    - format (str): The format of the input datetime strings. Default is "yyyy-mm-ddThh:mm:ssZ".

    Returns:
    - tuple: (start_time_dt, end_time_dt, duration_seconds)
      - start_time_dt: The start time converted to the target timezone.
      - end_time_dt: The end time converted to the target timezone.
      - duration_seconds: The duration in seconds between start and end times.
    """
    # Split the time range string to get start and end time strings
    start_time_str, end_time_str = time_range.split('/')

    # Parse the start and end time strings into datetime objects using the specified format
    start_time_dt = datetime.strptime(start_time_str, format)
    end_time_dt = datetime.strptime(end_time_str, format)

    # Convert each datetime to the target timezone using convert_timezone function
    start_time_dt = convert_timezone(dt=start_time_dt, from_tz=from_tz, to_tz=to_tz)
    end_time_dt = convert_timezone(dt=end_time_dt, from_tz=from_tz, to_tz=to_tz)

    # Calculate the duration in seconds between the converted start and end times
    duration_seconds = (end_time_dt - start_time_dt).total_seconds()

    return start_time_dt, end_time_dt, duration_seconds


def expand_Date_intervals(variables, df, startTime, endTime, freq, inclusive):
    """
    Expands data between startTime and endTime with the specified frequency.

    Parameters:
        - df (pd.DataFrame): The input dataframe containing startTime and endTime.
        - startTime (str): Column name for the start time.
        - endTime (str): Column name for the end time.
        - freq (str): Frequency for expanding the time intervals (e.g., '1H', '30T').
        - variables: Additional columns to replicate in the expanded dataframe.
        - inclusive: The side of time to include during timestep breakdown.
    Returns:
        - expanded_df (pd.DataFrame): The expanded dataframe with rows replicated based on time intervals.
    """

    expanded_rows = []  # List to store expanded rows

    # Iterate over the dataframe using itertuples for better performance
    for row in df.itertuples(index=False):
        start = getattr(row, startTime)
        end = getattr(row, endTime)
        time_range = pd.date_range(start=start, end=end, freq=freq, inclusive=inclusive)
        # Prebuild the additional columns that should be replicated (from variables)
        additional_data = {str(var): getattr(row, str(var)) for var in variables}
        # For each time point, create a new dictionary with the time and other column values
        for time_point in time_range:
            # Create a dictionary for each time point, adding the time and additional columns
            row_data = {'Time': time_point, str(startTime): getattr(row, str(startTime)), str(endTime): getattr(row, str(endTime))}
            row_data.update(additional_data)  # Add the additional columns
            expanded_rows.append(row_data)  # Append the new row to expanded_rows

    # Convert the list of expanded rows into a DataFrame
    expanded_df = pd.DataFrame(expanded_rows)

    # Handle empty case to avoid crash
    if expanded_df.empty or "Time" not in expanded_df.columns:
        print("No Valid time intervals were found. Skipping expansion.")
        return pd.DataFrame(columns=["Time"]).set_index("Time")

    # Set the "Time" column as the index
    expanded_df = expanded_df.set_index("Time")

    return expanded_df

class DataTimeOperations:
    def __init__(self, DataDF, freq, Time):
        """
        Initializes the DataTimeOperations class with the DataFrame, time frequency, and the time column.
        Parameters:
            - DataDF (DataFrame): The input DataFrame containing the data to operate on.
            - freq (str): The frequency for resampling (e.g., 'D', 'H', etc.).
            - Time (str): The column name that contains the time data.
        """
        self.DataDF = DataDF
        self.freq = freq
        self.Time = Time
        self.start_time = self.DataDF[self.Time].min()
        self.end_time = self.DataDF[self.Time].max()

    def validate_columns(self, columns):
        """
        Validates whether the provided columns exist in the DataFrame.
        Raises an error if any column is not found.
        """
        missing_columns = [col for col in columns if col not in self.DataDF.columns]
        if missing_columns:
            raise ValueError(f"The following columns are not present in the DataFrame: {missing_columns}")

    def reindex_data(self, df):
        """
        Reindexes the DataFrame to cover the full time range and handles missing values.
        """
        return df.set_index(self.Time).reindex(pd.date_range(self.start_time, self.end_time, freq=self.freq)).fillna(0).reset_index()

    def sum_data_variables(self, listOfVariables2Sum):
        """
        Resamples the data and calculates the sum of the specified variables over the given frequency.
        """
        self.validate_columns(listOfVariables2Sum)
        agg_dict = {var: 'sum' for var in listOfVariables2Sum}
        summedDF = self.DataDF.resample(self.freq, on=self.Time).agg(agg_dict).reset_index()
        summedDF = self.reindex_data(summedDF)
        # Readjust and ensure the column headers are in proper locations
        summedDF.columns = [self.Time] + listOfVariables2Sum
        return summedDF

    def average_data_variables(self, listOfVariables2Average):
        """
        Resamples the data and calculates the average of the specified variables over the given frequency.
        """
        self.validate_columns(listOfVariables2Average)
        agg_dict = {var: 'mean' for var in listOfVariables2Average}
        averagedDF = self.DataDF.resample(self.freq, on=self.Time).agg(agg_dict).reset_index()
        averagedDF = self.reindex_data(averagedDF)
        # Readjust and ensure the column headers are in proper locations
        averagedDF.columns = [self.Time] + listOfVariables2Average
        return averagedDF

    def listUnique_data_variables(self, listOfVariables2ListUnique):
        """
        Resamples the data and returns a list of unique values for the specified variables over the given frequency.
        """
        self.validate_columns(listOfVariables2ListUnique)
        uniqueListDF = self.DataDF.resample(self.freq, on=self.Time).agg({var: lambda x: x.unique().tolist() for var in listOfVariables2ListUnique}).reset_index()
        uniqueListDF = self.reindex_data(uniqueListDF)
        # Readjust and ensure the column headers are in proper locations
        uniqueListDF.columns = [self.Time] + listOfVariables2ListUnique
        return uniqueListDF

    def max_data_variables(self, listOfVariables2Max):
        """
        Resamples the data and calculates the maximum value of the specified variables over the given frequency.
        """
        self.validate_columns(listOfVariables2Max)
        agg_dict = {var: 'max' for var in listOfVariables2Max}
        maxDF = self.DataDF.resample(self.freq, on=self.Time).agg(agg_dict).reset_index()
        maxDF = self.reindex_data(maxDF)
        # Readjust and ensure the column headers are in proper locations
        maxDF.columns = [self.Time] + listOfVariables2Max
        return maxDF

    def min_data_variables(self, listOfVariables2Min):
        """
        Resamples the data and calculates the minimum value of the specified variables over the given frequency.
        """
        self.validate_columns(listOfVariables2Min)
        agg_dict = {var: 'min' for var in listOfVariables2Min}
        minDF = self.DataDF.resample(self.freq, on=self.Time).agg(agg_dict).reset_index()
        minDF = self.reindex_data(minDF)
        # Readjust and ensure the column headers are in proper locations
        minDF.columns = [self.Time] + listOfVariables2Min
        return minDF


def add_cumulative_duration(df, rate_col, freq):
    """
    Adds a column for cumulative duration where cumulative duration only counts for non-zero rates.
    Resets whenever a zero value is encountered in the 'rate_col' column.
    Adjustments:
    1. Resets cumulative duration to start at 0.
    2. Subtracts 1 from the cumulative duration, except for the first value in each group.

    Parameters:
        - df: Input DataFrame with time series data (index should be datetime-like).
        - rate_col: Column name for the emission rate.
        - freq: Frequency of the time intervals (e.g., 'T' for minute, 'H' for hour).

    Returns:
        - DataFrame with an additional 'Cumulative Duration' column.
    """
    # Validate if DataFrame index is datetime-like
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("The DataFrame index must be datetime-like.")

    # Validate if the rate column exists
    if rate_col not in df.columns:
        raise KeyError(f"Column '{rate_col}' does not exist in the DataFrame.")

    # Ensure 'freq' is valid and prepend '1' if it's a single letter
    if freq.isalpha():
        freq = '1' + freq
    try:
        pd.Timedelta(freq)  # Validate the frequency
    except ValueError as e:
        raise ValueError(f"Invalid frequency '{freq}': {e}")

    # Sort the DataFrame by its index
    df = df.sort_index()

    # Calculate time differences in specified frequency units
    df['time_diffs'] = pd.Timedelta(freq).total_seconds()

    # Identify non-zero rate segments
    rate_nonzero = df[rate_col] > 0

    # Calculate cumulative duration for non-zero segments
    df['Cumulative Duration'] = 0  # Initialize column
    #df.loc[rate_nonzero, 'Cumulative Duration'] = (rate_nonzero[rate_nonzero].groupby((rate_nonzero != rate_nonzero.shift()).cumsum()).transform(lambda g: ((g.cumsum()).clip(lower=0)) * df.loc[g.index, 'time_diffs']))
    df.loc[rate_nonzero, 'Cumulative Duration'] = (rate_nonzero[rate_nonzero].groupby(rate_nonzero.ne(rate_nonzero.shift()).cumsum()).transform(lambda g: ((g.cumsum()).clip(lower=0)) * df.loc[g.index, 'time_diffs']))

    # Drop helper column 'time_diffs'
    df.drop(columns=['time_diffs'], inplace=True)

    return df


def aggregateTimeSeriesDataByFreq(DataDF, freq, ID_column, min_count, **kwargs):
    """
    Aggregates a time-series DataFrame by a specified frequency.

    Parameters:
    - DataDF (pd.DataFrame): The DataFrame to aggregate, must contain a datetime column.
    - freq (str): The frequency string for resampling (e.g., '1H', 'D', 'T' for minutes).
    - ID_column (str): The column containing unique IDs to determine earliest/latest entries.
    - min_count (int): Minimum number of non-NA values required to perform the sum operation.
    - **kwargs: Variable categories for aggregation:
        - listOfVariables2Sum (list[str]): Variables to sum over the frequency.
        - listOfVariables2Average (list[str]): Variables to calculate the average.
        - listOfVariables2ListUnique (list[str]): Variables to extract unique values as a list.
        - listOfVariables2Max (list[str]): Variables to get the maximum value.
        - listOfVariables2Min (list[str]): Variables to get the minimum value.
        - listOfVariables2ObtainFirst (list[str]): Variables to get the first value based on ID.
        - listOfVariables2ObtainLast (list[str]): Variables to get the last value based on ID.

    Returns:
    - pd.DataFrame: Aggregated DataFrame with resampled data according to the specified operations.
    """

    def extract_time(series, DataDF, var, is_earliest=True):
        """
        Helper function to get a value from the smallest/largest ID in a group.

        Parameters:
        - series (pd.Series): Series containing IDs for the group.
        - DataDF (pd.DataFrame): The original DataFrame.
        - var (str): The column from which to extract the value.
        - is_earliest (bool): If True, selects the smallest/sent first ID. Otherwise, selects the largest/sent last.

        Returns:
        - The value from the specified column corresponding to the earliest/latest ID.
        """
        sorted_series = series.sort_values(ascending=is_earliest)
        vals4cor = DataDF.loc[sorted_series.index, var].tolist()
        return vals4cor[0] if vals4cor else None

    def returnList(series):
        """
        Returns a list of unique, non-null values from a pandas Series.
        If all elements in the Series are null, returns None.
        Parameters:
        - series (pd.Series): Input pandas Series.
        Returns:
        - list or None: List of unique non-null values or None if all values are null.
        """
        # If all elements are null, return None
        if series.isna().all():
            return None
        # Otherwise, return unique non-null values as a list
        return series.dropna().unique().tolist()

    # Ensure 'Time' is a datetime column and set it as the index if not already
    if DataDF.index.name != 'Time':
        DataDF['Time'] = pd.to_datetime(DataDF['Time'], errors='coerce')  # Convert to datetime
        DataDF = DataDF.set_index('Time')  # Set the datetime column as the index

    # Initialize a dictionary to store aggregated results
    aggregated_data = {}

    # Map operations to their corresponding variable lists
    agg_funcs = {
        'sum': kwargs.get('listOfVariables2Sum', []),
        'mean': kwargs.get('listOfVariables2Average', []),
        'unique': kwargs.get('listOfVariables2ListUnique', []),
        'max': kwargs.get('listOfVariables2Max', []),
        'min': kwargs.get('listOfVariables2Min', []),
        'first': kwargs.get('listOfVariables2ObtainFirst', []),
        'last': kwargs.get('listOfVariables2ObtainLast', [])
    }

    # Remove empty lists from aggregation functions
    agg_funcs = {key: val for key, val in agg_funcs.items() if val}

    # Perform aggregation for each operation and variable
    for operation, variables in agg_funcs.items():
        for var in variables:
            # Skip the variable if it doesn't exist in the DataFrame
            if var not in DataDF.columns:
                continue
            # Apply the appropriate aggregation operation
            if operation == 'sum':
                # Sum values, ensuring at least `min_count` non-NA values
                print(f"Performing summing operation for the column header: {var}")
                aggregated_data[f'{var}'] = DataDF[var].resample(rule=freq).sum(min_count=min_count)
                print("Successfully completed summing operation")

            elif operation == 'mean':
                # Calculate mean values
                print(f"Performing averaging operation for the column header: {var}")
                aggregated_data[f'{var}'] = DataDF[var].resample(rule=freq).mean()
                print("Successfully completed the averaging operation")

            elif operation == 'unique':
                # Extract unique values as a list
                print(f"Performing the extraction of the unique value operation for the column header: {var}")
                aggregated_data[f'{var}'] = DataDF[var].resample(rule=freq).apply(lambda x: returnList(x))
                print("Successfully completed the extraction of unique value operation")

            elif operation == 'max':
                # Get maximum value
                print(f"Performing the extraction of the maximum value operation for the column header: {var}")
                aggregated_data[f'{var}'] = DataDF[var].resample(rule=freq).max()
                print("Successfully completed the extraction of the maximum value operation")

            elif operation == 'min':
                # Get minimum value
                print(f"Performing the extraction of the minimum value operation for the column header: {var}")
                aggregated_data[f'{var}'] = DataDF[var].resample(rule=freq).min()
                print("Successfully completed the extraction of the minimum value operation")

            elif operation == 'first':
                # Get the first value based on ID
                print(f"Performing the extraction of the first value operation for the column header: {var} using {ID_column} as unique ID.")
                DataDF[ID_column] = DataDF[ID_column].apply(lambda id: int(id))
                aggregated_data[f'{var}'] = DataDF[ID_column].resample(rule=freq).apply(
                    lambda series: extract_time(series=series, DataDF=DataDF, var=var, is_earliest=True))
                print("Successfully completed the extraction of the first value operation")

            elif operation == 'last':
                # Get the last value based on ID
                print(f"Performing the extraction of the last value operation for the column header: {var} using {ID_column} as unique ID.")
                DataDF[ID_column] = DataDF[ID_column].apply(lambda id: int(id))
                aggregated_data[f'{var}'] = DataDF[ID_column].resample(rule=freq).apply(
                    lambda series: extract_time(series=series, DataDF=DataDF, var=var, is_earliest=False))
                print("Successfully completed the extraction of the last value operation")

    # Combine the aggregated results into a single DataFrame
    aggregated_df = pd.DataFrame(aggregated_data)

    # Handle duplicate columns (if any) by keeping the first occurrence
    aggregated_df = aggregated_df.loc[:, ~aggregated_df.columns.duplicated()]

    return aggregated_df

def frequency_to_seconds(freq_str):
    """
    Converts a frequency string (e.g., "1H", "30min", "2D") into its equivalent value in seconds.
    Parameters:
        - freq_str (str): A string representing a time duration in a format recognized by pandas.Timedelta.
                        Examples include "1H" (1 hour), "30min" (30 minutes), "2D" (2 days), and "1W" (1 week).
    Returns:
        - float: The total number of seconds corresponding to the input frequency.
    Raises:
        ValueError: If the input string is not a valid frequency format.
    """
    # Create a Timedelta object from the frequency string.
    # This parses the input into a pandas object representing the duration.
    timedelta = pd.Timedelta(freq_str)
    # Convert the Timedelta object into the total duration in seconds as a float.
    return timedelta.total_seconds()

class ThresholdingTimeSeriesData:

    def __init__(self, TimeSeriesData):
        self.TimeSeriesData = TimeSeriesData

    def DefaultRollingAvg(self, baselineEmissionRate, value_column, RollingAverageWindow, RollingAverageThreshold,
                          ThresholdColumn, freq="min"):
        """
        Apply the default (as it is supposed to work in literature) rolling/moving average thresholding method.
        In this approach, the moving mean with a window span of f"{RollingAverageWindow}{freq}" e.g 15mins if
        RollingAverageWindow = 15 and freq = min is evaluated and compared to a thresholding value: RollingAverageThreshold.
        Note that the index of the dataframe must be set to datetimes for the code to work

        Parameters
            - baselineEmissionRate: The baseline emission rate
            - value_column: The column on which to calculate the rolling average
            - RollingAverageWindow: The size of the rolling window (i.e., int)
            - RollingAverageThreshold: The threshold for the rolling average
            - ThresholdColumn: The column name for the thresholding result
            - freq: The frequency of the rolling operation (default is "T" for minutes)
        return
            - Dataframe: The updated DataFrame with a new column 'Satisfied Threshold'
        """

        # Total rate threshold to compare against
        Total = baselineEmissionRate + RollingAverageThreshold
        rolling_avg_column = f"Rolling_Avg_{RollingAverageWindow}{freq}" # i.e., Rolling_Avg_15min if RollingAverageWindow = 15 and freq = min
        # Calculate the rolling average over the given window size and frequency
        self.TimeSeriesData[rolling_avg_column] = self.TimeSeriesData[value_column].rolling(f'{RollingAverageWindow}{freq}').mean()
        # Compare rolling averages to the threshold
        self.TimeSeriesData[str(ThresholdColumn)] = self.TimeSeriesData[rolling_avg_column].apply(lambda x: None if pd.isna(x) else (1 if x > Total else 0))
        return self.TimeSeriesData

    def AverageOfRandomSample(self, baselineEmissionRate, RollingAverageThreshold, releaserate, ThresholdColumn):
        """
        This method evaluates the mean of random samples of data equivalent to the specified RollingAverageWindow.
        parameter
            - baselineEmissionRate: The baseline emission rate to compare against.
            - RollingAverageThreshold: The threshold value added to the baseline for comparison.
            - releaserate: The column of release rate.
            - ThresholdColumn: The name of the column to store the thresholding result.
        return:
            - A dataframe.
        """
        # Calculate the total threshold to compare the mean against.
        Total = baselineEmissionRate + RollingAverageThreshold
        # Apply threshold condition using both rate and cumulative duration
        self.TimeSeriesData[str(ThresholdColumn)] = self.TimeSeriesData.apply(lambda row: None if pd.isna(row[releaserate]) else (1 if row[releaserate] > Total else 0), axis=1)
        return self.TimeSeriesData

    def AboveThresholdAcrossCumulativeDuration(self, baselineEmissionRate, DetectionRateAboveBaseline,
                                               DetectionDurationThreshold, ThresholdColumn, rateColumnName):
        """
        Apply rate and cumulative duration threshold to the time series data. In his method, the values of the
        'Cumulative Duration' column (a column evaluating the cummulative duration at each time step (row) for
        which emission rate is non-zero. The value resets after every row with emission rate equal to zero) is
        compared to DetectionDurationThreshold provided, and the emission rate compared to
        baselineEmissionRate + DetectionRateAboveBaseline supplied.

        parameter
            - baselineEmissionRate: The baseline emission rate
            - DetectionRateAboveBaseline: Threshold rate above the baseline
            - DetectionDurationThreshold: Minimum cumulative duration to satisfy threshold
            - ThresholdColumn: The column name for the thresholding result
            - rateColumnName: The column containing the rate to be compared
        return:
            - The updated DataFrame with a new column 'Satisfied Threshold'
        """
        # Total rate threshold to compare against
        Total = baselineEmissionRate + DetectionRateAboveBaseline
        # Apply threshold condition using both rate and cumulative duration
        self.TimeSeriesData[str(ThresholdColumn)] = self.TimeSeriesData.apply(lambda row: None if pd.isna(row[rateColumnName]) or pd.isna(row['Cumulative Duration']) \
                                                            else (1 if row[rateColumnName] > Total and row['Cumulative Duration'] > DetectionDurationThreshold else 0),
                                                            axis=1)
        return self.TimeSeriesData

def align_time_indexes(crdf, drdf, dr_columns, detection):
    """
    Aligns the indexes of two DataFrames and adds a unique 'Marker' column to each.
    parameter
        - crdf: The first DataFrame (Controlled Release DataFrame).
        - drdf: The second DataFrame (Detection Report DataFrame).
        - dr_columns: The list of drdf columns to assign all zeros to prevent concatenating empty dataframes or dataframes with all-NA entries.
        - detection: Detection column
    return:
        - Dataframes: The updated crdf and drdf with aligned indexes and unique 'Marker' columns.
    """
    # Extract the index array from crdf
    cr_index_array = crdf.index.to_numpy()
    dr_index_array = drdf.index.to_numpy()

    # Identify missing indexes in drdf if any
    missing_indexes = np.setdiff1d(cr_index_array, dr_index_array).tolist()

    # To ensure consistent time format
    missing_indexes = [np.datetime64(ts, 'ns') for ts in missing_indexes]

    # Add a marker to identify appended timesteps
    drdf["OriginalReport"] = "True"

    # Add missing indexes with NaN values
    if len(missing_indexes) > 0:
        # Create a DataFrame with missing indexes and None values
        missing_df = pd.DataFrame(index=missing_indexes, columns=drdf.columns)
        # Assign "" to all elements of the dataframe to avoid an empty dataframe or a dataframe only containing Nan
        missing_df[:] = ""
        # Assign 0 to the specified columns
        for col in dr_columns:
            if col is not None:
                missing_df[col] = 0
        missing_df["dr_Classification"] = None
        missing_df["OriginalReport"] = "False"
        if detection is not None:
            missing_df[detection] = 0
        drdf = pd.concat([drdf, missing_df]).sort_index()

    # Align drdf with the indexes from crdf
    drdf = drdf.loc[cr_index_array]

    return crdf, drdf

def merge_intervals(exclusionDF, start, end):
    """
    Merges overlapping intervals in the exclusionDF.

    Parameters:
    - exclusionDF (pd.DataFrame): DataFrame containing the intervals with 'start' and 'end' columns.
    - start (str): The name of the column in exclusionDF that contains the start times of the intervals.
    - end (str): The name of the column in exclusionDF that contains the end times of the intervals.

    Returns:
    - DataFrame: A DataFrame with merged intervals.
    """
    # Return empty DataFrame if exclusionDF is empty
    if exclusionDF.empty:
        return exclusionDF

    # Sort by the start times
    exclusionDF = exclusionDF.sort_values(by=[start, end]).reset_index(drop=True)

    merged_intervals = []
    current_start = exclusionDF[start].iloc[0]
    current_end = exclusionDF[end].iloc[0]

    for i in range(1, len(exclusionDF)):
        # If intervals overlap or touch, merge them
        if exclusionDF[start].iloc[i] <= current_end:
            current_end = max(current_end, exclusionDF[end].iloc[i])
        else:
            # Add the merged interval
            merged_intervals.append((current_start, current_end))
            current_start = exclusionDF[start].iloc[i]
            current_end = exclusionDF[end].iloc[i]

    # Add the last interval
    merged_intervals.append((current_start, current_end))

    return pd.DataFrame(merged_intervals, columns=[start, end])


def flagData(dataDF, exclusionDF, flag, classification, start_exc, end_exc, freq="1min", ID=None,
             bufferTime=(0, 0)):
    """
    Flags data in a DataFrame based on temporal overlap with exclusion intervals.

    Parameters:
        - dataDF: The DataFrame containing data to be flagged. It must have a datetime index if ID=None.
        - exclusionDF: The DataFrame specifying exclusion intervals.
        - flag: The value to assign to the 'classification' column for flagged rows.
        - classification: The column name in `dataDF` where flags will be applied.
        - start_exc: Column name in `exclusionDF` representing the start of exclusion intervals.
        - end_exc: Column name in `exclusionDF` representing the end of exclusion intervals.
        - ID (str, optional): Column name in `dataDF` for unique identifiers to be flagged based on overlap. If None, the function flags rows based on their time index.
        - freq: Frequency string (e.g., '1min', '10s', '1h') used to interpret bufferTime.
        - bufferTime: A tuple (buffer_start, buffer_end) specifying additional time to extend exclusion intervals before applying flags.

    Returns:
        - DataFrame: The modified DataFrame with flagged rows in the 'classification' column.
    """

    # Infer unit from frequency
    if freq.endswith("s"):
        unit = 's'
    elif freq.endswith("min"):
        unit = 'min'
    else:
        unit = 'h'

    def make_interval(ts):
        """
        Convert a timestamp-like value into a UTC-based Interval.

        Parameters:
            - ts : A timestamp or value convertible to pandas.Timestamp.
        Returns:
            - interval: A closed interval [ts, ts] with timezone-aware UTC datetime.
        """

        # Convert input to a pandas Timestamp object
        ts = pd.Timestamp(ts)

        # Ensure timezone is UTC.
        if ts.tz is None:
            # If the timestamp is naive (no tz info), assign UTC.
            ts = ts.tz_localize("UTC")
        else:
            # If already tz-aware, convert to UTC for consistency.
            ts = ts.tz_convert("UTC")

        # Return a degenerate interval [ts, ts] (start == end)
        return pd.Interval(ts, ts, closed="both")

    if not exclusionDF.empty:
        # Extend exclusion windows
        exclusionDF = exclusionDF.copy()
        exclusionDF[start_exc] = exclusionDF[start_exc] - pd.to_timedelta(bufferTime[0], unit=unit)
        exclusionDF[end_exc] = exclusionDF[end_exc] + pd.to_timedelta(bufferTime[1], unit=unit)

        # Merge overlapping intervals
        exclusionDF = merge_intervals(exclusionDF, start_exc, end_exc)

        # Build IntervalIndex of exclusion windows
        exc_intervals = pd.IntervalIndex.from_arrays(exclusionDF[start_exc], exclusionDF[end_exc], closed="both")

        # Case 1: No ID, just flag by index timestamps
        if ID is None:
            mask = dataDF.index.to_series().apply(lambda ts: exc_intervals.overlaps(make_interval(ts)).any())

        # Case 2: Flag by ID if *any* row with that ID overlaps.
        else:
            # First find IDs with at least one overlap
            overlapping_ids = set(dataDF.loc[dataDF.index.to_series().apply(lambda ts: exc_intervals.overlaps(make_interval(ts)).any()), ID].unique())
            # Mask all rows of those IDs
            mask = dataDF[ID].isin(overlapping_ids)

        dataDF.loc[mask, classification] = flag

    return dataDF


def check_if_dataframe_column_exists(df, column_name):
    """
    Check if a specified column exists in a DataFrame and if it contains any non-NaN values.
    Parameters:
        - df (pd.DataFrame): The DataFrame to check.
        - column_name (str): The name of the column to check for.
    Returns:
        - Boolean (True or False)
    """
    # Check if the column exists
    column_exists = column_name in df.columns

    # Check if the column contains any non-NaN value
    column_exists_and_has_values = column_exists and df[column_name].notna().any()

    return column_exists, column_exists_and_has_values

def samplefromdata(CRDF, DRDF, cr_DateTime, dr_DateTime, nSampling, cr_releaseRate, dr_releaseRate,
                   ThresholdDuration):
    """
    Perform stratified sampling from controlled release and detection report dataframes.

    Parameters:
        - CRDF (pd.DataFrame): Controlled Releases dataframe with datetime ranges
        - DRDF (pd.DataFrame): Detection Reports dataframe with datetime ranges
        - cr_DateTime (list): [start_col, end_col] for controlled releases
        - dr_DateTime (list): [start_col, end_col] for detection reports
        - nSampling (int): Number of sampling iterations to perform
        - cr_releaseRate (str): Column name for controlled release rate values
        - dr_releaseRate (str): Column name for detection report rate values
        - ThresholdDuration (int): Target duration in minutes for sampling periods
    Returns:
        - tuple: Original CRDF, Original DRDF, Sampled CR means, Sampled DR means
    """

    try:
        # Filter unflagged data (where classification is NaN)
        unFlaggedCRDF = CRDF[CRDF["tc_Classification"].isna()].copy()
        unFlaggedDRDF = DRDF[DRDF["dr_Classification"].isna()].copy()

        # Reset index to avoid datetime index issues
        unFlaggedCRDF = unFlaggedCRDF.reset_index(drop=True)
        unFlaggedDRDF = unFlaggedDRDF.reset_index(drop=True)

        # Remove duplicate time ranges
        unFlaggedCRDF = unFlaggedCRDF.drop_duplicates(subset=cr_DateTime)
        unFlaggedDRDF = unFlaggedDRDF.drop_duplicates(subset=dr_DateTime)

        target_duration = ThresholdDuration  # already in time basis

        # Lists to store sampled means
        cr_means = []
        dr_means = []

        for _ in range(nSampling):
            # 1. Randomly sample detection reports
            sampled_dr = []
            total_dr_duration = 0
            # Obtain random sample of rows from the unflagged detection data
            shuffled_dr = unFlaggedDRDF.sample(frac=1, random_state=None)  # Random sampling of detection data

            # Iterate through the shuffled detection report data and select data with cummulative duration is < target_duration
            for idx, row in shuffled_dr.iterrows():
                dr_duration = (row[dr_DateTime[1]] - row[dr_DateTime[0]]).total_seconds()
                if total_dr_duration >= target_duration:
                    break
                sampled_dr.append(idx) # Keep track of the index of the sampled row
                total_dr_duration += dr_duration
            # Get sampled detection reports
            sampledDRDF = unFlaggedDRDF.loc[sampled_dr]
            # Calculate the mean of reported emission rate
            dr_means.append(sampledDRDF[dr_releaseRate].mean())

            # 2. Find overlapping controlled release data using vectorized operations
            cr_start = unFlaggedCRDF[cr_DateTime[0]].to_numpy()
            cr_end = unFlaggedCRDF[cr_DateTime[1]].to_numpy()
            dr_start = sampledDRDF[dr_DateTime[0]].to_numpy()
            dr_end = sampledDRDF[dr_DateTime[1]].to_numpy()
            # Compute overlaps using broadcasting
            overlap_mask = ((cr_start[:, None] <= dr_end) & (cr_end[:, None] >= dr_start)).any(axis=1)
            potentialCRDF = unFlaggedCRDF.loc[overlap_mask]

            # 3. Sample controlled releases
            sampled_cr = []
            total_cr_duration = 0
            # Obtain random sample from the controlled release data with overlapping timestep as the sampled detection report.
            shuffled_cr = potentialCRDF.sample(frac=1, random_state=None)  # Random sampling of controlled release data
            # Iterate through the controlled release data and select data with cummulative duration is < target_duration
            for idx, row in shuffled_cr.iterrows():
                cr_duration = (row[cr_DateTime[1]] - row[cr_DateTime[0]]).total_seconds()
                if total_cr_duration >= target_duration:
                    break
                sampled_cr.append(idx)
                total_cr_duration += cr_duration
            # Get sampled controlled releases
            sampledCRDF = unFlaggedCRDF.loc[sampled_cr]
            # Calculate controlled release mean
            cr_means.append(sampledCRDF[cr_releaseRate].mean())

        # 4. Convert lists to DataFrame
        CRDF_rs = pd.DataFrame({cr_releaseRate: cr_means})
        DRDF_rs = pd.DataFrame({dr_releaseRate: dr_means})

        # Append classification columns
        CRDF_rs["tc_Classification"] = None
        DRDF_rs["dr_Classification"] = None

        return CRDF, DRDF, CRDF_rs, DRDF_rs

    except Exception as e:
        print(f"Error: {e}")
        return CRDF, DRDF, pd.DataFrame(), pd.DataFrame()


def DefaultRollingAvgApplication(DRDF, CRDF, detection, dr_releaseRate, cr_releaseRate, dr_baselineEmissionRate,
                                 tc_baselineEmissionRate, RollingAverageThreshold, RollingAverageWindow, freq,
                                 column_exists, column_exists_and_has_values):
    """
    Applies the DefaultRollingAvgMethod to two datasets: Detection Report Data and Controlled Release Data.

    Parameters
        - DRDF (dataframe): Timeseries detection report data.
        - CRDF (dataframe): Timeseries controlled release data.
        - detection (str): Column name in DRDF indicating detection status.
        - dr_releaseRate (str): Column name representing the release rate in the detection dataset.
        - cr_releaseRate (str): Column name representing the release rate in the controlled release dataset.
        - dr_baselineEmissionRate (float): Baseline emission rate for the detection dataset.
        - tc_baselineEmissionRate (float): Baseline emission rate for the controlled release dataset.
        - RollingAverageWindow (int): Window size (number of time steps) for the rolling average calculation.
        - RollingAverageThreshold (float): Threshold value for the rolling average comparison.
        - freq (str): Frequency of the time intervals used in rolling average computations (e.g., "1H", "min").
        - column_exists, column_exists_and_has_values (True or False): Booleans indicating if DRDF contains detection column.
    return:
        - Two DataFrames, DRDF and CRDF, containing rolling average results and thresholding detection columns.
    """

    # Create an instance of the ThresholdingTimeSeriesData class for detection and controlled data
    ThresholdDetectionReportData = ThresholdingTimeSeriesData(TimeSeriesData=DRDF)
    ThresholdControlledReleaseData = ThresholdingTimeSeriesData(TimeSeriesData=CRDF)

    # Apply DefaultRollingAvgMethod on the controlled release dataset
    CRDF = ThresholdControlledReleaseData.DefaultRollingAvg(value_column=cr_releaseRate,
                                                            baselineEmissionRate=tc_baselineEmissionRate,
                                                            RollingAverageWindow=RollingAverageWindow,
                                                            ThresholdColumn="tc_Detection",
                                                            freq=freq,
                                                            RollingAverageThreshold=RollingAverageThreshold)

    # Apply thresholding to DRDF if detection column does not exist or has no values
    if not column_exists or not column_exists_and_has_values:
        DRDF = ThresholdDetectionReportData.DefaultRollingAvg(value_column=dr_releaseRate,
                                                              baselineEmissionRate=dr_baselineEmissionRate,
                                                              RollingAverageWindow=RollingAverageWindow,
                                                              ThresholdColumn="dr_Detection",
                                                              freq=freq,
                                                              RollingAverageThreshold=RollingAverageThreshold)
    else: # Rename the column and drop the previous one to maintain consistency
        DRDF = DRDF.rename(columns={str(detection): "dr_Detection"})

    return DRDF, CRDF


def AboveThresholdAcrossCumDurationApplication(DRDF, CRDF, detection, dr_baselineEmissionRate, tc_baselineEmissionRate,
                                               dr_releaseRate, cr_releaseRate, DetectionRateThresholdAboveBaseline,
                                               DetectionDurationThreshold, column_exists, column_exists_and_has_values):
    """
    Applies the Cumulative Duration Above Threshold Method to two datasets: Detection Report Data and Controlled Release Data.

    Parameter
        - DRDF (dataframe): Timeseries of detection report data.
        - CRDF (dataframe): Timeseries of controlled release data.
        - detection (str): Column name in DRDF indicating detection status.
        - dr_baselineEmissionRate (float): Baseline emission rate for the detection dataset.
        - tc_baselineEmissionRate (float): Baseline emission rate for the controlled release dataset.
        - dr_releaseRate (str): Column name representing the release rate in the detection dataset.
        - cr_releaseRate (str): Column name representing the release rate in the controlled release dataset.
        - DetectionRateThresholdAboveBaseline (float): Threshold rate above the baseline for detection.
        - DetectionDurationThreshold (int): Minimum duration above the threshold to trigger detection.
        - column_exists, column_exists_and_has_values (True or False): Booleans indicating if DRDF contains detection column.

    return: Two DataFrames:
             - DRDF: Processed detection dataset with cumulative duration results and thresholding.
             - CRDF: Processed controlled release dataset with similar results.
    """

    # Create an instance of the ThresholdingTimeSeriesData class for detection and controlled data
    ThresholdDetectionReportData = ThresholdingTimeSeriesData(TimeSeriesData=DRDF)
    ThresholdControlledReleaseData = ThresholdingTimeSeriesData(TimeSeriesData=CRDF)


    # Apply the method to the controlled release dataset
    CRDF = ThresholdControlledReleaseData.AboveThresholdAcrossCumulativeDuration(baselineEmissionRate=tc_baselineEmissionRate,
                                                                                 DetectionRateAboveBaseline=DetectionRateThresholdAboveBaseline,
                                                                                 DetectionDurationThreshold=DetectionDurationThreshold,
                                                                                 ThresholdColumn="tc_Detection",
                                                                                 rateColumnName=cr_releaseRate)
    # Apply thresholding to DRDF if detection column does not exist or has no values
    if not column_exists or not column_exists_and_has_values:
        DRDF = ThresholdDetectionReportData.AboveThresholdAcrossCumulativeDuration(baselineEmissionRate=dr_baselineEmissionRate,
                                                                                   DetectionRateAboveBaseline=DetectionRateThresholdAboveBaseline,
                                                                                   DetectionDurationThreshold=DetectionDurationThreshold,
                                                                                   ThresholdColumn="dr_Detection",
                                                                                   rateColumnName=dr_releaseRate)
    else: # Rename the column and drop the previous one to maintain consistency
        DRDF = DRDF.rename(columns={str(detection): "dr_Detection"})

    # Return both processed DataFrames
    return DRDF, CRDF


def AverageOfRandomSampleApplication(DRDF, CRDF, dr_baselineEmissionRate, tc_baselineEmissionRate, dr_releaseRate,
                                     cr_releaseRate, RollingAverageThreshold):
    """
    Applies the Random Sampling Rolling Average Method multiple times (bootstrapping) to Detection and Controlled Release Data.
    Parameters:
        - DRDF (dataframe): Timeseries detection report data.
        - CRDF (dataframe): Timeseries controlled release data.
        - dr_baselineEmissionRate (float): Baseline emission rate for the detection dataset.
        - tc_baselineEmissionRate (float): Baseline emission rate for the controlled release dataset.
        - dr_releaseRate (str): Column name representing the release rate in the detection dataset.
        - cr_releaseRate (str): Column name representing the release rate in the controlled release dataset.
        - RollingAverageThreshold (float): Threshold value for the rolling average comparison.

    return: Two DataFrames:
        - CRDF: Results for the controlled release dataset.
        - DRDF: Results for the detection dataset.
    """
    # Create an instance of the ThresholdingTimeSeriesData class for detection and controlled data
    ThresholdDetectionReportData = ThresholdingTimeSeriesData(TimeSeriesData=DRDF)
    ThresholdControlledReleaseData = ThresholdingTimeSeriesData(TimeSeriesData=CRDF)

    rCRDF = ThresholdControlledReleaseData.AverageOfRandomSample(baselineEmissionRate=tc_baselineEmissionRate,
                                                                 RollingAverageThreshold=RollingAverageThreshold,
                                                                 releaserate=cr_releaseRate,
                                                                 ThresholdColumn="tc_Detection")
    rDRDF = ThresholdDetectionReportData.AverageOfRandomSample(baselineEmissionRate=dr_baselineEmissionRate,
                                                               RollingAverageThreshold=RollingAverageThreshold,
                                                               releaserate=dr_releaseRate,
                                                               ThresholdColumn="dr_Detection")
    return rDRDF, rCRDF


def thresholdingApplication(CRDF, DRDF, rs_CRDF, rs_DRDF, detection, cr_releaseRate, dr_releaseRate,
                            ThresholdRateAboveBaseline, ThresholdDuration, tc_baselineEmissionRate,
                            dr_baselineEmissionRate, freq, Method="Rolling Average"):
    """
    Apply thresholding techniques to detection and controlled release data.

    Parameters:
        - CRDF (pd.DataFrame): Timeseries DataFrame containing controlled release data (e.g., test center data).
        - DRDF (pd.DataFrame): Timeseries DataFrame containing detection report data.
        - rs_CRDF (pd.DataFrame): DataFrame containing the mean of resampled controlled release data over ThresholdDuration
        - rs_DRDF (pd.DataFrame): DataFrame containing the mean of resampled detection report data over ThresholdDuration
        - detection (str): Column name in DRDF indicating detection status.
        - cr_releaseRate (str): Column name in CRDF for release rates.
        - dr_releaseRate (str): Column name in DRDF for release rates.
        - tc_baselineEmissionRate: Baseline emission rate as determined by the test center.
        - dr_baselineEmissionRate: Baseline emission rate as determined by the solution/technology.
        - nSampling: Number of sampling simulations
        - ThresholdRateAboveBaseline (float): Detection threshold emission rate above the baseline emission rate.
        - ThresholdDuration (int): Duration with unit specified as freq (i.e., min, sec) used for thresholding.
        - freq: The frequency of temporal resolution (default is "T" for minutes)
        - Method: The thresholding method used i.e., "Rolling Average", "Random Sample Average", "Above Threshold Across Duration", etc.

    Returns:
        - tuple(DataFrame, DataFrame): Updated DRDF and CRDF after applying the thresholding methods.
    """

    # Step 1: Check if the `detection` column exists in DRDF and contains values
    column_exists, column_exists_and_has_values = check_if_dataframe_column_exists(df=DRDF,
                                                                                   column_name=detection)

    # Step 2: Add cumulative duration.
    DRDF = add_cumulative_duration(df=DRDF,
                                   rate_col=dr_releaseRate,
                                   freq=freq)
    CRDF = add_cumulative_duration(df=CRDF,
                                   rate_col=cr_releaseRate,
                                   freq=freq)

    # Step 3: Thresholding controlled release and detection report data
    if Method == "Rolling Average":
        # Thresholding using the default rolling average method.
        DRDF, CRDF = DefaultRollingAvgApplication(DRDF=DRDF,
                                                  CRDF=CRDF,
                                                  detection=detection,
                                                  dr_releaseRate=dr_releaseRate,
                                                  cr_releaseRate=cr_releaseRate,
                                                  dr_baselineEmissionRate=dr_baselineEmissionRate,
                                                  tc_baselineEmissionRate=tc_baselineEmissionRate,
                                                  RollingAverageThreshold=ThresholdRateAboveBaseline,
                                                  RollingAverageWindow=ThresholdDuration,
                                                  freq=freq,
                                                  column_exists=column_exists,
                                                  column_exists_and_has_values=column_exists_and_has_values)

        return DRDF, CRDF

    elif Method == "Random Sample Average":
        # Thresholding by taking the mean of random samples.
        rs_DRDF, rs_CRDF = AverageOfRandomSampleApplication(DRDF=rs_DRDF,
                                                            CRDF=rs_CRDF,
                                                            dr_baselineEmissionRate=dr_baselineEmissionRate,
                                                            tc_baselineEmissionRate=tc_baselineEmissionRate,
                                                            dr_releaseRate=dr_releaseRate,
                                                            cr_releaseRate=cr_releaseRate,
                                                            RollingAverageThreshold=ThresholdRateAboveBaseline)

        return rs_DRDF, rs_CRDF

    elif Method == "Above Threshold Across Duration":
        # Thresholding using above threshold across cumulative duration method.
        DRDF, CRDF = AboveThresholdAcrossCumDurationApplication(DRDF=DRDF,
                                                                CRDF=CRDF,
                                                                detection=detection,
                                                                dr_baselineEmissionRate=dr_baselineEmissionRate,
                                                                tc_baselineEmissionRate=tc_baselineEmissionRate,
                                                                dr_releaseRate=dr_releaseRate,
                                                                cr_releaseRate=cr_releaseRate,
                                                                DetectionRateThresholdAboveBaseline=ThresholdRateAboveBaseline,
                                                                DetectionDurationThreshold=ThresholdDuration,
                                                                column_exists=column_exists,
                                                                column_exists_and_has_values=column_exists_and_has_values)

        return DRDF, CRDF


def compareDataFrames(CRDF, DRDF, tc_Detected="tc_Detection", dr_Detected="dr_Detection",
                      tc_Classification="tc_Classification", dr_Classification="dr_Classification"):
    """
    Compare CRDF and DRDF DataFrames based on detection thresholds and classify rows into categories.

    Parameters:
    - CRDF (pd.DataFrame): DataFrame containing controlled release data.
    - DRDF (pd.DataFrame): DataFrame containing detection report data.
    - tc_Detected (str): Column in CRDF indicating detection status (1 for detected, 0 for not detected, NaN for missing).
    - dr_Detected (str): Column in DRDF indicating detection status (1 for detected, 0 for not detected, NaN for missing).
    - tc_Classification (str): Column in CRDF to store classification results.
    - dr_Classification (str): Column in DRDF to store classification results.

    Returns:
    - pd.DataFrame: A merged DataFrame combining CRDF and DRDF with final classification results.
    """

    # Step 1: Create masks for CRDF detection states
    mask_CRDF_is_1 = CRDF[tc_Detected] > 0  # CRDF detected (1)
    mask_CRDF_is_0 = CRDF[tc_Detected] < 1  # CRDF not detected (0)
    mask_CRDF_is_Nan = CRDF[tc_Detected].isna()  # CRDF missing data (NaN)

    # Step 2: Create masks for DRDF detection states
    mask_DRDF_is_1 = DRDF[dr_Detected] > 0  # DRDF detected (1)
    mask_DRDF_is_0 = DRDF[dr_Detected] < 1  # DRDF not detected (0)
    mask_DRDF_is_Nan = DRDF[dr_Detected].isna()  # DRDF missing data (NaN)

    # Step 3: Create masks for unclassified rows
    mask_TC_Classification_isnan = CRDF[tc_Classification].isna()  # Unclassified rows in CRDF
    mask_Det_Classification_isnan = DRDF[dr_Classification].isna()  # Unclassified rows in DRDF

    # Step 4: Classification logic
    # True Positive (TP): Both CRDF and DRDF detected (1)
    CRDF.loc[mask_CRDF_is_1 & mask_DRDF_is_1 & mask_TC_Classification_isnan & mask_Det_Classification_isnan, tc_Classification] = 'TP'
    DRDF.loc[mask_CRDF_is_1 & mask_DRDF_is_1 & mask_TC_Classification_isnan & mask_Det_Classification_isnan, dr_Classification] = 'TP'

    # False Negative (FN): CRDF detected (1), DRDF not detected (0) or missing (Nan)
    CRDF.loc[mask_CRDF_is_1 & (mask_DRDF_is_0 | mask_DRDF_is_Nan) & mask_TC_Classification_isnan & mask_Det_Classification_isnan, tc_Classification] = 'FN'

    # False Positive (FP): CRDF not detected (0), DRDF detected (1)
    DRDF.loc[mask_CRDF_is_0 & mask_DRDF_is_1 & mask_TC_Classification_isnan & mask_Det_Classification_isnan, dr_Classification] = 'FP'

    # True Negative (TN): Both CRDF and DRDF not detected (0)
    CRDF.loc[mask_CRDF_is_0 & (mask_DRDF_is_0 | mask_DRDF_is_Nan) & mask_TC_Classification_isnan & mask_Det_Classification_isnan, tc_Classification] = 'TN'
    DRDF.loc[mask_CRDF_is_0 & mask_DRDF_is_0 & mask_TC_Classification_isnan & mask_Det_Classification_isnan, dr_Classification] = 'TN'

    # Exclusion: Missing CRDF data (NaN), DRDF detected (1) or not Detected (0) or missing (Nan)
    #CRDF.loc[mask_CRDF_is_Nan & (mask_DRDF_is_1 | mask_DRDF_is_0 | mask_DRDF_is_Nan) & mask_TC_Classification_isnan, tc_Classification] = "Missing controlled release rate: Data Excluded"

    # Step 5: Combine CRDF and DRDF into a single DataFrame
    ClassifiedDF = pd.concat([CRDF, DRDF], axis=1, join='inner')

    # Step 6: Assign flags to timesteps in CRDF and DRDF with threshold = 0 when the corresponding timesteps in DRDF and CRDF are flagged FP and FN
    CRDF.loc[mask_CRDF_is_0 & (DRDF[dr_Classification] == "FP") & mask_TC_Classification_isnan, tc_Classification] = "Below threshold to be classified an FN"
    DRDF.loc[mask_DRDF_is_0 & (CRDF[tc_Classification] == "FN") & mask_Det_Classification_isnan, dr_Classification] = "Below threshold to be classified an FP"

    # Step 7: Create a unified 'Classification' column
    ClassifiedDF['Classification'] = ClassifiedDF[tc_Classification].combine_first(ClassifiedDF[dr_Classification])

    # Step 8: Remove the original classification columns as they are now merged
    ClassifiedDF.drop([tc_Classification, dr_Classification], axis=1, inplace=True)

    # Step 9: Return the final merged and classified DataFrame
    return ClassifiedDF, CRDF, DRDF


def merge_lists(*lists):
    """
    Merges multiple lists and returns the elements as a single list without braces.
    Parameters:
        - *lists: Variable number of lists to be merged.
    Returns:
        - list: A list containing all elements from the input lists as strings.
    """
    # Merge the lists
    merged_list = [str(item) for lst in lists if lst is not None for item in lst]

    return merged_list

def timestep_correction(*listofcols, df, eventID):
    """
    Zeroes out the emission rate values at the beginning of each new event group.

    Parameters:
        - df: pandas DataFrame containing the data
        - eventID: name of the column representing event identifiers
        - *listofcols: lists of columns of interest

    Returns:
        - df: the modified DataFrame with cols in listofcols set to 0 at the start of each event
    """

    print(df.columns)

    if eventID is None or eventID not in df.columns:
        raise ValueError(f"`eventID` is None or not found in DataFrame columns: {df.columns}")

    # Identify rows where the event ID changes (i.e., start of a new group)
    change_points = df[eventID] != df[eventID].shift()

    # Create a 'Flag' column:
    # Set 0 where the event ID changes (first row of new group)
    # Set 1 elsewhere (continuation of the same event)
    df['Flag'] = 0
    df.loc[change_points, 'Flag'] = 0
    df.loc[~change_points, 'Flag'] = 1

    # prepare lists
    valid_lists = [lst for lst in listofcols if lst is not None]
    # Merge all valid lists
    merged = []
    for lst in valid_lists:
        merged.extend(lst)

    # Zero out the first record of each new event group
    for col in merged:
        if col is not None:
            df[col] = df.apply(lambda row: row[col] * row['Flag'], axis=1)

    # Drop "Flag" from list of columns
    df.drop(columns=['Flag'], inplace=True)

    # Return the updated DataFrame
    return df

def process_lists(DF, *lists):
    """
    Processes multiple lists of columns in a DataFrame by converting values to floats.
    Parameter:
        - DF: The DataFrame containing the data.
        - *lists : One or more lists of column names to process.

    Returns:
        - Dataframe: Modified DataFrame with specified columns converted to float.
    """
    # Merge all provided lists into one flat list
    all_columns = [col for lst in lists if lst is not None for col in lst]

    if len(all_columns) > 0:
        for col in all_columns:
            if col not in DF.columns:
                print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
                continue
            # Convert values to float, keeping nulls as np.nan
            DF[col] = DF[col].apply(lambda x: float(x) if pd.notnull(x) else np.nan)

    return DF

def drdf_constant_runs(drdf, dr_emissionrate, dr_emissionrateBounds, dr_Classification, timefactor, freq):
    """
        Extracts runs of constant emission rate and flag values, and summarizes them.

        Parameter:
            - drdf : Input DataFrame with datetime index and columns for emission rate and flag.
            - dr_emissionrate : The reported emission rate column.
            - dr_emissionrateBounds: The lower and upper bound on reported emission rate.
            - dr_Classification : The detection classification column.
            - timefactor: A factor to convert the time basis of emission rate to seconds.
            - freq: frequency
        returns:
            - Dataframes: agg_drdf, drdf
    """

    # Make a copy of dataframe and ensure index is datetime
    DF = drdf.copy()
    DF.index = pd.to_datetime(DF.index)
    # Reset the index to access it as a column for aggregation
    DF = DF.reset_index().rename(columns={"index": "Time"})
    # Convert frequency to seconds
    freq_duration = pd.Timedelta(freq).total_seconds()

    # Identify change points when either emission or flag changes
    DF['shift'] = (DF[dr_emissionrate].ne(DF[dr_emissionrate].shift()) | DF[dr_Classification].ne(DF[dr_Classification].shift())).cumsum()

    # Group by consecutive runs
    if (dr_emissionrateBounds[0] in DF.columns) and (dr_emissionrateBounds[1] in DF.columns):
        agg_drdf = DF.groupby('shift').agg(dr_EmissionRate=(dr_emissionrate, "first"),
                                           dr_EmissionRateLower=(dr_emissionrateBounds[0], "first"),
                                           dr_EmissionRateUpper=(dr_emissionrateBounds[1], "first"),
                                           Classification=(dr_Classification, "first"),
                                           dr_StartTime=('Time', 'first'),
                                           dr_EndTime=('Time', 'last')).reset_index(drop=True)
    else:
        agg_drdf = DF.groupby('shift').agg(dr_EmissionRate=(dr_emissionrate, "first"),
                                           Classification=(dr_Classification, "first"),
                                           dr_StartTime=('Time', 'first'),
                                           dr_EndTime=('Time', 'last')).reset_index(drop=True)

    # Adjust the start datetime
    agg_drdf["dr_StartTime"] = agg_drdf["dr_StartTime"].apply(lambda dt: datetime.utcfromtimestamp(dt.timestamp() - freq_duration))

    # Compute duration in seconds
    agg_drdf["dr_Duration"] = (agg_drdf["dr_EndTime"] - agg_drdf["dr_StartTime"]).dt.total_seconds()

    # Compute total emissions
    agg_drdf["dr_total_emission"] = agg_drdf["dr_EmissionRate"] * agg_drdf["dr_Duration"] * timefactor

    # Calculate total emission for each time step
    drdf["dr_total_emission"] = drdf[dr_emissionrate] * freq_duration * timefactor

    # Add marker for only when detection was reported
    reported = (agg_drdf["dr_EmissionRate"] > 0) & (agg_drdf["Classification"].isna())
    agg_drdf.loc[reported, "Classification"] = "1"
    agg_drdf.loc[agg_drdf["Classification"].isna(), "Classification"] = "0"

    # Rename dataframe columns
    agg_drdf.rename(columns={"dr_EmissionRateLower": f"{dr_emissionrateBounds[0]}"}, inplace=True)
    agg_drdf.rename(columns={"dr_EmissionRateUpper": f"{dr_emissionrateBounds[1]}"}, inplace=True)

    return agg_drdf, drdf

def crdf_constant_runs(drdf, crdf, timefactor, cr_emissionrate, cr_emissionrateBounds, freq):
    """
        Compute total emissions and average emission rate per time window.

        Parameters
            - drdf : Detection report dataframe.
            - crdf : Controlled release dataframe with datetime index.
            - timefactor : Factor to convert emission rate time basis to seconds.
            - emissionrate : Column header for controlled release emission rate.
            - cr_emissionrateBounds: Lower and upper bound on controlled release rate
            - freq: frequency

        return:
            - Dataframes: agg_crdf, crdf
    """

    results = []
    # Ensure datetime format
    drdf["dr_StartTime"] = pd.to_datetime(drdf["dr_StartTime"])
    drdf["dr_EndTime"] = pd.to_datetime(drdf["dr_EndTime"])
    crdf.index = pd.to_datetime(crdf.index)
    # Convert frequency to seconds
    freq_duration = pd.Timedelta(freq).total_seconds()

    for _, row in drdf.iterrows():
        start, end = row["dr_StartTime"], row["dr_EndTime"]
        # Correct the detection report start time due to previous operations
        starttime = datetime.utcfromtimestamp(start.timestamp() + freq_duration)

        # Slice crdf between window
        this_crdf = crdf.loc[starttime:end]
        if this_crdf.empty and (cr_emissionrateBounds[0] in crdf.columns) and (cr_emissionrateBounds[1] in crdf.columns):
            results.append({"cr_StartTime": start, "cr_EndTime": end, "tc_total_emission": 0, f"{cr_emissionrate}": 0,
                            f"{cr_emissionrateBounds[0]}": 0, f"{cr_emissionrateBounds[1]}": 0})
            continue
        elif this_crdf.empty and (cr_emissionrateBounds[0] not in crdf.columns) and (cr_emissionrateBounds[1] not in crdf.columns):
            results.append({"cr_StartTime": start, "cr_EndTime": end, "tc_total_emission": 0, f"{cr_emissionrate}": 0})
            continue
        # Identify constant runs of emission rate
        this_crdf = (this_crdf.assign(shift=this_crdf[cr_emissionrate].ne(this_crdf[cr_emissionrate].shift()).cumsum()))
        this_crdf = this_crdf.reset_index().rename(columns={"index": "Time"})
        if (cr_emissionrateBounds[0] in crdf.columns) and (cr_emissionrateBounds[1] in crdf.columns):
            runs = this_crdf.groupby("shift").agg(EmissionRate=(cr_emissionrate, "first"),
                                                  EmissionRateLower=(cr_emissionrateBounds[0], "first"),
                                                  EmissionRateUpper=(cr_emissionrateBounds[1], "first"),
                                                  StartTime=("Time", "first"),
                                                  EndTime=("Time", "last")).reset_index(drop=True)
        else:
            runs = this_crdf.groupby("shift").agg(EmissionRate=(cr_emissionrate, "first"),
                                                  StartTime=("Time", "first"),
                                                  EndTime=("Time", "last")).reset_index(drop=True)
        # Adjust the start datetime
        runs["StartTime"] = runs["StartTime"].apply(lambda dt: datetime.utcfromtimestamp(dt.timestamp() - freq_duration))

        # Duration in seconds
        runs["Duration"] = (runs["EndTime"] - runs["StartTime"]).dt.total_seconds()

        # Vectorized computation of emissions
        if (cr_emissionrateBounds[0] in crdf.columns) and (cr_emissionrateBounds[1] in crdf.columns):
            runs["Total"] = runs["EmissionRate"] * runs["Duration"] * timefactor
            runs["TotalLower"] = runs["EmissionRateLower"] * runs["Duration"] * timefactor
            runs["TotalUpper"] = runs["EmissionRateUpper"] * runs["Duration"] * timefactor
            # Total emissions
            total = runs["Total"].sum()
            totalLower = runs["TotalLower"].sum()
            totalUpper = runs["TotalUpper"].sum()
            # Average emission rate across the window
            duration = (end - start).total_seconds()
            avg_rate = (total / (timefactor * duration)) if duration > 0 else 0
            avg_ratelower = (totalLower / (timefactor * duration)) if duration > 0 else 0
            avg_rateupper = (totalUpper / (timefactor * duration)) if duration > 0 else 0
            results.append({"cr_StartTime": start, "cr_EndTime": end, "tc_total_emission": total,
                            f"{cr_emissionrate}": avg_rate, f"{cr_emissionrateBounds[0]}": avg_ratelower,
                            f"{cr_emissionrateBounds[1]}": avg_rateupper})
        else:
            runs["Total"] = runs["EmissionRate"] * runs["Duration"] * timefactor
            # Total emissions
            total = runs["Total"].sum()
            # Average emission rate across the window
            duration = (end - start).total_seconds()
            avg_rate = (total / (timefactor * duration)) if duration > 0 else 0
            results.append({"cr_StartTime": start, "cr_EndTime": end, "tc_total_emission": total, f"{cr_emissionrate}": avg_rate})

    # Calculate total emission for each time step
    crdf["tc_total_emission"] = crdf[cr_emissionrate] * freq_duration * timefactor

    # Rename dataframe columns
    agg_crdf = pd.DataFrame(results)
    if (cr_emissionrateBounds[0] in crdf.columns) and (cr_emissionrateBounds[1] in crdf.columns):
        agg_crdf.rename(columns={"EmissionRateLower": f"{cr_emissionrateBounds[0]}"}, inplace=True)
        agg_crdf.rename(columns={"EmissionRateUpper": f"{cr_emissionrateBounds[1]}"}, inplace=True)

    return agg_crdf, crdf


def QuantAnalysis(drdf, crdf, timefactor, cr_emissionrate, dr_emissionrate, cr_emissionrateBounds, dr_emissionrateBounds,
                  dr_Classification, freq):
    """
    Parameters:
        - drdf: Detection report dataframe.
        - crdf: Controlled release dataframe.
        - timefactor: Factor to convert emission rate time basis to seconds.
        - cr_emissionrate: Column header for controlled release emission rate.
        - dr_emissionrate: Column header for detection report emission rate.
        - cr_emissionrateBounds: Lower and upper bound on controlled release rate
        - dr_emissionrateBounds: Lower and upper bound on detction report rate
        - dr_Classification: The classification column name
        - freq: frequency
    return:
         - Dataframes: pairedDF, DRdf, crdf
    """


    # Compute the aggregate reported emission rate for reported duration for the given ASE
    agg_drdf, DRdf = drdf_constant_runs(drdf=drdf,
                                        dr_emissionrate=dr_emissionrate,
                                        dr_emissionrateBounds=dr_emissionrateBounds,
                                        dr_Classification=dr_Classification,
                                        timefactor=timefactor,
                                        freq=freq)
    # Compute the corresponding aggregate controlled release rate for reported duration for the given ASE
    agg_crdf, crdf = crdf_constant_runs(drdf=agg_drdf,
                                        crdf=crdf,
                                        timefactor=timefactor,
                                        cr_emissionrate=cr_emissionrate,
                                        cr_emissionrateBounds=cr_emissionrateBounds,
                                        freq=freq)
    # Concatenate two dataframes along the row.
    pairedDF = pd.concat([agg_drdf, agg_crdf], axis=1)

    pairedDF.rename(columns={"dr_EmissionRate": f"{dr_emissionrate}"}, inplace=True)

    return pairedDF, DRdf, crdf

def filterData(df, start, end, threshold):
    """
        Filters a DataFrame to keep only rows where the duration between two
        datetime columns exceeds a given threshold (in seconds).

    Parameters
        - df : Input dataframe.
        - start : Column name containing start times.
        - end : Column name containing end times.
        - threshold : Minimum duration (in seconds) required to keep a row.

    Returns
        - Dataframe: Filtered dataframe.
    """

    # Convert start and end columns to datetime
    df[start] = pd.to_datetime(df[start], errors="coerce")
    df[end] = pd.to_datetime(df[end], errors="coerce")

    # Compute duration in seconds between start and end times
    df["Duration"] = (df[end] - df[start]).dt.total_seconds()

    initial_count = len(df)
    # Filter rows where duration exceeds the threshold
    df = df.loc[df["Duration"] > threshold]

    # Drop Duration column
    df.drop("Duration", axis=1)

    print(f"The number of rows excluded because the difference between start date time and end date time is less or equal to {threshold} is {initial_count - len(df)}")
    return df

def mergeRuns(classified, cr_emissionrate, factor, freq):
    """
    The program aggregates sequence of rows with same detection classification as unique detection events.

    Parameters:
        - classified: A dataframe of paired controlled release and detection data
        - cr_emissionrate: The column header for controlled release rate
        - factor: float to convert from seconds to the time basis of the emission rate data
        - freq: frequency of analysis.
    return:
        - Dataframe:
    """
    # Convert frequency to seconds
    freq_duration = pd.Timedelta(freq).total_seconds()

    # Create a new column "shift" identifying when values of the "Classification" column changes
    crdf = (classified.assign(shift=classified['Classification'].ne(classified['Classification'].shift()).cumsum()))

    # Calculate total emission for each time step
    crdf["Total"] = crdf[cr_emissionrate] * freq_duration * factor

    # Aggregating common runs
    runs = (crdf.groupby("shift").agg(Total=("Total", "sum"),  # sum of Total
                                      Count=("Total", "size"),  # number of rows in run
                                      Classification=("Classification", "first"),  # first classification
                                      ).reset_index(drop=True))

    # Compute EmissionRate
    runs["EmissionRate"] = runs["Total"] / (runs["Count"] * freq_duration * factor)

    # Rename dataframe columns
    runs.rename(columns={"EmissionRate": f"{cr_emissionrate}"}, inplace=True)

    # Drop columns
    runs.drop(columns=['Count', 'Total'], inplace=True)

    return runs

def DataPairing(controlledReleaseDF, detectionReportDF, tc_baselineEmissionRate, dr_baselineEmissionRate,
                cr_releaseRate, dr_releaseRate, cr_emissionrateBounds, dr_emissionrateBounds, MaintenanceRecordDF,
                mr_DateTime, cr_DateTime, dr_DateTime, OfflineRecordDF, OfflineDateTime, OnlineDateTime,
                dr_ID_column=None, tc_ID_column=None, freq="T", bufferTime=(0, 0), overLappingFrac=0, ThresholdDuration=0,
                ThresholdRateAboveBaseline=0, detection=None, nSampling=500, Method="Rolling Average",
                listOfVariables2Sum_CR=None, listOfVariables2Average_CR=None, listOfVariables2ListUnique_CR=None,
                listOfVariables2Max_CR=None, listOfVariables2Min_CR=None, listOfVariables2Sum_DR=None,
                listOfVariables2Average_DR=None, listOfVariables2ListUnique_DR=None, listOfVariables2Max_DR=None,
                listOfVariables2Min_DR=None, listOfVariables2ObtainFirst_CR=None, listOfVariables2ObtainLast_CR=None,
                factor=1/3600, listOfVariables2ObtainFirst_DR=None, listOfVariables2ObtainLast_DR=None, tc_PolygonCoord=None,
                tc_BoundingZ=None, tc_localization=None, dr_PolygonCoord=None, dr_BoundingZ=None, dr_localization=None,
                ASE_PolygonCoord_Dict=None, ASE_textual_mapper=None, tc_bufferDistance=(0,0), dr_bufferDistance=(0, 0)):
    """
    DataPairing integrates and processes time-series data from two sources (controlled release and detection report).
    It performs spatial filtering, temporal alignment, exclusion of invalid periods, aggregation, thresholding,
    and classification of paired data.

    Parameters:
    - controlledReleaseDF: Controlled release data (test center).
    - detectionReportDF: Detection report data.
    - tc_baselineEmissionRate: Baseline emission rate as determined by the test center.
    - dr_baselineEmissionRate: Baseline emission rate as determined by the solution/technology.
    - cr_releaseRate: Release rate column in controlled release data.
    - dr_releaseRate: Release rate column in detection report data.
    - MaintenanceRecordDF: DataFrame of maintenance periods to exclude.
    - mr_DateTime: Column names for maintenance period start and end times.
    - cr_DateTime: Column names for controlled release start and end times.
    - dr_DateTime: Column names for detection report start and end times.
    - OfflineRecordDF: DataFrame of offline periods to exclude.
    - OfflineDateTime: Column name for offline period start time.
    - OnlineDateTime: Column name for offline period end time.
    - freq: Frequency for time-series aggregation (e.g., "T" for minutes).
    - TimeUnit: Unit for rounding timestamps (e.g., "minute").
    - bufferTime: Buffer time around flagged periods.
    - RollingAverageDuration, RollingAverageRate: Parameters for rolling average thresholding.
    - overLappingFrac:
    - DetectionRateThresholdAboveBaseline: Detection rate threshold above baseline.
    - DetectionDurationThreshold: Minimum detection duration for thresholding.
    - detection:
    - listOfVariables2Sum_CR:
    - listOfVariables2Average_CR:
    - listOfVariables2ListUnique_CR:
    - listOfVariables2Max_CR:
    - listOfVariables2Min_CR:
    - listOfVariables2ObtainFirst_CR:
    - listOfVariables2ObtainLast_CR:
    - listOfVariables2Sum_DR:
    - listOfVariables2Average_DR:
    - listOfVariables2ListUnique_DR:
    - listOfVariables2Max_DR:
    - listOfVariables2Min_DR:
    - listOfVariables2ObtainFirst_DR:
    - listOfVariables2ObtainLast_DR:
    - tc_PolygonCoord:
    - tc_BoundingZ:
    - tc_localization:
    - dr_PolygonCoord:
    - dr_BoundingZ:
    - dr_localization:
    - ASE_PolygonCoord_Dict:
    - ASE_textual_mapper:
    - tc_bufferDistance: A tuple of the distance in meters applied as buffer to the test center ASE polygon (i.e., element can be positive or negative).
                         The first and second elements of the tuple are the distance for the test center ASE and the subASE, respecively.
    - dr_bufferDistance: A tuple of the distance in meters applied as buffer to the detection report ASE polygon (i.e., element can be positive or negative).
                         The first and second elements of the tuple are the distance for the detection report ASE and the subASE, respecively.

    Returns:
    - pd.DataFrame: Final classified and merged DataFrame with results.

    ***STEPS***:
    1. Spatial Filtering: to filter out controlled release and detection report data within the ASE defined for the analysis
        *Assumptions*
        a. The test center specified an ASE for the analysis (format: WKT Polygon)
        b. Each controlled release data has localization in the form of a WKT Polygon or an asset for mapping.
        c. Each detection report data has localization in the form of a WKT Polygon or an asset for mapping.
        *Code Structure*
        a. Calculates the intersection fraction of controlled release data/detection reports with ASE.
        b. Extracts controlled releases and detection reports with intersection fractions greater than a threshold (default = 0, can be changed since it is a variable in the code).

    2. Temporal Resolution: To break down reported and controlled release data into timesteps specified by the freq variable.
        *Assumptions*
        a. The start and end date times of controlled releases and detection reports are in UTC Datetime formats.
        *Code Structure*
        a. For each controlled release and detection report data, round up start and end times to the nearest time unit (default = minute, can be changed since it is a variable in the code).
        b. For each controlled release and detection report data, subdivide into times steps specified by input datetime frequency (default = 1 minute, can be changed since it is a variable in the code).

    3. Maintenance and Offline Exclusion: To exclude controlled release and detection report during maintenance and offline records.
        *Assumptions*
        a. The maintenance and offline record data frame exists. Empty data frames are allowed.
        b. The maintenance and offline record start and end date-times are in UTC Datetime formats.
        *Code Structure*
        a. If the maintenance and offline record data frames are non-empty, apply buffer times to their respective start and end times (default buffer times = 0 minutes, can be changed since it is a variable in the code).
        b. Merge offline records or maintenance activities with overlapping durations such that each offline record or maintenance activity covers a distinct period/interval.
        c. Flag every controlled release and detection report time step within any maintenance record interval/duration.
        d. Flag every controlled release time step within any offline record interval/duration.
        e. Flag the time steps of every detection report that whole or partially overlaps with any offline record interval.

    4. Data random sampling: This generates random samples of controlled release and detection data to be used if the thresholding method selected is: Random Sample Average
        *Assumptions*
        a. The controlled release and detection report data has start and end datetimes in UTC.
        b. The above dataframes must have controlled release and emission rate data.
        c. Only controlled release and detection report data not excluded are considered
        *Code Structure*
        a. Detection reports data are randomly sampled until the reported accumulated detection duration is > ThresholdDuration
        b. The sampled controlled release data overlap with the selected detection reports.
        c. The means of the sampled controlled release data and detection report data are evaluated respectively.
        d. sampling is done nSampling times to obtain nSampling mean values

    5. Aggregation by Frequency: This aggregates all the multiple data with a common timestep. The aggregation process includes summation, mean, etc.. depending on the data columns populated in the lists.
       The code accepts lists of data frame column headers to produce a (1) sum, (2) mean, (3) unique list, (4) maximum, (5) minimum, (6) first value based on ID, and (7) last value based on ID, for every unique timestep of a data frame. By default, these lists can be empty.
        *Assumptions*
        a. The index of the time-series controlled release and detection report data is set to the resultant datetimes from the temporal resolution performed earlier.
        *Code Structure*
        a. For either controlled release or detection report time-series data frame, aggregate all data within each unique time step by:
              i.  Summing the data associated with each column header in the list of columns to be summed.
             ii.  Evaluating the mean of the data associated with each column header in the list of columns to be averaged.
             iii. Extracting unique elements of the data associated with each column header in the list of columns to be listed.
             iv.  Evaluating the minimum and maximum of the data associated with column header in the list of columns of which the minimum and maximum values are of interest, respectively.
             v.   Extracting values associated with the first and last detection report ID for each column in the list of columns of which the first and last value depending on detection report ID are of interest, respectively.

    6. Align time indexes: Ensures that both time-resolved controlled release and detection report data have the same stat, end, and length of datetimes as their indexes.
       *Assumptions*
       a. The range of datetimes of the time-series controlled release and detection report data are respectively within the period of interest for analysis.
       *Code Structure*
       a. Use the range of datetime of the controlled releases to filter out or extend the datetime steps of the relevant detection reports

    7 Processing data for quantification analysis: This ensures that the controlled release data and detection report data are processed for sample metrics i.e., one to one pairing.

    8. Apply Thresholding: Applying the detection criteria provided equally to controlled release and detection report data.
        *Assumptions*
        a. The detection critera is captured in the 3 thresholding methods defined otherwise a new method will have to be defined and implemented
        *Code Structure*
        a. A value of 1 indicates data passed detection criteria while 0 means failed. We call this process thresholding.
        b. Thresholding is conducted depending on selected method. They include:
            * Above Threshold Across Duration: Produces a thresholding value of 1 when the cummulative emissions duration at
              a timestep/instance > specified thresholding duration (default = 0), and controlled release/estimated emission
              rate > baseline estimate + detection threshold above baseline. If condition is not met, thresholding value is 0.
            * Rolling Average: Produces a thresholding value of 1 when controlled release/estimated emission rate at a given
              timestep is > the rolling average rate/estimate evaluated over a specified thresholding duration. If condition
              is not met, thresholding value is 0.
            * Random Sample Average: Produces a thresholding value of 1 when the mean of random samples of controlled
              release/estimated emission rate with total duration <= specified thresholding duration, is > than baseline
              estimate + detection threshold above baseline. If the condition is not met, then thresholding value is 0.

    9. Compare and Classify: Compares the thresholding values of controlled releases and detection reports to produce TP, FN, FP, and TN classification
        *Assumptions*
        a. Time resolved controlled release detection report data must have been thresholded (i.e, with values of 1s and 0s)
        *Code Structure*
        a. Thresholding value is 1 and 1 for controlled release and detection reports, respectively ==> TP
        b. Thresholding value is 1 and 0 for controlled release and detection reports, respectively ==> FN
        a. Thresholding value is 0 and 1 for controlled release and detection reports, respectively ==> FP
        a. Thresholding value is 0 and 0 for controlled release and detection reports, respectively ==> TN
        a. Thresholding value is Nan for controlled release ==> flagged and excluded

    10. Final Output Preparation
    """

    ### Step 1: Spatial Filtering ###
    try:
        start_time = time.time()
        # Subset controlled release data (test center) based on spatial extent.
        Data4thisASE_tc = extractSpatialSubsetData(DataDF=controlledReleaseDF,
                                                   DataDF_PolygonCoord=tc_PolygonCoord,
                                                   DataDF_BoundingZMin=tc_BoundingZ[0],
                                                   DataDF_BoundingZMax=tc_BoundingZ[1],
                                                   DataDF_localization=tc_localization,
                                                   ASE_PolygonCoord_Dict=ASE_PolygonCoord_Dict,
                                                   ASE_textual_mapper=ASE_textual_mapper,
                                                   overLappingFrac=overLappingFrac,
                                                   ASE_bufferDistance=tc_bufferDistance[0],
                                                   subASE_bufferDistance=tc_bufferDistance[1])
        # Ensure relevant columns are in float format for subsequent Analysis
        Data4thisASE_tc = process_lists(Data4thisASE_tc, listOfVariables2Sum_CR, listOfVariables2Average_CR)
        # Remove controlled release data with reported duration less or equal to 0 seconds
        Data4thisASE_tc = filterData(df=Data4thisASE_tc, start=cr_DateTime[0], end=cr_DateTime[1], threshold=0)

        # Subset detection report data based on spatial extent.
        Data4thisASE_dr = extractSpatialSubsetData(DataDF=detectionReportDF,
                                                   DataDF_PolygonCoord=dr_PolygonCoord,
                                                   DataDF_BoundingZMin=dr_BoundingZ[0],
                                                   DataDF_BoundingZMax=dr_BoundingZ[1],
                                                   DataDF_localization=dr_localization,
                                                   ASE_PolygonCoord_Dict=ASE_PolygonCoord_Dict,
                                                   ASE_textual_mapper=ASE_textual_mapper,
                                                   overLappingFrac=overLappingFrac,
                                                   ASE_bufferDistance=dr_bufferDistance[0],
                                                   subASE_bufferDistance=dr_bufferDistance[1])
        # Ensure relevant columns are in float format for subsequent Analysis
        Data4thisASE_dr = process_lists(Data4thisASE_dr, listOfVariables2Sum_DR, listOfVariables2Average_DR)

        # Remove detection report data with reported duration less or equal to 0 seconds
        Data4thisASE_dr = filterData(df=Data4thisASE_dr,
                                     start=dr_DateTime[0],
                                     end=dr_DateTime[1],
                                     threshold=0)
        # End the timer
        end_time = time.time()
        print(f"Spatial filtering code executed in {end_time - start_time:.6f} seconds")
    except Exception as e:
        print(f"Could not conduct spatial filtering due to: {e}")
        raise

    ### Step 2: Temporal Resolution ###
    try:
        start_time = time.time()
        # Merge a relevant controlled release column header lists
        tc_variables = merge_lists(listOfVariables2Sum_CR, listOfVariables2Average_CR, listOfVariables2ListUnique_CR,
                                   listOfVariables2Max_CR, listOfVariables2Min_CR, listOfVariables2ObtainFirst_CR,
                                   listOfVariables2ObtainLast_CR)

        # Merge a relevant detection report column header lists
        dr_variables = merge_lists(listOfVariables2Sum_DR, listOfVariables2Average_DR, listOfVariables2ListUnique_DR,
                                   listOfVariables2Max_DR, listOfVariables2Min_DR, listOfVariables2ObtainFirst_DR,
                                   listOfVariables2ObtainLast_DR)

        # Round controlled release data timestamps to the nearest frequency provided.
        Data4thisASE_tc[cr_DateTime[0]] = Data4thisASE_tc[cr_DateTime[0]].apply(lambda dt: roundDatetime(dt, freq=freq))
        Data4thisASE_tc[cr_DateTime[1]] = Data4thisASE_tc[cr_DateTime[1]].apply(lambda dt: roundDatetime(dt, freq=freq))
        # Round detection report data timestamps to the nearest frequency provided.
        Data4thisASE_dr[dr_DateTime[0]] = Data4thisASE_dr[dr_DateTime[0]].apply(lambda dt: roundDatetime(dt, freq=freq))
        Data4thisASE_dr[dr_DateTime[1]] = Data4thisASE_dr[dr_DateTime[1]].apply(lambda dt: roundDatetime(dt, freq=freq))
        # Expand data into time steps specified by frequency.
        tc_TimeSeriesDF = expand_Date_intervals(variables=tc_variables,
                                                df=Data4thisASE_tc,
                                                startTime=cr_DateTime[0],
                                                endTime=cr_DateTime[1],
                                                freq=freq,
                                                inclusive="right")
        dr_TimeSeriesDF = expand_Date_intervals(variables=dr_variables,
                                                df=Data4thisASE_dr,
                                                startTime=dr_DateTime[0],
                                                endTime=dr_DateTime[1],
                                                freq=freq,
                                                inclusive="right")
        end_time = time.time()
        print(f"Temporal resolution code executed in {end_time - start_time:.6f} seconds")
    except Exception as e:
        print(f"Could not conduct temporal resolution due to: {e}")
        raise

    ### Step 3: Maintenance and Offline Exclusion ###
    try:
        start_time = time.time()
        # Exclude test center and detection report data during maintenance periods - data based on specified frequency.
        listOfVariables2ListUnique_CR.append("tc_Classification")
        listOfVariables2ListUnique_DR.append("dr_Classification")
        tc_TimeSeriesDF["tc_Classification"] = None
        dr_TimeSeriesDF["dr_Classification"] = None
        tc_TimeSeriesDF = flagData(dataDF=tc_TimeSeriesDF,
                                   exclusionDF=MaintenanceRecordDF,
                                   freq=freq,
                                   flag="Maintenance Record: Data Excluded",
                                   classification="tc_Classification",
                                   start_exc=mr_DateTime[0],
                                   end_exc=mr_DateTime[1],
                                   bufferTime=bufferTime)

        dr_TimeSeriesDF = flagData(dataDF=dr_TimeSeriesDF,
                                   exclusionDF=MaintenanceRecordDF,
                                   freq=freq,
                                   flag="Maintenance Record: Data Excluded",
                                   classification="dr_Classification",
                                   start_exc=mr_DateTime[0],
                                   end_exc=mr_DateTime[1],
                                   bufferTime=bufferTime)

        # Exclude data during offline periods.
        tc_TimeSeriesDF = flagData(dataDF=tc_TimeSeriesDF,
                                   exclusionDF=OfflineRecordDF,
                                   freq=freq,
                                   flag="Offline Record: Data Excluded",
                                   classification="tc_Classification",
                                   start_exc=OfflineDateTime,
                                   end_exc=OnlineDateTime,
                                   bufferTime=bufferTime)

        dr_TimeSeriesDF = flagData(dataDF=dr_TimeSeriesDF,
                                   exclusionDF=OfflineRecordDF,
                                   freq=freq,
                                   flag="Offline Record: Data Excluded",
                                   classification="dr_Classification",
                                   start_exc=OfflineDateTime,
                                   end_exc=OnlineDateTime,
                                   ID=dr_ID_column,
                                   bufferTime=bufferTime)

        end_time = time.time()
        print(f"Maintenance and offline exclusion code executed in {end_time - start_time:.6f} seconds")
    except Exception as e:
        print(f"Could not conduct data exclusion due to: {e}")
        raise

    ### Step 4: Data random sampling ###
    try:
        start_time = time.time()
        tcDF, drDF, rs_CRDF, rs_DRDF = samplefromdata(CRDF=tc_TimeSeriesDF,
                                                      DRDF=dr_TimeSeriesDF,
                                                      cr_DateTime=cr_DateTime,
                                                      dr_DateTime=dr_DateTime,
                                                      nSampling=nSampling,
                                                      cr_releaseRate=cr_releaseRate,
                                                      dr_releaseRate=dr_releaseRate,
                                                      ThresholdDuration=ThresholdDuration)
        end_time = time.time()
        print(f"Data random sampling code executed in {end_time - start_time:.6f} seconds")
    except Exception as e:
        print(f"Could not random sampling due to: {e}")
        raise

    ### Step 5: Aggregation by Frequency ###
    try:
        start_time = time.time()
        # Aggregate controlled release data.
        print(f"Starting controlled release aggregation")
        CRDF = aggregateTimeSeriesDataByFreq(DataDF=tcDF,
                                             freq=freq,
                                             ID_column=tc_ID_column,
                                             min_count=0,
                                             listOfVariables2Sum=listOfVariables2Sum_CR,
                                             listOfVariables2Average=listOfVariables2Average_CR,
                                             listOfVariables2ListUnique=listOfVariables2ListUnique_CR,
                                             listOfVariables2Max=listOfVariables2Max_CR,
                                             listOfVariables2Min=listOfVariables2Min_CR,
                                             listOfVariables2ObtainFirst=listOfVariables2ObtainFirst_CR,
                                             listOfVariables2ObtainLast=listOfVariables2ObtainLast_CR)

        cr_time = time.time()
        print(f"Controlled release aggregation took {cr_time - start_time:.2f} seconds")
        # Aggregate detection report data.
        print(f"Starting detection data aggregation")
        DRDF = aggregateTimeSeriesDataByFreq(DataDF=drDF,
                                             freq=freq,
                                             ID_column=dr_ID_column,
                                             min_count=0,
                                             listOfVariables2Sum=listOfVariables2Sum_DR,
                                             listOfVariables2Average=listOfVariables2Average_DR,
                                             listOfVariables2ListUnique=listOfVariables2ListUnique_DR,
                                             listOfVariables2Max=listOfVariables2Max_DR,
                                             listOfVariables2Min=listOfVariables2Min_DR,
                                             listOfVariables2ObtainFirst=listOfVariables2ObtainFirst_DR,
                                             listOfVariables2ObtainLast=listOfVariables2ObtainLast_DR)
        end_time = time.time()
        print(f"Detection report aggregation executed in {end_time - start_time:.6f} seconds")
    except Exception as e:
        print(f"Could not aggregate data due to the exception: {e}")
        raise

    ### Step 6: Align the time indexes of the two dataframes for comparison - make them equal in length.
    try:
        start_time = time.time()
        CRDF, DRDF = align_time_indexes(crdf=CRDF,
                                        drdf=DRDF,
                                        dr_columns=[dr_releaseRate, dr_emissionrateBounds[0], dr_emissionrateBounds[1]],
                                        detection=detection)
        end_time = time.time()
        print(f"Aligning time index code executed in {end_time - start_time:.6f} seconds")
    except Exception as e:
        print(f"Could not align time indexes due to the exception: {e}")
        raise

    ### Step 7: Processing data for quantification analysis
    try:
        start_time = time.time()
        pairedDF, DRDF, CRDF = QuantAnalysis(drdf=DRDF,
                                             crdf=CRDF,
                                             timefactor=factor,
                                             cr_emissionrate=cr_releaseRate,
                                             dr_emissionrate=dr_releaseRate,
                                             cr_emissionrateBounds=cr_emissionrateBounds,
                                             dr_emissionrateBounds=dr_emissionrateBounds,
                                             dr_Classification="dr_Classification",
                                             freq=freq)
        end_time = time.time()
        print(f"Processing data for quantification analysis code executed in {end_time - start_time:.6f} seconds")
    except Exception as e:
        pairedDF = pd.DataFrame()
        print(f"Could not processing data for quantification analysis due to the exception: {e}")
        raise

    # ### Step 8: Apply Thresholding ###
    try:
        start_time = time.time()
        DRDF, CRDF = thresholdingApplication(CRDF=CRDF,
                                             DRDF=DRDF,
                                             rs_CRDF=rs_CRDF,
                                             rs_DRDF=rs_DRDF,
                                             detection=detection,
                                             cr_releaseRate=cr_releaseRate,
                                             dr_releaseRate=dr_releaseRate,
                                             ThresholdDuration=ThresholdDuration,
                                             ThresholdRateAboveBaseline=ThresholdRateAboveBaseline,
                                             tc_baselineEmissionRate=tc_baselineEmissionRate,
                                             dr_baselineEmissionRate=dr_baselineEmissionRate,
                                             freq=freq,
                                             Method=Method)
        end_time = time.time()
        print(f"Thresholding code executed in {end_time - start_time:.6f} seconds")
    except Exception as e:
        print(f"Thresholding step failed with error: {e}")
        raise

    ### Step 8: Compare and Classify ###
    try:
        start_time = time.time()
        ClassifiedDataFrame, CRDF, DRDF = compareDataFrames(CRDF=CRDF,
                                                            DRDF=DRDF)
        end_time = time.time()
        print(f"Pairing controlled release with detection report code executed in {end_time - start_time:.6f} seconds")
    except Exception as e:
        print(f"Could not pair data after thresholding due to the exception: {e}")
        raise

    ### Step 9: Final Output Preparation ###
    try: # Combine index into a single column and reset index.
        ClassifiedDataFrame["Time Series"] = ClassifiedDataFrame.index
        ClassifiedDataFrame = ClassifiedDataFrame.reset_index(drop=True)
    except Exception as e:
        print(f"ClassifiedDataFrame does not have its index in form of datetime due to {e}")
        raise

    ### Step 10: Re-initialize selected variables
    listOfVariables2ListUnique_CR.remove("tc_Classification")
    listOfVariables2ListUnique_DR.remove("dr_Classification")

    return ClassifiedDataFrame, pairedDF, DRDF, CRDF
