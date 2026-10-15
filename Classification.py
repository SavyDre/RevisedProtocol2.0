import pytz
import math
import pandas as pd
import numpy as np
import time
import warnings
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
        ASE_distances = [geodesic(reference_coord, (lat, lon)).meters for lat, lon in list(ASE_polygon.exterior.coords)]
        ASE_bearings = [math.radians(calc_bearing(cord1=reference_coord, cord2=(lat, lon))) for lat, lon in list(ASE_polygon.exterior.coords)]

        # Calculate distances and bearings for subASE
        subASE_distances = [geodesic(reference_coord, (lat, lon)).meters for lat, lon in list(subASE_polygon.exterior.coords)]
        subASE_bearings = [math.radians(calc_bearing(cord1=reference_coord, cord2=(lat, lon))) for lat, lon in list(subASE_polygon.exterior.coords)]

        # Convert distances and bearings to Cartesian coordinates
        coords_cartesian_ASE = [(r * math.sin(theta), r * math.cos(theta)) for r, theta in zip(ASE_distances, ASE_bearings)]
        coords_cartesian_subASE = [(r * math.sin(theta), r * math.cos(theta)) for r, theta in zip(subASE_distances, subASE_bearings)]

        # Create polygons using the calculated Cartesian coordinates
        polygon_cartesian_ASE = Polygon(coords_cartesian_ASE)
        polygon_cartesian_subASE = Polygon(coords_cartesian_subASE)
        for polygon in (polygon_cartesian_ASE, polygon_cartesian_subASE):
            if not polygon.is_valid or polygon.is_empty:
                raise ValueError('Spatial polygons must be valid and nonempty.')

        # Buffer polygon edges in meters. Adding distance to each vertex radius
        # instead changes the polygon relative to the reference point.
        polygon_cartesian_ASE = polygon_cartesian_ASE.buffer(float(ASE_bufferDistance), join_style=2)
        polygon_cartesian_subASE = polygon_cartesian_subASE.buffer(float(subASE_bufferDistance), join_style=2)

        # Calculate intersection and union areas
        polygons_intersect = polygon_cartesian_ASE.intersection(polygon_cartesian_subASE)

        # Check if the height ranges overlap
        if not pd.isna(BoundingZsubASE_min) and not pd.isna(BoundingZsubASE_max):
            height_overlap = (BoundingZsubASE_min >= BoundingZASE_min) and (BoundingZsubASE_max <= BoundingZASE_max)
        else:
            print("BoundingZMinimum and BoundingZMaximum are not provided, hence the minimum and maximum values defined for this ASE is used.")
            height_overlap = True
        if polygons_intersect and height_overlap:
            # Calculate intersection and union areas
            intersection_area = polygons_intersect.area
            union_area = polygon_cartesian_ASE.union(polygon_cartesian_subASE).area
            # Calculate degree of overlap
            overlap_degree = (intersection_area / union_area) if union_area > 0 else 0.0
            return overlap_degree
        else:
            return 0.0  # Return 0 if no intersection or height bounds don't overlap

    except Exception as e:
        raise ValueError(f'Invalid spatial bounds: {e}') from e


def extractSpatialSubsetData(DataDF, ASE, DataDF_PolygonCoord, DataDF_BoundingZMin, DataDF_BoundingZMax, DataDF_localization,
                             ASE_PolygonCoord_Dict, ASE_textual_mapper, overLappingFrac=0, ASE_bufferDistance=0,
                             subASE_bufferDistance=0):
    """
        Filters a DataFrame to return rows that fall within a specified spatial bounding area.

        Parameters:
        - DataDF (DataFrame): The input DataFrame containing spatial data to be filtered.
        - ASE (str): The analysis spatial extent defined for the analysis.
        - DataDF_PolygonCoord (str): Column name in DataDF containing WKT format polygons.
        - DataDF_BoundingZMin (str): Column name in DataDF specifying minimum height of each entry.
        - DataDF_BoundingZMax (str): Column name in DataDF specifying maximum height of each entry.
        - DataDF_localization (str): Column name in DataDF for identifiers used if polygon data is absent.
        - ASE_PolygonCoord_Dict (dict): A dictionary consisting of the parameters of the ASEs defined by the test center for the analysis.
                                        For the defined ASE, the "BoundingXY" for the polygon (WKT format) and "BoundingZ" for height bounds is provided.
                                        For example, {"Wellpad1": {"BoundingXY": "POLYGON((lat_1 lon_1, ..., lat_n lon_n))",
                                             "BoundingZ": ["MinimumZ", "MaximumZ"], "ReferenceCoord": ["lat", "lon"]}}
        - ASE_textual_mapper (dict): Mapping of DataDF_localization identifiers to spatial properties (polygon and height).
                                        For example, {"IT-1": {"BoundingXY": "POLYGON((lat_1 lon_1, ..., lat_n lon_n))",
                                             "BoundingZ": ["MinimumZ", "MaximumZ"], "ReferenceCoord": ["lat", "lon"]}}
        - overLappingFrac (float): Minimum intersection fraction required for filtering (default is 0).
        - ASE_bufferDistance (float): The buffer distance for the ASE polygon (meters)
        - subASE_bufferDistance (float): The buffer distance for the subASE polygon (meters)
        Returns:
        - DataFrame: A filtered subset of DataDF where rows intersect the ASE bounding area by the specified fraction.
    """
    # Preserve an empty report schema for the later missing-report alignment.
    if DataDF.empty:
        return DataDF.copy()
    DataDF = DataDF.copy(deep=True)

    # Retrieve bounding polygon and height range from ASE_PolygonCoord_Dict
    ASE_BoundingXY = ASE_PolygonCoord_Dict[ASE]["BoundingXY"]
    ASE_ReferenceCoord = ASE_PolygonCoord_Dict[ASE]["ReferenceCoord"]
    ASE_BoundingZMinimum = ASE_PolygonCoord_Dict[ASE]["BoundingZ"][0]
    ASE_BoundingZMaximum = ASE_PolygonCoord_Dict[ASE]["BoundingZ"][1]

    # Case 1: If DataDF contains polygon coordinates, apply spatial filtering based on polygons
    if (DataDF_PolygonCoord in DataDF.columns) and not pd.isna(ASE_BoundingXY):
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
    elif (DataDF_localization in DataDF.columns) and not pd.isna(ASE_BoundingXY):
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

    # Case 3: If DataDF contains localization data, use textual mapping for spatial filtering
    elif (DataDF_localization in DataDF.columns) and pd.isna(ASE_BoundingXY):
        Data4thisASE = DataDF.loc[DataDF[DataDF_localization].eq(ASE)].copy()

    # Case 4: If no spatial data is present, return a copy of the entire DataFrame without filtering
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
        Round a datetime to the nearest frequency grid point (ties to even).
        Parameters
            - dt : The datetime to be rounded.
            - freq : Frequency string (e.g., 's', 'min', '5min', '10min', '15min', '30min', 'h', 'D', etc).
        Returns
            - datetime: The rounded-up datetime.
    """

    # Round to the nearest grid point, preserving an aware timestamp's zone.
    return pd.Timestamp(dt).round(freq)


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
        Expand intervals to samples while supporting arbitrary column names.

        Samples normally represent interval ends, i.e. (start, end]. Repeat rows in
        column blocks instead of constructing a Python dictionary per time step.
        Empty inputs retain their schema and a UTC DatetimeIndex for later alignment.
    """
    columns = list(dict.fromkeys([startTime, endTime] + list(variables or [])))
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns for temporal expansion: {sorted(missing)}')
    frames = []
    for values in df[columns].itertuples(index=False, name=None):
        record = dict(zip(columns, values))
        index = pd.date_range(record[startTime], record[endTime], freq=freq, inclusive=inclusive)
        if len(index):
            frames.append(pd.DataFrame({c: [record[c]] * len(index) for c in columns}, index=index))
    if not frames:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz='UTC', name='Time'))
    expanded_df = pd.concat(frames)
    expanded_df.index.name = 'Time'
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
    df.loc[rate_nonzero, 'Cumulative Duration'] = (rate_nonzero[rate_nonzero].groupby(rate_nonzero.ne(rate_nonzero.shift()).cumsum()).transform(lambda g: ((g.cumsum()).clip(lower=0)) * df.loc[g.index, 'time_diffs']))

    # Keep time_diffs: operational factors use these per-sample durations.
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
        'last': kwargs.get('listOfVariables2ObtainLast', [])}

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
                ordered = DataDF.sort_values(ID_column, kind='stable')
                aggregated_data[f'{var}'] = ordered.resample(rule=freq)[var].agg(lambda values: values.iloc[0] if len(values) else None)
                print("Successfully completed the extraction of the first value operation")

            elif operation == 'last':
                # Get the last value based on ID
                print(f"Performing the extraction of the last value operation for the column header: {var} using {ID_column} as unique ID.")
                ordered = DataDF.sort_values(ID_column, kind='stable')
                aggregated_data[f'{var}'] = ordered.resample(rule=freq)[var].agg(lambda values: values.iloc[-1] if len(values) else None)
                print("Successfully completed the extraction of the last value operation")

    # Combine the aggregated results into a single DataFrame
    aggregated_df = pd.DataFrame(aggregated_data)
    if 'dr_Classification' in DataDF:
        # Empty bins between reports are also unreported time steps.
        counts = DataDF.resample(rule=freq).size()
        aggregated_df['OriginalReport'] = np.where(counts.reindex(aggregated_df.index, fill_value=0) > 0, 'True', 'False')

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
        try:
            # Calculate the rolling average over the given window size and frequency; Compare rolling averages to the threshold
            self.TimeSeriesData[rolling_avg_column] = self.TimeSeriesData[value_column].rolling(pd.tseries.frequencies.to_offset(freq) * RollingAverageWindow).mean()
            self.TimeSeriesData[str(ThresholdColumn)] = self.TimeSeriesData[rolling_avg_column].apply(lambda x: None if pd.isna(x) else (1 if x > Total else 0))
            return self.TimeSeriesData
        except Exception as e:
            print(f"Could not perform rolling average on time series data due to: {e}")
            return pd.DataFrame()

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
        try:
            # Apply threshold condition using both rate and cumulative duration
            self.TimeSeriesData[str(ThresholdColumn)] = self.TimeSeriesData.apply(lambda row: None if pd.isna(row[rateColumnName]) or pd.isna(row['Cumulative Duration']) \
                                                                else (1 if row[rateColumnName] > Total and row['Cumulative Duration'] > DetectionDurationThreshold else 0),
                                                                axis=1)
        except Exception as e:
            # Apply threshold condition using both rate and cumulative duration
            self.TimeSeriesData[str(ThresholdColumn)] = self.TimeSeriesData.apply(lambda row: None if pd.isna(row[rateColumnName]) or pd.isna(row['Duration']) \
                    else (1 if row[rateColumnName] > Total and row['Duration'] > DetectionDurationThreshold else 0), axis=1)
        return self.TimeSeriesData

def align_time_indexes(crdf, drdf, dr_columns, detection, analysis_starttime, analysis_endtime, freq):
    """
        Align controlled-release and detection-report data to the full analysis
        time grid.

        Parameters
        ----------
        crdf : pd.DataFrame
            Controlled-release time-series dataframe.
        drdf : pd.DataFrame
            Detection-report time-series dataframe.
        dr_columns : list
            Detection-report columns that should be set to zero when no
            detection report exists at a time step.
        detection : str
            Detection indicator column.
        analysis_starttime : datetime-like
            Analysis start time supplied by -s.
        analysis_endtime : datetime-like
            Analysis end time supplied by -e.
        freq : str
            Analysis temporal resolution, e.g. "30s", "1min".
        Returns
        -------
        crdf, drdf : tuple[pd.DataFrame, pd.DataFrame]
            CR and DR dataframes aligned to the same analysis time grid.
    """

    crdf = crdf.copy()
    drdf = drdf.copy()

    # ---------------------------------------------------------
    # 1. Standardize all timestamps to UTC
    # ---------------------------------------------------------

    crdf.index = pd.to_datetime(crdf.index, utc=True)
    drdf.index = pd.to_datetime(drdf.index, utc=True)

    analysis_starttime = pd.to_datetime(analysis_starttime, utc=True)

    analysis_endtime = pd.to_datetime(analysis_endtime, utc=True)

    # ---------------------------------------------------------
    # 2. Validate inputs
    # ---------------------------------------------------------
    if analysis_starttime >= analysis_endtime:
        raise ValueError("analysis_starttime must be earlier than analysis_endtime.")

    if crdf.index.has_duplicates:
        raise ValueError("CRDF contains duplicate timestamps. Aggregate duplicate time steps before alignment.")

    if drdf.index.has_duplicates:
        raise ValueError("DRDF contains duplicate timestamps. Aggregate duplicate time steps before alignment.")

    # ---------------------------------------------------------
    # 3. Construct the authoritative analysis grid the pipeline uses (start, end] samples, so use the same convention here.
    # ---------------------------------------------------------
    analysis_index = pd.date_range(start=analysis_starttime, end=analysis_endtime, freq=freq,inclusive="right")
    analysis_index.name = "Time"

    # ---------------------------------------------------------
    # 4. Determine which timestamps originally contained data BEFORE reindexing.
    # ---------------------------------------------------------
    cr_original_index = crdf.index.copy()
    dr_original_index = drdf.index.copy()
    cr_missing = ~analysis_index.isin(cr_original_index)
    dr_missing = ~analysis_index.isin(dr_original_index)

    # ---------------------------------------------------------
    # 5. Reindex BOTH datasets independently onto the same analysis grid.
    # IMPORTANT: DRDF is no longer reindexed using CRDF.index.
    # ---------------------------------------------------------
    crdf = crdf.reindex(analysis_index)
    drdf = drdf.reindex(analysis_index)

    # ---------------------------------------------------------
    # 6. Mark timestamps where no original DR report existed
    # ---------------------------------------------------------
    if "OriginalReport" not in drdf.columns:
        drdf["OriginalReport"] = "True"

    # Existing DR timestamps are genuine reports.
    drdf.loc[~dr_missing, "OriginalReport"] = "True"

    # Newly created timestamps are not original reports.
    drdf.loc[dr_missing, "OriginalReport"] = "False"

    # ---------------------------------------------------------
    # 7. Set DR rate/detection variables to zero only for timestamps where no original report existed.
    # Do NOT blanket-fill NaNs because some NaNs may carry analytical meaning.
    # ---------------------------------------------------------
    columns_to_zero = list(dr_columns)
    if detection is not None:
        columns_to_zero.append(detection)

    for column in columns_to_zero:
        if column is not None and column in drdf.columns:
            drdf.loc[dr_missing, column] = 0

    # ---------------------------------------------------------
    # 8. Preserve classification semantics
    # ---------------------------------------------------------
    if "dr_Classification" not in drdf.columns:
        drdf["dr_Classification"] = None

    # ---------------------------------------------------------
    # 9. Optional CR bookkeeping
    # ---------------------------------------------------------
    if "OriginalRelease" not in crdf.columns:
        crdf["OriginalRelease"] = "True"

    crdf.loc[~cr_missing, "OriginalRelease"] = "True"
    crdf.loc[cr_missing, "OriginalRelease"] = "False"

    # ---------------------------------------------------------
    # 10. Sanity checks
    # ---------------------------------------------------------
    if not crdf.index.equals(drdf.index):
        raise RuntimeError("CRDF and DRDF indexes failed to align.")

    if not crdf.index.equals(analysis_index):
        raise RuntimeError("Output index does not match requested analysis interval.")

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
            - freq: Sampling frequency (retained for API compatibility); buffers are always minutes.
            - bufferTime: A tuple (buffer_start, buffer_end) specifying additional time to extend exclusion intervals before applying flags.

        Returns:
            - DataFrame: The modified DataFrame with flagged rows in the 'classification' column.
    """

    # The public API defines buffers in minutes, independently of sampling frequency.
    unit = 'min'

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
                                                            freq=freq, RollingAverageThreshold=RollingAverageThreshold)

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


def thresholdingApplication(CRDF, DRDF, detection, cr_releaseRate, dr_releaseRate, ThresholdRateAboveBaseline,
                            ThresholdDuration, tc_baselineEmissionRate, dr_baselineEmissionRate, freq, Method="Rolling Average"):
    """
        Apply thresholding techniques to detection and controlled release data.

        Parameters:
            - CRDF (pd.DataFrame): DataFrame containing controlled release data (e.g., test center data).
            - DRDF (pd.DataFrame): DataFrame containing detection report data.
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
    column_exists, column_exists_and_has_values = check_if_dataframe_column_exists(df=DRDF, column_name=detection)

    # Time-series durations are accumulated in seconds. Summarized windows
    # already carry their physical duration; do not pretend rows are time steps.
    if isinstance(DRDF.index, pd.DatetimeIndex) and isinstance(CRDF.index, pd.DatetimeIndex):
        DRDF = add_cumulative_duration(df=DRDF, rate_col=dr_releaseRate, freq=freq)
        CRDF = add_cumulative_duration(df=CRDF, rate_col=cr_releaseRate, freq=freq)
    elif Method == "Rolling Average":
        # Rolling windows need datetime indexes even for summarized inputs.
        DRDF = DRDF.set_index('dr_EndTime', drop=False)
        CRDF = CRDF.set_index('cr_EndTime', drop=False)

    # Step 3: Thresholding controlled release and detection report data
    if Method == "Rolling Average":
        # Thresholding using the default rolling average method.
        DRDF, CRDF = DefaultRollingAvgApplication(DRDF=DRDF, CRDF=CRDF,
                                                  detection=detection,
                                                  dr_releaseRate=dr_releaseRate,
                                                  cr_releaseRate=cr_releaseRate,
                                                  dr_baselineEmissionRate=dr_baselineEmissionRate,
                                                  tc_baselineEmissionRate=tc_baselineEmissionRate,
                                                  RollingAverageThreshold=ThresholdRateAboveBaseline,
                                                  RollingAverageWindow=ThresholdDuration,
                                                  freq=freq, column_exists=column_exists,
                                                  column_exists_and_has_values=column_exists_and_has_values)

        return DRDF, CRDF

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

    raise ValueError(f"Unsupported threshold method: {Method!r}. Use 'Rolling Average' or 'Above Threshold Across Duration'.")


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

    CRDF[tc_Classification] = CRDF[tc_Classification].astype(object)
    DRDF[dr_Classification] = DRDF[dr_Classification].astype(object)

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

def compareSummarizedData(CRDF, DRDF, tc_Detected="tc_Detection", dr_Detected="dr_Detection",
                          tc_Classification="tc_Classification", dr_Classification="dr_Classification"):
    """
        Compare controlled-release (CR) and detection-report (DR) periods.
        Time interval convention
        ------------------------
        Windows are interpreted as: (StartTime, EndTime]
        Therefore two intervals overlap when: dr_EndTime > cr_StartTime AND cr_EndTime > dr_StartTime
        Merely touching at a boundary is NOT considered overlap.
        Classification rules
        --------------------
        For overlapping CR/DR rows:
            CR detected > 0, DR detected > 0 -> TP for both rows
            CR detected > 0, DR detected <= 0 or NaN -> FN for CR row
            CR detected <= 0 or NaN, DR detected > 0 -> FP for DR row
            CR detected <= 0 or NaN, DR detected <= 0 or NaN -> TN for both rows

        For rows with no overlap: unmatched CR -> FN; unmatched DR -> FP
        Existing non-null classifications are preserved.

        Returns pairedDF (pd.DataFrame): One row for every overlapping CR/DR pair plus unmatched rows.
                CRDF (pd.DataFrame): Classified CR dataframe.
                DRDF (pd.DataFrame): Classified DR dataframe.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Work on copies so the original input DataFrames are not modified.
    # ------------------------------------------------------------------------------------------------------------------
    CRDF = CRDF.copy()
    DRDF = DRDF.copy()

    # ------------------------------------------------------------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------------------------------------------------------------
    required_cr = {"cr_StartTime", "cr_EndTime", tc_Detected}
    required_dr = {"dr_StartTime", "dr_EndTime", dr_Detected}

    missing_cr = required_cr - set(CRDF.columns)
    missing_dr = required_dr - set(DRDF.columns)

    if missing_cr:
        raise KeyError(f"CRDF is missing required column(s): {sorted(missing_cr)}")
    if missing_dr:
        raise KeyError(f"DRDF is missing required column(s): {sorted(missing_dr)}")

    # ------------------------------------------------------------------------------------------------------------------
    # Create classification columns if they do not already exist.
    # ------------------------------------------------------------------------------------------------------------------
    if tc_Classification not in CRDF.columns:
        CRDF[tc_Classification] = None
    if dr_Classification not in DRDF.columns:
        DRDF[dr_Classification] = None

    # ------------------------------------------------------------------------------------------------------------------
    # Convert times to datetime.
    # ------------------------------------------------------------------------------------------------------------------
    CRDF[tc_Classification] = CRDF[tc_Classification].astype(object)
    DRDF[dr_Classification] = DRDF[dr_Classification].astype(object)

    CRDF["cr_StartTime"] = pd.to_datetime(CRDF["cr_StartTime"], errors="raise")
    CRDF["cr_EndTime"] = pd.to_datetime(CRDF["cr_EndTime"], errors="raise")
    DRDF["dr_StartTime"] = pd.to_datetime(DRDF["dr_StartTime"], errors="raise")
    DRDF["dr_EndTime"] = pd.to_datetime(DRDF["dr_EndTime"], errors="raise")

    # ------------------------------------------------------------------------------------------------------------------
    # Preserve original indices.
    # These are useful for keeping track of paired rows.
    # ------------------------------------------------------------------------------------------------------------------
    CRDF = CRDF.reset_index(drop=True)
    DRDF = DRDF.reset_index(drop=True)
    CRDF["cr_index"] = CRDF.index
    DRDF["dr_index"] = DRDF.index
    original_cr_flags = CRDF[tc_Classification].copy()
    original_dr_flags = DRDF[dr_Classification].copy()

    # ------------------------------------------------------------------------------------------------------------------
    # Create all candidate CR / DR combinations.
    # This replaces the nested iterrows() loops and avoids repeated concat.
    # ------------------------------------------------------------------------------------------------------------------
    cr_candidates = CRDF[["cr_index", "cr_StartTime", "cr_EndTime"]].copy()
    dr_candidates = DRDF[["dr_index", "dr_StartTime", "dr_EndTime"]].copy()
    dr_candidates = dr_candidates.sort_values('dr_StartTime')
    starts = pd.DatetimeIndex(dr_candidates['dr_StartTime'])
    pairs = []
    for row in cr_candidates.itertuples(index=False):
        # For (start, end] intervals, touching boundaries do not overlap.
        stop = starts.searchsorted(row.cr_EndTime, side='left')
        candidates = dr_candidates.iloc[:stop]
        for dr_index in candidates.loc[candidates['dr_EndTime'] > row.cr_StartTime, 'dr_index']:
            pairs.append((row.cr_index, dr_index))
    overlaps = pd.DataFrame(pairs, columns=['cr_index', 'dr_index'])

    # ------------------------------------------------------------------------------------------------------------------
    # Determine which CR and DR rows participate in at least one match.
    # ------------------------------------------------------------------------------------------------------------------
    matched_cr_indices = set(overlaps["cr_index"])
    matched_dr_indices = set(overlaps["dr_index"])

    # ==================================================================================================================
    # Classify OVERLAPPING pairs
    # ==================================================================================================================
    for pair in overlaps.itertuples(index=False):
        c_idx = pair.cr_index
        d_idx = pair.dr_index
        tc_detection = CRDF.loc[c_idx, tc_Detected]
        dr_detection = DRDF.loc[d_idx, dr_Detected]

        # ---------------------------------------------------------------------------------------------------------------
        # Detection status: NaN is treated the same as <= 0.
        # ---------------------------------------------------------------------------------------------------------------
        tc_positive = (pd.notna(tc_detection) and tc_detection > 0)
        dr_positive = (pd.notna(dr_detection) and dr_detection > 0)

        # ---------------------------------------------------------------------------------------------------------------
        # TP
        # ---------------------------------------------------------------------------------------------------------------
        if tc_positive and dr_positive:
            value = CRDF.at[c_idx, tc_Classification]
            if not isinstance(value, (list, np.ndarray)) and pd.isna(value):
                CRDF.at[c_idx, tc_Classification] = "TP"

            value = DRDF.at[d_idx, dr_Classification]
            if not isinstance(value, (list, np.ndarray)) and pd.isna(value):
                DRDF.at[d_idx, dr_Classification] = "TP"

        # ---------------------------------------------------------------------------------------------------------------
        # FN: CR exists/detected but DR is not detected.
        # ---------------------------------------------------------------------------------------------------------------
        elif tc_positive and not dr_positive:
            value = CRDF.at[c_idx, tc_Classification]
            if not isinstance(value, (list, np.ndarray)) and pd.isna(value):
                CRDF.at[c_idx, tc_Classification] = "FN"

        # --------------------------------------------------------------------------------------------------------------
        # FP: DR reports detection but CR is absent/not detected.
        # --------------------------------------------------------------------------------------------------------------
        elif not tc_positive and dr_positive:
            value = DRDF.at[d_idx, dr_Classification]
            if not isinstance(value, (list, np.ndarray)) and pd.isna(value):
                DRDF.at[d_idx, dr_Classification] = "FP"

        # ---------------------------------------------------------------------------------------------------------------
        # TN
        # ---------------------------------------------------------------------------------------------------------------
        else:
            value = CRDF.at[c_idx, tc_Classification]
            if not isinstance(value, (list, np.ndarray)) and pd.isna(value):
                CRDF.at[c_idx, tc_Classification] = "TN"
            value = DRDF.at[d_idx, dr_Classification]
            if not isinstance(value, (list, np.ndarray)) and pd.isna(value):
                DRDF.at[d_idx, dr_Classification] = "TN"

    # ==================================================================================================================
    # Classify UNMATCHED CR rows
    # ==================================================================================================================
    unmatched_cr = (~CRDF["cr_index"].isin(matched_cr_indices))
    CRDF.loc[unmatched_cr & CRDF[tc_Classification].isna(), tc_Classification] = "FN"

    # ==================================================================================================================
    # Classify UNMATCHED DR rows
    # ==================================================================================================================
    unmatched_dr = (~DRDF["dr_index"].isin(matched_dr_indices))
    DRDF.loc[unmatched_dr & DRDF[dr_Classification].isna(), dr_Classification] = "FP"

    # ==================================================================================================================
    # Build pairedDF
    # One output row for every overlapping CR/DR combination.
    # ==================================================================================================================
    if not overlaps.empty:
        pairedDF = overlaps.merge(CRDF, on="cr_index", how="left", suffixes=("", "_cr")).merge(DRDF, on="dr_index",
                                   how="left", suffixes=("_cr", "_dr"))

    else:
        pairedDF = pd.DataFrame()

    # ===================================================================================================================
    # Add unmatched CR rows
    # ===================================================================================================================
    unmatched_CRDF = CRDF.loc[unmatched_cr].copy()

    # ===================================================================================================================
    # Add unmatched DR rows
    # ===================================================================================================================
    unmatched_DRDF = DRDF.loc[unmatched_dr].copy()

    # -------------------------------------------------------------------------------------------------------------------
    # Add missing columns so all records can be concatenated cleanly.
    # -------------------------------------------------------------------------------------------------------------------
    frames = []
    if not pairedDF.empty:
        frames.append(pairedDF)
    if not unmatched_CRDF.empty:
        frames.append(unmatched_CRDF)
    if not unmatched_DRDF.empty:
        frames.append(unmatched_DRDF)

    if frames:
        pairedDF = pd.concat(frames, ignore_index=True, sort=False)
    else:
        pairedDF = pd.DataFrame()

    # ==================================================================================================================
    # Unified classification
    # ==================================================================================================================
    if tc_Classification in pairedDF.columns:
        tc_class = pairedDF[tc_Classification]
    elif f"{tc_Classification}_cr" in pairedDF.columns:
        tc_class = pairedDF[f"{tc_Classification}_cr"]
    else:
        tc_class = pd.Series(None, index=pairedDF.index)

    if dr_Classification in pairedDF.columns:
        dr_class = pairedDF[dr_Classification]
    elif f"{dr_Classification}_dr" in pairedDF.columns:
        dr_class = pairedDF[f"{dr_Classification}_dr"]
    else:
        dr_class = pd.Series(None, index=pairedDF.index)

    def existing_flag(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return '; '.join(str(v) for v in value if pd.notna(v)) or None
        return None if pd.isna(value) else value

    classifications = []
    for _, pair in pairedDF.iterrows():
        ci, di = pair.get('cr_index'), pair.get('dr_index')
        cflag = existing_flag(original_cr_flags.loc[ci]) if pd.notna(ci) else None
        dflag = existing_flag(original_dr_flags.loc[di]) if pd.notna(di) else None
        if cflag is not None or dflag is not None:
            classifications.append(cflag if cflag is not None else dflag)
            continue
        cvalue = CRDF.at[ci, tc_Detected] if pd.notna(ci) else 0
        dvalue = DRDF.at[di, dr_Detected] if pd.notna(di) else 0
        cp = pd.notna(cvalue) and cvalue > 0
        dp = pd.notna(dvalue) and dvalue > 0
        classifications.append('TP' if cp and dp else 'FN' if cp else 'FP' if dp else 'TN')
    pairedDF['Classification'] = classifications

    # Returned per-source summaries are deterministic too: any valid TP match
    # takes precedence over a miss against another overlapping window.
    for frame, key, column, originals, priority in (
        (CRDF, 'cr_index', tc_Classification, original_cr_flags, ('TP', 'FN', 'TN')),
        (DRDF, 'dr_index', dr_Classification, original_dr_flags, ('TP', 'FP', 'TN'))):
        for index in frame.index:
            if existing_flag(originals.loc[index]) is not None:
                continue
            labels = set(pairedDF.loc[pairedDF[key].eq(index), 'Classification']) if key in pairedDF else set()
            frame.at[index, column] = next((label for label in priority if label in labels), None)


    # ===================================================================================================================
    # Cleanup
    # ===================================================================================================================
    helper_columns = ["cr_index", "dr_index"]
    pairedDF.drop(columns=[col for col in helper_columns if col in pairedDF.columns], inplace=True, errors="ignore")
    pairedDF.drop(["DR_RunID_dr", "DR_RunID_cr", "cr_NSamples", "dr_NSamples", "dr_RequiresCRSplit", "CR_SubrunID", "cr_HasData", "cr_WasSplit"], axis=1, inplace=True, errors="ignore")
    CRDF.drop(columns=["cr_index", "DR_RunID", "CR_SubrunID", "cr_HasData", "cr_NSamples", "cr_WasSplit"], inplace=True, errors="ignore")
    DRDF.drop(columns=["dr_index", "DR_RunID", "dr_NSamples", "dr_RequiresCRSplit"], inplace=True, errors="ignore")

    return pairedDF, CRDF, DRDF

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

def _validate_frequency(freq):
    """
        Return positive fixed sampling duration and seconds; reject NaT.

        Parameters
        ----------
        freq: Sampling interval accepted by pandas.Timedelta, e.g. "1min"; must be finite and positive.

        Local variables and comprehension/loop names
        --------------------------------------------
        td: Validated sample duration as a pandas Timedelta.
        seconds: Same duration as a floating-point number of seconds.

        Returns
        -------
        tuple: (td, seconds), representing the same positive sample interval.
    """
    # Set td: Validated sample duration as a pandas Timedelta.
    td = pd.Timedelta(freq)
    # Set seconds: Same duration as a floating-point number of seconds.
    seconds = td.total_seconds()
    # Reject non-finite or nonpositive durations/conversion factors.
    if not np.isfinite(seconds) or seconds <= 0:
        # Stop with an explicit error so invalid data/configuration cannot silently affect
        # results.
        raise ValueError('freq must be a finite positive duration.')
    # Return tuple: (td, seconds), representing the same positive sample interval.
    return (td, seconds)

def _prepare(df, freq, timefactor):
    """
        Copy, sort and validate the regularly spaced interval-ending input.

        Parameters
        ----------
        df: Input DataFrame; its interpretation depends on the helper (samples or selected values).
        freq: Sampling interval accepted by pandas.Timedelta, e.g. "1min"; must be finite and positive.
        timefactor: Positive time-basis conversion for rate integration: 1/3600 for rates per hour, 1 for rates per second.

        Local variables and comprehension/loop names
        --------------------------------------------
        td: Validated interval Timedelta returned by _validate_frequency.
        seconds: Validated interval length in seconds.
        work: Copied, datetime-indexed, chronologically sorted input used for checks and calculations.

        Returns
        -------
        tuple: (work, td, seconds); copied validated observations and sample duration.
    """
    # Set td: Validated interval Timedelta returned by _validate_frequency.; seconds: Validated interval length in seconds.
    td, seconds = _validate_frequency(freq)
    # Reject non-finite or nonpositive durations/conversion factors.
    if not np.isfinite(timefactor) or timefactor <= 0:
        # Stop with an explicit error so invalid data/configuration cannot silently affect results.
        raise ValueError('timefactor must be finite and positive.')
    # Set work: Copied, datetime-indexed, chronologically sorted input used for checks and calculations.
    work = df.copy(deep=True)
    # Parse the copied index as datetimes without converting its timezone.
    work.index = pd.to_datetime(work.index)
    # Reject missing/duplicate timestamps and duplicate source column names.
    if work.index.hasnans or work.index.has_duplicates or (not work.columns.is_unique):
        # Stop with an explicit error so invalid data/configuration cannot silently affect results.
        raise ValueError('Timestamps must be unique/non-missing and columns unique.')
    # Sort the copied observations chronologically before comparing adjacent rows.
    work.sort_index(inplace=True)
    # Require the declared sampling interval at every consecutive timestamp.
    if len(work) > 1 and (not work.index.to_series().diff().iloc[1:].eq(td).all()):
        # Stop with an explicit error so invalid data/configuration cannot silently affect
        # results.
        raise ValueError('Input timestamp spacing must equal freq.')
    # Return tuple: (work, td, seconds); copied validated observations and sample duration.
    return (work, td, seconds)

def _is_null(value):
    """
        Test scalar missingness without ambiguous array truth values.

        Parameters
        ----------
        value: Cell value to test for scalar missingness; containers are treated as values, not missing scalars.
        Local variables and comprehension/loop names
        --------------------------------------------
        result: Scalar Boolean or array-like result of pandas.isna; only scalar results become a Boolean.

        Returns
        -------
        bool: True only when the supplied scalar is missing.
    """
    # Set result: Scalar Boolean or array-like result of pandas.isna; only scalar results become
    # a Boolean.
    result = pd.isna(value)
    # Return bool: True only when the supplied scalar is missing.
    return bool(result) if np.ndim(result) == 0 else False


def _equal(a, b):
    """
        Exact nested-cell equality, with missing scalars considered equal.

        Parameters
        ----------
        a: Left cell value in an exact comparison; may be scalar or a supported nested container.
        b: Right cell value in an exact comparison; consecutive missing scalars compare equal.

        Local variables and comprehension/loop names
        --------------------------------------------
        k: Dictionary key used to compare corresponding nested values.
        x: Left sequence element in a paired comparison.
        y: Right sequence element corresponding to x.

        Returns
        -------
        bool: exact equality under the supported nested/null comparison rules.
    """
    # Normalize a NumPy array into Python values for recursive comparison.
    if isinstance(a, np.ndarray):
        # Set a: value described in this function docstring
        a = a.tolist()
    # Normalize a NumPy array into Python values for recursive comparison.
    if isinstance(b, np.ndarray):
        # Set b: value described in this function docstring
        b = b.tolist()
    # Compare dictionaries by keys and corresponding nested values.
    if isinstance(a, dict) or isinstance(b, dict):
        # Return bool: exact equality under the supported nested/null comparison rules.
        return isinstance(a, dict) and isinstance(b, dict) and (a.keys() == b.keys()) and all((_equal(a[k], b[k]) for k in a))
    # Handle ordered or supported nested container elements recursively.
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        # Return bool: exact equality under the supported nested/null comparison rules.
        return isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and (len(a) == len(b)) and all((_equal(x, y) for x, y in zip(a, b)))
    # Treat two missing scalars as equal and a known/missing transition as different.
    if _is_null(a) or _is_null(b):
        # Return bool: exact equality under the supported nested/null comparison rules.
        return _is_null(a) and _is_null(b)
    # Return bool: exact equality under the supported nested/null comparison rules.
    return bool(a == b)

def _rows_changed(df, columns):
    """
        Mark run boundaries; consecutive nulls equal, null/known transitions differ.

        Nullable comparisons yield pd.NA, so explicitly fill equality with False.
        Nested values use recursive comparisons rather than ambiguous array truth.

        Parameters
        ----------
        df: Input DataFrame; its interpretation depends on the helper (samples or selected values).
        columns: Ordered source column names selected for presence checks or consecutive-row comparisons.

        Local variables and comprehension/loop names
        --------------------------------------------
        c: Selected column name in the nested-value scan.
        v: Cell value examined to detect containers requiring recursive equality.
        rows: Selected observations represented as ordered tuples.
        flags: Boolean run-start flags from recursive comparisons; first observation is always True.
        p: Previous observation tuple in consecutive comparisons.
        r: Current observation tuple corresponding to p.
        a: Previous cell value passed to _equal.
        b: Current cell value passed to _equal.
        curr: DataFrame restricted to selected comparison columns.
        prev: curr shifted one row so each observation aligns with its predecessor.
        same: Per-cell equality flags, including consecutive null equality and nullable-dtype handling.
        changed: Boolean Series marking any column change and explicitly marking the first row.

        Returns
        -------
        pandas.Series[bool]: index-aligned flags marking starts of consecutive runs.
    """
    # Return an empty Boolean boundary Series for an empty input.
    if df.empty:
        # Return pandas.Series[bool]: index-aligned flags marking starts of consecutive runs.
        return pd.Series([], dtype=bool, index=df.index)
    # Normalize a NumPy array into Python values for recursive comparison.
    if any((isinstance(v, (list, tuple, dict, set, np.ndarray)) for c in columns for v in df[c])):
        # Set rows: Selected observations represented as ordered tuples.
        rows = list(df[list(columns)].itertuples(index=False, name=None))
        # Set flags: Boolean run-start flags from recursive comparisons; first observation is
        # always True.
        flags = [True] + [not all((_equal(a, b) for a, b in zip(p, r))) for p, r in zip(rows, rows[1:])]
        # Return pandas.Series[bool]: index-aligned flags marking starts of consecutive runs.
        return pd.Series(flags, index=df.index)
    # Set curr: DataFrame restricted to selected comparison columns.
    curr = df[list(columns)]
    # Set prev: curr shifted one row so each observation aligns with its predecessor.
    prev = curr.shift()
    # Set same: Per-cell equality flags, including consecutive null equality and nullable-dtype handling.
    same = (curr.eq(prev) | curr.isna() & prev.isna()).fillna(False)
    # Set changed: Boolean Series marking any column change and explicitly marking the first row.
    changed = ~same.all(axis=1)
    # Force the first observation to begin a run because it has no predecessor.
    changed.iloc[0] = True
    # Return pandas.Series[bool]: index-aligned flags marking starts of consecutive runs.
    return changed

def _unique_list(series):
    """
        Flatten nested values into ordered, unique non-null items; sets unordered.

        Parameters
        ----------
        series: Series whose scalar/nested values are flattened into unique non-null items.

        Local variables and comprehension/loop names
        --------------------------------------------
        result: Accumulated unique non-null items, preserving first encounter order.
        v: Current source value or recursively unpacked scalar/container.
        item: Element unpacked from a nested supported container.
        old: Previously retained item tested to avoid duplicates.
        add: Nested recursive helper that closes over and updates result.

        Returns
        -------
        list: flattened unique non-null items; sets have no defined source order.
    """
    # Set result: Accumulated unique non-null items, preserving first encounter order.
    result = []

    # Function purpose, arguments, local variables and return values are documented below.
    def add(v):
        """
            Recursively add supported values to the enclosing unique list.

            Parameters
            ----------
            v: One scalar/container value being recursively flattened into the enclosing unique-value result.

            Local variables and comprehension/loop names
            --------------------------------------------
            item: Element unpacked from v for recursive processing.
            old: Existing result element checked for equality before appending.

            Returns
            -------
            None: updates enclosing result in place rather than returning a collection.
        """
        # Convert a zero-dimensional array to its scalar value before attempting iteration.
        if isinstance(v, np.ndarray) and v.ndim == 0:
            # Set v: value described in this function docstring
            v = v.item()
        # Normalize a NumPy array into Python values for recursive comparison.
        if isinstance(v, (list, tuple, set, np.ndarray)):
            # Iterate item: Element unpacked from v for recursive processing.
            for item in v:
                # Recursively collect this value into the unique-list accumulator.
                add(item)
        # Treat two missing scalars as equal and a known/missing transition as different.
        elif not _is_null(v) and (not any((_equal(v, old) for old in result))):
            # Keep this non-null value only after checking that it is not already retained.
            result.append(v)

    # Iterate v: Current source value or recursively unpacked scalar/container.
    for v in series:
        # Recursively collect this value into the unique-list accumulator.
        add(v)
    # Return list: flattened unique non-null items; sets have no defined source order.
    return result


def _weighted_rate_summary(window, column, freq_seconds, timefactor, duration_seconds=None):
    """
        Integrate observed rates, then divide by the FULL window duration.

        Missing samples make no contribution to the observed integrated amount,
        but their intervals remain in the denominator. This deliberately matches
        the required full-window effective-rate calculation. All-null or empty data
        returns NaN rather than indicating a measured zero release.

        Parameters
        ----------
        window: CR samples belonging to one whole DR window or one CR subrun.
        column: Name of the numeric rate-like CR variable to integrate and average.
        freq_seconds: Sampling interval length in seconds; each observation represents this duration.
        timefactor: Positive time-basis conversion for rate integration: 1/3600 for rates per hour, 1 for rates per second.
        duration_seconds: Optional full-window duration in seconds; None uses number of rows times freq_seconds.

        Local variables and comprehension/loop names
        --------------------------------------------
        values: Requested rate column coerced to numeric; unparseable values become NaN.
        duration: Full window length in seconds, including missing-rate observation intervals.
        total: Observed integrated amount in rate amount units; missing contributions are skipped.
        avg: Observed total divided by timefactor and full duration; NaN if duration is nonpositive.

        Returns
        -------
        tuple[float, float]: observed integrated total and full-window effective rate; (NaN, NaN) for no valid observations.
    """
    # Set values: Requested rate column coerced to numeric; unparseable values become NaN.
    values = pd.to_numeric(window[column], errors='coerce')
    # Return unknown totals/rates when no numeric rate observations are available.
    if not values.notna().any():
        # Return tuple[float, float]: observed integrated total and full-window effective rate;
        # (NaN, NaN) for no valid observations.
        return (np.nan, np.nan)
    # Set duration: Full window length in seconds, including missing-rate observation intervals.
    duration = len(window) * freq_seconds if duration_seconds is None else duration_seconds
    # Set total: Observed integrated amount in rate amount units; missing contributions are skipped.
    total = float((values * freq_seconds * timefactor).sum(min_count=1))
    # Set avg: Observed total divided by timefactor and full duration; NaN if duration is non-positive.
    avg = total / (timefactor * duration) if duration > 0 else np.nan
    # Return tuple[float, float]: observed integrated total and full-window effective rate; (NaN, NaN) for no valid observations.
    return (total, float(avg))

def summarize_dr_runs(drdf, dr_emissionrate, dr_detection=None, dr_classification=None, freq='1min', timefactor=1.0):
    """
        Identify consecutive DR runs used by summarizeData.

        Every original column defines boundaries; bounds/detection changes are
        therefore retained. Detection is optional even when its requested column
        is absent. Rate and classification columns are required. Missing detection
        is unknown, not zero. Existing sample totals are excluded and recomputed.
        A timestamp t represents (t-freq, t]; n samples span n*freq seconds.

        Parameters
        ----------
        drdf: DR observations with an interval-ending datetime index.
        dr_emissionrate: Required DR central-rate column name; original values define runs and numeric values calculate amounts.
        dr_detection: Optional detection column name. None or an absent named column means unavailable; rate-based splitting still applies.
        dr_classification: Required DR classification column name; retained unchanged by DR run extraction.
        freq: Sampling interval accepted by pandas.Timedelta, e.g. "1min"; must be finite and positive.
        timefactor: Positive time-basis conversion for rate integration: 1/3600 for rates per hour, 1 for rates per second.

        Local variables and comprehension/loop names
        --------------------------------------------
        work: Validated chronological DR copy.
        td: Sampling interval Timedelta used to find run start boundaries.
        seconds: Sample duration in seconds.
        col: Required DR source name checked for existence.
        metadata: Ordered generated DR output column names.
        conflicts: Source columns that would collide with reserved generated metadata.
        columns: Original DR comparison columns excluding preexisting sample totals.
        c: Source column name in filtering or representative-row comprehensions.
        ids: Cumulative run-start flags giving consecutive parent DR IDs.
        records: List of completed whole-DR-run dictionaries.
        run_id: ID of the current consecutive DR run.
        run: Underlying DR sample rows sharing run_id.
        row: Representative original source values plus generated DR metadata for one run.
        rate: Representative numeric central DR rate; constant within the DR run.
        detection: Representative numeric detection value or NaN when unavailable.
        duration: Whole DR run length, len(run) times sampling seconds.

        Returns
        -------
        pandas.DataFrame: one row per consecutive DR run, retaining source values and
        generated metadata.

        Important conventions
        ---------------------
        Timestamps represent interval endings; durations are seconds. Rate totals
        use rate * duration_seconds * timefactor. For kg/hour use timefactor=1/3600.
        Missing DR rate/detection values are not converted to zero. Every original
        DR column participates in boundaries, even when dr_detection is None.
        No caller DataFrame is modified; nested source objects are read, not mutated.
    """
    # Set work: Validated chronological DR copy.; td: Sampling interval Timedelta used to find
    # run start boundaries.; seconds: Sample duration in seconds.
    work, td, seconds = _prepare(drdf, freq, timefactor)
    # Iterate col: Required DR source name checked for existence.
    for col in [dr_emissionrate, dr_classification]:
        # Check required input columns; optional classification is filtered separately.
        if col not in work:
            # Stop with an explicit error so invalid data/configuration cannot silently affect
            # results.
            raise KeyError(f'Missing required DR column: {col}')
    # Set metadata: Ordered generated DR output column names.
    metadata = ['DR_RunID', 'dr_StartTime', 'dr_EndTime', 'Duration', 'dr_NSamples', 'dr_total_emission',
                'dr_RequiresCRSplit']
    # Set conflicts: Source columns that would collide with reserved generated metadata.
    conflicts = set(work) & set(metadata) - {'dr_total_emission'}
    # Reject ambiguous source/output column names rather than overwrite values.
    if conflicts:
        # Stop with an explicit error so invalid data/configuration cannot silently affect
        # results.
        raise ValueError(f'Reserved DR metadata names: {sorted(conflicts)}')
    # Set columns: Original DR comparison columns excluding preexisting sample totals.
    columns = [c for c in work if c != 'dr_total_emission']
    # Set ids: Cumulative run-start flags giving consecutive parent DR IDs.
    ids = _rows_changed(work, columns).cumsum()
    # Set records: List of completed whole-DR-run dictionaries.
    records = []
    # Iterate run_id, run: ID of the current consecutive DR run.; Underlying DR sample rows sharing run_id.
    for run_id, run in work.groupby(ids, sort=True):
        # Set row: Representative original source values plus generated DR metadata for one run.
        row = {c: run.iloc[0][c] for c in columns}
        # Set rate: Representative numeric central DR rate; constant within the DR run.
        rate = pd.to_numeric(pd.Series([row[dr_emissionrate]]), errors='coerce').iloc[0]
        # Set detection: Representative numeric detection value or NaN when unavailable.
        detection = pd.to_numeric(pd.Series([row.get(dr_detection, np.nan)]), errors='coerce').iloc[0]
        # Set duration: Whole DR run length, len(run) times sampling seconds.
        duration = len(run) * seconds
        # Attach whole-run times, sample count, duration, emission amount and CR split eligibility.
        row.update(DR_RunID=int(run_id), dr_StartTime=run.index[0] - td, dr_EndTime=run.index[-1],
                   Duration=duration, dr_NSamples=len(run), dr_total_emission=rate * duration * timefactor,
                   dr_RequiresCRSplit=bool(pd.notna(rate) and rate <= 0 or (pd.notna(detection) and detection <= 0)))
        # Save this completed summary record for final DataFrame construction.
        records.append(row)
    # Return pandas.DataFrame: one row per consecutive DR run, retaining source values and generated metadata.
    return pd.DataFrame(records, columns=list(dict.fromkeys(metadata + columns)))

def _summarize_cr_window(window, dr_run_id, cr_subrun_id, start, end, freq_seconds, timefactor,
                         listOfVariables2Sum_CR, listOfVariables2Average_CR, listOfVariables2ListUnique_CR,
                         listOfVariables2Max_CR, listOfVariables2Min_CR, cr_emissionrate=None):
    """
        Aggregate one CR subrun; each rate has its own total and coverage count.

        tc_total_emission is derived from the explicitly named central rate. With
        no explicit rate, a caller-provided sample-total column can be used instead.
        Arithmetic-mean variables retain their ordinary available-value means;
        only rate-like variables use the full-window denominator.

        Parameters
        ----------
        window: CR samples belonging to one whole DR window or one CR subrun.
        dr_run_id: Integer ID of the parent DR run; used to relate a CR record back to its DR record.
        cr_subrun_id: Integer CR subrun ID, starting at 1 inside each parent DR run.
        start: Exclusive interval start boundary; one sample interval precedes the first included sample timestamp.
        end: Inclusive interval end boundary; normally the final included sample timestamp.
        freq_seconds: Sampling interval length in seconds; each observation represents this duration.
        timefactor: Positive time-basis conversion for rate integration: 1/3600 for rates per hour, 1 for rates per second.
        listOfVariables2Sum_CR: CR rate-like column names. Each receives an integrated X_total and a full-window effective rate X; not ordinary sums.
        listOfVariables2Average_CR: CR columns coerced to numeric and averaged over non-null values; these use ordinary available-value means.
        listOfVariables2ListUnique_CR: CR columns flattened into ordered unique non-null lists. Missing tc_Classification is optional.
        listOfVariables2Max_CR: CR columns reduced with pandas max without numeric coercion.
        listOfVariables2Min_CR: CR columns reduced with pandas min without numeric coercion.
        cr_emissionrate: Central CR-rate column name. Explicitly name it for multiple rate aggregations; a single requested rate can be inferred.

        Local variables and comprehension/loop names
        --------------------------------------------
        duration: Full selected interval length in seconds, end minus start.
        result: Output dictionary for one CR window/subrun.
        col: Current requested CR variable being aggregated.
        total: Integrated amount for the current rate-like variable.
        avg: Full-duration effective rate for the current rate-like variable.
        values: Unique flattened non-null elements for a list aggregation.

        Returns
        -------
        dict: metadata, requested aggregate values, per-rate totals/counts and tc_total_emission.
    """
    # Set duration: Full selected interval length in seconds, end minus start.
    duration = (end - start).total_seconds()
    # Set result: Output dictionary for one CR window/subrun.
    result = dict(DR_RunID=int(dr_run_id), CR_SubrunID=int(cr_subrun_id), cr_StartTime=start,
                  cr_EndTime=end, Duration=duration, cr_NSamples=len(window))
    # Iterate col: Current requested CR variable being aggregated.
    for col in listOfVariables2Sum_CR:
        # Set total: Integrated amount for the current rate-like variable.; avg: Full-duration
        # effective rate for the current rate-like variable.
        total, avg = _weighted_rate_summary(window, col, freq_seconds, timefactor, duration)
        # Store the calculated value in (result[f'{col}_total'], result[col]).
        result[f'{col}_total'], result[col] = (total, avg)
        # Store the calculated value in result[f'cr_NValid_{col}'].
        result[f'cr_NValid_{col}'] = int(pd.to_numeric(window[col], errors='coerce').notna().sum())
    # Prefer the explicitly named central rate instead of depending on aggregation-list order.
    if cr_emissionrate is not None:
        # Write the emission amount for this sample/window; this is not an additional amount to double-count.
        result['tc_total_emission'] = result[f'{cr_emissionrate}_total']
    # Use caller-provided sample amounts only when no explicit central rate identifies the total.
    elif 'tc_total_emission' in window:
        # Write the emission amount for this sample/window; this is not an additional amount to double-count.
        result['tc_total_emission'] = pd.to_numeric(window.tc_total_emission,
                                                    errors='coerce').sum(min_count=1)
    else:
        # Write the emission amount for this sample/window; this is not an additional amount to double-count.
        result['tc_total_emission'] = np.nan
    # Iterate col: Current requested CR variable being aggregated.
    for col in listOfVariables2Average_CR:
        # Store the calculated value in result[col].
        result[col] = pd.to_numeric(window[col], errors='coerce').mean()
    # Iterate col: Current requested CR variable being aggregated.
    for col in listOfVariables2ListUnique_CR:
        # Set values: Unique flattened non-null elements for a list aggregation.
        values = _unique_list(window[col])
        # Store the calculated value in result[col].
        result[col] = None if col == 'tc_Classification' and (not values) else values
    # Iterate col: Current requested CR variable being aggregated.
    for col in listOfVariables2Max_CR:
        # Store the calculated value in result[col].
        result[col] = window[col].max()
    # Iterate col: Current requested CR variable being aggregated.
    for col in listOfVariables2Min_CR:
        # Store the calculated value in result[col].
        result[col] = window[col].min()
    # Return dict: metadata, requested aggregate values, per-rate totals/counts and
    # tc_total_emission.
    return result

def summarize_cr_runs(dDF, crdf, freq, timefactor, listOfVariables2Sum_CR=None,
                      listOfVariables2Average_CR=None, listOfVariables2ListUnique_CR=None, listOfVariables2Max_CR=None,
                      listOfVariables2Min_CR=None, split_zero_dr_by_all_cr_columns=True, *, cr_emissionrate=None):
    """
        summarize matched CR windows, retaining eligible CR subruns.

        cr_emissionrate explicitly identifies the central rate used for
        tc_total_emission and is automatically included in rate aggregations.
        For backward compatibility it can be omitted when exactly one rate is
        requested, or when a trusted tc_total_emission sample column exists. With
        multiple requested rates and no sample totals, specify it to avoid using
        list order to choose a physically unrelated total.

        False for split_zero_dr_by_all_cr_columns splits on selected aggregation
        columns rather than disabling splitting. Missing optional classification
        produces no classification column, and never a KeyError. Requested absent
        CR variables are errors, except the optional tc_Classification column.

        Parameters
        ----------
        dDF: DR run summary with DR_RunID, dr_StartTime, dr_EndTime and dr_RequiresCRSplit.
        crdf: CR observations with interval-ending datetime index and source rate/aggregation columns.
        freq: Sampling interval accepted by pandas.Timedelta, e.g. "1min"; must be finite and positive.
        timefactor: Positive time-basis conversion for rate integration: 1/3600 for rates per hour, 1 for rates per second.
        listOfVariables2Sum_CR: CR rate-like column names. Each receives an integrated X_total and a full-window effective rate X; not ordinary sums.
        listOfVariables2Average_CR: CR columns coerced to numeric and averaged over non-null values; these use ordinary available-value means.
        listOfVariables2ListUnique_CR: CR columns flattened into ordered unique non-null lists. Missing tc_Classification is optional.
        listOfVariables2Max_CR: CR columns reduced with pandas max without numeric coercion.
        listOfVariables2Min_CR: CR columns reduced with pandas min without numeric coercion.
        split_zero_dr_by_all_cr_columns: True compares every original CR column in eligible windows; False compares selected aggregation columns. Neither changes split eligibility.
        cr_emissionrate: Central CR-rate column name. Explicitly name it for multiple rate aggregations; a single requested rate can be inferred.

        Local variables and comprehension/loop names
        --------------------------------------------
        cr: Validated sorted CR observations.
        td: Sampling interval used to translate between boundaries and sample stamps.
        seconds: Sampling duration in seconds.
        required: DR summary metadata columns required to locate windows and determine splitting.
        groups: Five ordered aggregation families: rates, means, unique lists, maxima, minima.
        x: One original aggregation-list argument being normalized.
        g: One normalized aggregation family in a comprehension.
        c: CR variable name in filtering, deduplication or schema construction.
        requested: Flattened selected CR variable names across all families.
        absent: Requested variable names missing from CR after allowing absent tc_Classification.
        reserved: Generated CR metadata names that cannot be requested as source variables.
        generated: Rate total/count output names reserved to avoid collisions.
        columns: Original CR boundary columns excluding precomputed sample totals.
        records: Completed CR summary dictionaries across all DR parent windows.
        _: Unused DR summary row index.
        row: Current parent DR summary row.
        start: Parent DR interval start boundary.
        end: Parent DR interval end boundary.
        window: CR observations inside the parent DR interval.
        split: Whether the parent is eligible for CR splitting from its DR rate/detection values.
        boundary_cols: CR columns used to detect consecutive subrun boundaries.
        ids: Consecutive subrun IDs, or a constant 1 for a whole-window summary.
        pieces: List of (subrun ID, sample DataFrame) pairs; one empty placeholder if no observations.
        sub_id: Current CR subrun ID within its parent.
        piece: Current CR subrun sample DataFrame.
        sub_start: Current subrun start boundary or parent boundary for an unsplit/empty window.
        sub_end: Current subrun end boundary or parent boundary for an unsplit/empty window.
        result: Completed aggregate record for the current subrun.
        output_cols: Stable deduplicated output column order, also used for empty outputs.
        name: Generated total/count column name in schema construction.
        variable_list: One aggregation-family sequence checked against bare-string input.

        Returns
        -------
        pandas.DataFrame: CR windows/subruns linked by DR_RunID and CR_SubrunID.
    """
    # Set cr: Validated sorted CR observations.; td: Sampling interval used to translate between
    # boundaries and sample stamps.; seconds: Sampling duration in seconds.
    cr, td, seconds = _prepare(crdf, freq, timefactor)
    # Set required: DR summary metadata columns required to locate windows and determine splitting.
    required = {'DR_RunID', 'dr_StartTime', 'dr_EndTime', 'dr_RequiresCRSplit'}
    # Check that the DR summary supplies the metadata required to select CR windows.
    if not required.issubset(dDF):
        # Stop with an explicit error so invalid data/configuration cannot silently affect results.
        raise KeyError(f'Missing DR metadata: {sorted(required - set(dDF))}')
    # Treat names as a sequence of columns, never as individual characters.
    for variable_list in (listOfVariables2Sum_CR, listOfVariables2Average_CR, listOfVariables2ListUnique_CR,
                          listOfVariables2Max_CR, listOfVariables2Min_CR):
        if isinstance(variable_list, str):
            raise TypeError("Aggregation variables must be sequences of column names.")
    # Set groups: Five ordered aggregation families: rates, means, unique lists, maxima, minima.
    groups = [list(dict.fromkeys([] if x is None else x)) for x in [listOfVariables2Sum_CR, listOfVariables2Average_CR,
                                                     listOfVariables2ListUnique_CR, listOfVariables2Max_CR, listOfVariables2Min_CR]]
    # Set groups: Five ordered aggregation families: rates, means, unique lists, maxima, minima.
    groups = [[c for c in g if c != 'tc_Classification' or c in cr] for g in groups]
    # Check required input columns; optional classification is filtered separately.
    if cr_emissionrate is None and len(groups[0]) == 1 and ('tc_total_emission' not in cr):
        # Set cr_emissionrate: value described in this function docstring
        cr_emissionrate = groups[0][0]
    # Check required input columns; optional classification is filtered separately.
    if cr_emissionrate is None and len(groups[0]) > 1 and ('tc_total_emission' not in cr):
        # Stop with an explicit error so invalid data/configuration cannot silently affect
        # results.
        raise ValueError('Specify cr_emissionrate when integrating multiple CR rates.')
    # Prefer the explicitly named central rate instead of depending on aggregation-list order.
    if cr_emissionrate is not None:
        # Include the explicit central rate in the rate aggregation family, preserving unique names.
        groups[0] = list(dict.fromkeys([cr_emissionrate] + groups[0]))
    # Set requested: Flattened selected CR variable names across all families.
    requested = [c for g in groups for c in g]
    # Set absent: Requested variable names missing from CR after allowing absent tc_Classification.
    absent = set(requested) - set(cr)
    # Check required input columns; optional classification is filtered separately.
    if absent:
        # Stop with an explicit error so invalid data/configuration cannot silently affect results.
        raise KeyError(f'Missing CR variables: {sorted(absent)}')
    # Reject conflicting aggregation families for the same CR variable.
    if len(requested) != len(set(requested)):
        # Stop with an explicit error so invalid data/configuration cannot silently affect results.
        raise ValueError('A CR variable must belong to only one aggregation family.')
    # Set reserved: Generated CR metadata names that cannot be requested as source variables.
    reserved = {'DR_RunID', 'CR_SubrunID', 'cr_StartTime', 'cr_EndTime', 'Duration', 'cr_NSamples',
                'cr_HasData', 'cr_WasSplit', 'tc_total_emission'}
    # Set generated: Rate total/count output names reserved to avoid collisions.
    generated = {f'{c}_total' for c in groups[0]} | {f'cr_NValid_{c}' for c in groups[0]}
    # Reject ambiguous source/output column names rather than overwrite values.
    if set(requested) & (reserved | generated):
        # Stop with an explicit error so invalid data/configuration cannot silently affect results.
        raise ValueError('CR variable conflicts with output metadata.')
    # Set columns: Original CR boundary columns excluding precomputed sample totals.
    columns = [c for c in cr if c not in ('tc_total_emission', 'cr_total_emission')]
    # Set records: Completed CR summary dictionaries across all DR parent windows.
    records = []
    # Iterate _, row: Current parent DR summary row.
    for _, row in dDF.iterrows():
        # Set start: Parent DR interval start boundary.; end: Parent DR interval end boundary.
        start, end = (pd.Timestamp(row.dr_StartTime), pd.Timestamp(row.dr_EndTime))
        # Set window: CR observations inside the parent DR interval.
        window = cr.loc[start + td:end]
        # Set split: Whether the parent is eligible for CR splitting from its DR rate/detection values.
        split = bool(row.dr_RequiresCRSplit)
        # Set boundary_cols: CR columns used to detect consecutive subrun boundaries.
        boundary_cols = columns if split_zero_dr_by_all_cr_columns else requested
        # Set ids: Consecutive subrun IDs, or a constant 1 for a whole-window summary.
        ids = _rows_changed(window, boundary_cols).cumsum() if split else pd.Series(1, index=window.index, dtype='int64')
        # Set pieces: List of (subrun ID, sample DataFrame) pairs; one empty placeholder if no observations.
        pieces = list(window.groupby(ids, sort=True)) if not window.empty else [(1, window)]
        # Iterate sub_id, piece: Current CR subrun ID within its parent.; Current CR subrun sample DataFrame.
        for sub_id, piece in pieces:
            # Set sub_start: Current subrun start boundary or parent boundary for an unsplit/empty window.
            sub_start = piece.index[0] - td if split and (not piece.empty) else start
            # Set sub_end: Current subrun end boundary or parent boundary for an unsplit/empty window.
            sub_end = piece.index[-1] if split and (not piece.empty) else end
            # Set result: Completed aggregate record for the current subrun.
            result = _summarize_cr_window(piece, row.DR_RunID, sub_id, sub_start, sub_end, seconds,
                                          timefactor, *groups, cr_emissionrate=cr_emissionrate)
            # Report observation presence separately from whether multiple CR subruns actually occurred.
            result.update(cr_HasData=not piece.empty, cr_WasSplit=len(pieces) > 1)
            # Save this completed summary record for final DataFrame construction.
            records.append(result)
    # Set output_cols: Stable deduplicated output column order, also used for empty outputs.
    output_cols = list(dict.fromkeys(['DR_RunID', 'CR_SubrunID', 'cr_StartTime', 'cr_EndTime', 'Duration',
                                      'cr_NSamples', 'tc_total_emission', 'cr_HasData',
                                      'cr_WasSplit'] + requested + [name for c in groups[0] for name in [f'{c}_total',
                                                                                                         f'cr_NValid_{c}']]))
    # Return pandas.DataFrame: CR windows/subruns linked by DR_RunID and CR_SubrunID.
    return pd.DataFrame(records, columns=output_cols)

def summarizeData(drdf, crdf, timefactor, dr_emissionrate, dr_detection=None, dr_classification=None,
                  freq='1min', listOfVariables2Sum_CR=None, listOfVariables2Average_CR=None,
                  listOfVariables2ListUnique_CR=None, listOfVariables2Max_CR=None, listOfVariables2Min_CR=None, *,
                  cr_emissionrate=None, split_zero_dr_by_all_cr_columns=True):
    """
        Return CRdf, DRdf, cDF, dDF. Old positional argument order is retained.

        Detection is optional: omit it or pass None. A named but absent detection
        column is treated as unavailable, so CR splitting depends on rate alone.
        Explicitly supply cr_emissionrate when summing more than one rate variable.
        If omitted, one requested rate is inferred consistently for every output.
        With no inferable rate, an existing tc_total_emission sample column can
        supply trusted CR amounts; otherwise a clear ValueError is raised.
        Returned sample DataFrames are sorted copies, not caller-owned inputs.
        For each rate-like X: X_total integrates valid amounts; X divides that
        total by full subrun duration; cr_NValid_X exposes the observed coverage.
        For example, [10, NaN, 30] over three equal intervals gives X=13.3333.
        This is observed amount/full duration, not an imputed unknown rate.

        Parameters
        ----------
        drdf: DR observations with an interval-ending datetime index.
        crdf: CR observations with interval-ending datetime index and source rate/aggregation columns.
        timefactor: Positive time-basis conversion for rate integration: 1/3600 for rates per hour, 1 for rates per second.
        dr_emissionrate: Required DR central-rate column name; original values define runs and numeric values calculate amounts.
        dr_detection: Optional detection column name. None or an absent named column means unavailable; rate-based splitting still applies.
        dr_classification: Required DR classification column name; retained unchanged by DR run extraction.
        freq: Sampling interval accepted by pandas.Timedelta, e.g. "1min"; must be finite and positive.
        listOfVariables2Sum_CR: CR rate-like column names. Each receives an integrated X_total and a full-window effective rate X; not ordinary sums.
        listOfVariables2Average_CR: CR columns coerced to numeric and averaged over non-null values; these use ordinary available-value means.
        listOfVariables2ListUnique_CR: CR columns flattened into ordered unique non-null lists. Missing tc_Classification is optional.
        listOfVariables2Max_CR: CR columns reduced with pandas max without numeric coercion.
        listOfVariables2Min_CR: CR columns reduced with pandas min without numeric coercion.
        cr_emissionrate: Central CR-rate column name. Explicitly name it for multiple rate aggregations; a single requested rate can be inferred.
        split_zero_dr_by_all_cr_columns: True compares every original CR column in eligible windows; False compares selected aggregation columns. Neither changes split eligibility.

        Local variables and comprehension/loop names
        --------------------------------------------
        dr: Validated chronological DR observations.
        cr: Validated chronological CR observations.
        _: Unused sampling Timedelta/seconds from _prepare.
        d: Shared whole-DR-run output, returned as dDF.
        c: Matching CR window/subrun output, returned as cDF.
        rate_names / name: Unique rate aggregation candidates / candidate column name used to resolve the central rate consistently across all outputs.

        Returns
        -------
        tuple: (cr, dr, c, d): CR sample copy, DR sample copy, CR subruns, DR runs.

        Important conventions
        ---------------------
        Timestamps represent interval endings; durations are seconds. Rate totals
        use rate * duration_seconds * timefactor. For kg/hour use timefactor=1/3600.
        Missing DR rate/detection values are not converted to zero. Every original
        DR column participates in boundaries, even when dr_detection is None.
        No caller DataFrame is modified; nested source objects are read, not mutated.

    """
    # Set dr: Validated chronological DR observations.
    dr, _, _ = _prepare(drdf, freq, timefactor)
    # Set cr: Validated chronological CR observations.
    cr, _, _ = _prepare(crdf, freq, timefactor)
    # Require matching DR/CR grids before constructing paired windows.
    if not dr.index.equals(cr.index):
        # Stop with an explicit error so invalid data/configuration cannot silently affect results.
        raise ValueError('DR and CR grids must match.')
    # Resolve the central rate in the public function so sample totals and
    # subrun totals use the same column. Previously the subrun helper inferred
    # it locally, but the final sample calculation still indexed crdf[None].
    if isinstance(listOfVariables2Sum_CR, str):
        raise TypeError("listOfVariables2Sum_CR must be a sequence of column names, not a string.")
    if cr_emissionrate is None:
        rate_names = list(dict.fromkeys([] if listOfVariables2Sum_CR is None else listOfVariables2Sum_CR))
        rate_names = [name for name in rate_names if name != 'tc_Classification' or name in cr]
        if len(rate_names) == 1:
            cr_emissionrate = rate_names[0]
        elif 'tc_total_emission' not in cr:
            raise ValueError("Specify cr_emissionrate to calculate central CR sample totals")

    if cr_emissionrate is not None and cr_emissionrate not in cr:
        raise KeyError(f"Missing central CR rate column: {cr_emissionrate}")
    # Set d: Shared whole-DR-run output, returned as dDF.
    d = summarize_dr_runs(dr, dr_emissionrate, dr_detection, dr_classification, freq, timefactor)
    # Set c: Matching CR window/subrun output, returned as cDF.
    c = summarize_cr_runs(d, cr, freq, timefactor, listOfVariables2Sum_CR, listOfVariables2Average_CR,
                          listOfVariables2ListUnique_CR, listOfVariables2Max_CR, listOfVariables2Min_CR,
                          split_zero_dr_by_all_cr_columns, cr_emissionrate=cr_emissionrate)

    # Calculate sample totals on the validated, sorted COPIES. Returning the
    # original inputs previously changed caller data and discarded normalization.
    _, seconds = _validate_frequency(freq)
    cr['tc_total_emission'] = (pd.to_numeric(cr[cr_emissionrate], errors='coerce') * seconds * timefactor)
    dr['dr_total_emission'] = (pd.to_numeric(dr[dr_emissionrate], errors='coerce') * seconds * timefactor)
    return cr, dr, c, d

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

def mergeRunsV1(classified, cr_emissionrate, tc_localization, factor, freq):
    """
        Aggregate consecutive rows having the same Classification,
        controlled release rate, and localization into unique runs.

        Parameters
        classified (pandas.DataFrame): DataFrame of paired controlled release and detection data.
        cr_emissionrate (str): Column name containing the controlled release emission rate.
        tc_localization (str): Column name containing the localization/equipment group.
        factor (float) : Conversion factor from seconds to the time basis of the emission-rate data.
        freq (str): Frequency of analysis, e.g. "1min", "30s", "5min".
        Returns (pandas.DataFrame): DataFrame containing one row per consecutive run.
    """

    # Convert frequency to seconds
    freq_duration = pd.Timedelta(freq).total_seconds()
    # Columns that define a consecutive run
    group_columns = ["Classification", cr_emissionrate, tc_localization]
    # Create a copy so the original dataframe is not modified
    crdf = classified.copy()
    crdf[tc_localization] = crdf[tc_localization].fillna(0)
    # Create a new group whenever any grouping column changes
    crdf["shift"] = (crdf[group_columns].ne(crdf[group_columns].shift()).any(axis=1).cumsum())
    # Calculate total emitted mass/quantity for each time step
    crdf["Total"] = (crdf[cr_emissionrate] * freq_duration * factor)

    # Aggregate consecutive runs
    runs = (crdf.groupby("shift").agg(Total=("Total", "sum"), Count=("Total", "size"),
                                      Classification=("Classification", "first"),
                                      ControlledReleaseRate=(cr_emissionrate, "first"),
                                      Localization=(tc_localization, "first")).reset_index(drop=True))

    # Calculate average emission rate for each run
    runs["EmissionRate"] = (runs["Total"] / (runs["Count"] * freq_duration * factor))
    # Remove intermediate columns
    runs.drop(columns=["Count", "Total", "ControlledReleaseRate"], inplace=True)
    # Rename output columns back to original names
    runs.rename(columns={"EmissionRate": cr_emissionrate, "Localization": tc_localization}, inplace=True)

    return runs

def mergeRuns(classified, cr_emissionrate, tc_localization, factor, freq):
    """
        Aggregate consecutive rows having the same controlled release rate
        and localization into unique runs.

        Classification rules for each run:
            1. If at least one row is "TP", classify the run as "TP".
            2. Otherwise, if at least one row is "FP", classify the run as "FP".
            3. Otherwise, retain the Classification of the run.
        Capture fraction:
            Percentage of rows within the run classified as either "TP" or "TN":
                Capture fraction = n / nRows * 100
            where:
                n = number of rows classified as TP or TN
                nRows = total number of rows in the run

        Parameters
        ----------
        classified (pandas.DataFrame) : DataFrame containing paired controlled-release and detection data.
        cr_emissionrate (str) : Column name containing the controlled release emission rate.
        tc_localization (str) : Column name containing the localization/equipment group.
        factor (float) : Conversion factor from seconds to the time basis of the emission-rate data.
        freq (str) : Frequency of analysis, e.g. "1min", "30s", "5min".

        Returns (pandas.DataFrame) : One row per consecutive controlled-release/localization run, including
                                    Classification and Capture fraction.
    """

    # Convert frequency to seconds
    freq_duration = pd.Timedelta(freq).total_seconds()

    # Work on a copy to avoid modifying the original dataframe
    crdf = classified.copy()
    crdf[tc_localization] = crdf[tc_localization].fillna(0)
    crdf[cr_emissionrate] = crdf[cr_emissionrate].round(5)

    # ------------------------------------------------------------------------------------------------------------------
    # 1. Define consecutive runs
    # ------------------------------------------------------------------------------------------------------------------

    # Classification is deliberately excluded.
    # A change in either emission rate or localization starts a new run.
    group_columns = [cr_emissionrate, tc_localization]
    crdf["shift"] = (crdf[group_columns].ne(crdf[group_columns].shift()).any(axis=1).cumsum())

    # ------------------------------------------------------------------------------------------------------------------
    # 2. Calculate total emission for each time step
    # ------------------------------------------------------------------------------------------------------------------
    crdf["Total"] = (crdf[cr_emissionrate] * freq_duration * factor)

    # ------------------------------------------------------------------------------------------------------------------
    # 3. Determine Classification for each run
    # ------------------------------------------------------------------------------------------------------------------
    def classify_run(series):
        """
            Assign a single Classification to an aggregated run.
            Priority: TP > FP > existing classification
        """

        values = series.dropna()
        if (values == "TP").any():
            return "TP"
        if (values == "FP").any():
            return "FP"
        if len(values) > 0:
            return values.iloc[0]
        return None

    # ------------------------------------------------------------------------------------------------------------------
    # 4. Calculate capture fraction
    # ------------------------------------------------------------------------------------------------------------------
    def capture_fraction(series):
        # Total number of rows in the run
        nRows = len(series)

        # Number of TP and TN rows
        nTP = (series == "TP").sum()
        nTN = (series == "TN").sum()

        # Capture fraction (%)
        return ((nTP + nTN) / nRows) * 100

    # ------------------------------------------------------------------------------------------------------------------
    # 5. Aggregate consecutive runs
    # ------------------------------------------------------------------------------------------------------------------
    runs = (crdf.groupby("shift", sort=False).agg(Total=("Total", "sum"),
                                                  Count=("Total", "size"),
                                                  Classification=("Classification", classify_run),
                                                  Capture_fraction=("Classification", capture_fraction),
                                                  ControlledReleaseRate=(cr_emissionrate, "first"),
                                                  Localization=(tc_localization, "first")).reset_index(drop=True))

    # ------------------------------------------------------------------------------------------------------------------
    # 6. Calculate emission rate for each run
    # ------------------------------------------------------------------------------------------------------------------
    runs["EmissionRate"] = (runs["Total"] / (runs["Count"] * freq_duration * factor))

    # ------------------------------------------------------------------------------------------------------------------
    # 7. Remove intermediate columns
    # ------------------------------------------------------------------------------------------------------------------
    runs.drop(columns=["Count", "Total", "ControlledReleaseRate"], inplace=True)

    # ------------------------------------------------------------------------------------------------------------------
    # 8. Restore original column names
    # ------------------------------------------------------------------------------------------------------------------
    runs.rename(columns={"EmissionRate": cr_emissionrate, "Localization": tc_localization,
                         "Capture_fraction": "Capture fraction"}, inplace=True)

    return runs

def validate_spatial_inputs(controlledReleaseDF, detectionReportDF, tc_PolygonCoord, tc_localization,
                            dr_PolygonCoord, dr_localization):
    """
        Prevent inconsistent spatial filtering between releases and reports.
        Spatial-filtering information is considered available when a nonempty dataset contains either:
          1. Its configured polygon column; or
          2. Its configured localization column.

        Polygon columns take precedence over localization columns, consistent with extractSpatialSubsetData().
        Partially populated spatial columns are considered valid. This function only checks whether the relevant
        spatial column exists; it does not validate individual values within that column.

        Behavior:
          - Both datasets have spatial-filtering information:
                Continue analysis.

          - Only one dataset has spatial-filtering information:
                Raise ValueError.

          - Neither dataset has spatial-filtering information:
                Warn that spatial filtering will not occur and continue analysis.

          - Empty datasets are ignored.
    """

    def has_spatial_input(df, polygon_col, location_col):
        """
            Return whether a nonempty dataset provides a spatial input column.
        """

        if df.empty:
            return None

        # Polygon takes precedence when the configured column exists.
        if polygon_col is not None and polygon_col in df.columns:
            return True
        # Otherwise, use the configured localization column.
        if location_col is not None and location_col in df.columns:
            return True

        return False

    tc_has_spatial = has_spatial_input(controlledReleaseDF, tc_PolygonCoord, tc_localization)
    dr_has_spatial = has_spatial_input(detectionReportDF, dr_PolygonCoord, dr_localization)
    # Both datasets are empty.
    if tc_has_spatial is None and dr_has_spatial is None:
        return
    # One dataset is empty. There are no observations in the empty dataset to spatially filter, so allow analysis to continue.
    if tc_has_spatial is None or dr_has_spatial is None:
        return

    # Both datasets provide spatial-filtering information.
    if tc_has_spatial and dr_has_spatial:
        return

    # Neither dataset provides spatial-filtering information.
    if not tc_has_spatial and not dr_has_spatial:
        warnings.warn("Neither controlledReleaseDF nor detectionReportDF provides "
                      "information required for spatial filtering. Neither dataset "
                      "will be spatially filtered, and the analysis will continue.", UserWarning)
        return

    # Exactly one dataset provides spatial-filtering information.
    if tc_has_spatial:
        available = "controlledReleaseDF"
        unavailable = "detectionReportDF"
    else:
        available = "detectionReportDF"
        unavailable = "controlledReleaseDF"

    raise ValueError("Spatial-input mismatch: "
                     f"{available} provides information for spatial filtering, but "
                     f"{unavailable} does not. Equivalent spatial filtering cannot be assured.")

def DataPairing(analyStarttime, analyEndtime, controlledReleaseDF, detectionReportDF, tc_baselineEmissionRate,
                dr_baselineEmissionRate, cr_releaseRate, dr_releaseRate, cr_emissionrateBounds, dr_emissionrateBounds,
                MaintenanceRecordDF, mr_DateTime, cr_DateTime, dr_DateTime, OfflineRecordDF, OfflineDateTime, OnlineDateTime,
                pob_DateTime, passiveObservationDF, dr_ID_column=None, tc_ID_column=None, freq="T", bufferTime=(0, 0),
                overLappingFrac=0, ThresholdDuration=0, ThresholdRateAboveBaseline=0, detection=None,
                Method="Rolling Average", listOfVariables2Sum_CR=None, listOfVariables2Average_CR=None,
                listOfVariables2ListUnique_CR=None, listOfVariables2Max_CR=None, listOfVariables2Min_CR=None,
                listOfVariables2Sum_DR=None, listOfVariables2Average_DR=None, listOfVariables2ListUnique_DR=None,
                listOfVariables2Max_DR=None, listOfVariables2Min_DR=None, listOfVariables2ObtainFirst_CR=None,
                listOfVariables2ObtainLast_CR=None, factor=1/3600, listOfVariables2ObtainFirst_DR=None,
                listOfVariables2ObtainLast_DR=None, ASE=None, tc_PolygonCoord=None, tc_BoundingZ=None, tc_localization=None,
                dr_PolygonCoord=None, dr_BoundingZ=None, dr_localization=None, ASE_PolygonCoord_Dict=None,
                ASE_textual_mapper=None, tc_bufferDistance=(0,0), dr_bufferDistance=(0, 0)):
    """
        DataPairing integrates and processes time-series data from two sources (controlled release and detection report).
        It performs spatial filtering, temporal alignment, exclusion of invalid periods, aggregation, thresholding,
        and classification of paired data.

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

        7. Apply Thresholding: Applying the detection criteria provided equally to controlled release and detection report data.
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

        8. Compare and Classify: Compares the thresholding values of controlled releases and detection reports to produce TP, FN, FP, and TN classification
            *Assumptions*
            a. Time resolved controlled release detection report data must have been thresholded (i.e, with values of 1s and 0s)
            *Code Structure*
            a. Thresholding value is 1 and 1 for controlled release and detection reports, respectively ==> TP
            b. Thresholding value is 1 and 0 for controlled release and detection reports, respectively ==> FN
            a. Thresholding value is 0 and 1 for controlled release and detection reports, respectively ==> FP
            a. Thresholding value is 0 and 0 for controlled release and detection reports, respectively ==> TN
            a. Thresholding value is Nan for controlled release ==> flagged and excluded

        9. Final Output Preparation
    """

    # Work with local lists: adding exclusion columns must not mutate a reused
    # configuration or fail when an optional aggregation list is None.
    listOfVariables2ListUnique_CR = list(listOfVariables2ListUnique_CR or [])
    listOfVariables2ListUnique_DR = list(listOfVariables2ListUnique_DR or [])

    ### Step 0: It checks both datasets before spatial filtering to ensure one will not be filtered while the other passes through unfiltered.
    print("Check both datasets before spatial filtering analysis...")
    validate_spatial_inputs(controlledReleaseDF=controlledReleaseDF, detectionReportDF=detectionReportDF,
                            tc_PolygonCoord=tc_PolygonCoord, tc_localization=tc_localization,
                            dr_PolygonCoord=dr_PolygonCoord, dr_localization=dr_localization)

    ### Step 1: Spatial Filtering ###
    try:
        start_time = time.time()
        # Subset controlled release data (test center) based on spatial extent.
        Data4thisASE_tc = extractSpatialSubsetData(DataDF=controlledReleaseDF,
                                                   ASE=ASE,
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
                                                   ASE=ASE,
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

        tc_TimeSeriesDF = flagData(dataDF=tc_TimeSeriesDF,
                                   exclusionDF=passiveObservationDF,
                                   freq=freq,
                                   flag="Passive Observation: Data Excluded",
                                   classification="tc_Classification",
                                   start_exc=pob_DateTime[0],
                                   end_exc=pob_DateTime[1],
                                   bufferTime=bufferTime)

        dr_TimeSeriesDF = flagData(dataDF=dr_TimeSeriesDF,
                                   exclusionDF=passiveObservationDF,
                                   freq=freq,
                                   flag="Passive Observation: Data Excluded",
                                   classification="dr_Classification",
                                   start_exc=pob_DateTime[0],
                                   end_exc=pob_DateTime[1],
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

    ### Step 5: Aggregation by Frequency ###
    try:
        start_time = time.time()
        # Aggregate controlled release data.
        print(f"Starting controlled release aggregation")
        CRDF = aggregateTimeSeriesDataByFreq(DataDF=tc_TimeSeriesDF,
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
        DRDF = aggregateTimeSeriesDataByFreq(DataDF=dr_TimeSeriesDF,
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
        CRDF, DRDF = align_time_indexes(crdf=CRDF, drdf=DRDF,
                                        dr_columns=[dr_releaseRate, dr_emissionrateBounds[0], dr_emissionrateBounds[1]],
                                        detection=detection,
                                        analysis_starttime=analyStarttime,
                                        analysis_endtime=analyEndtime,
                                        freq=freq)
        end_time = time.time()
        print(f"Aligning time index code executed in {end_time - start_time:.6f} seconds")
    except Exception as e:
        print(f"Could not align time indexes due to the exception: {e}")
        raise

    ### Step 7: Group detection report and controlled release data into windows.
    try:
        start_time = time.time()
        CRDF, DRDF, cDF, dDF = summarizeData(drdf=DRDF, crdf=CRDF, timefactor=factor, dr_emissionrate=dr_releaseRate,
                                             dr_detection=detection, dr_classification="dr_Classification", freq=freq,
                                             listOfVariables2Sum_CR=listOfVariables2Sum_CR,
                                             listOfVariables2Average_CR=listOfVariables2Average_CR,
                                             listOfVariables2ListUnique_CR=listOfVariables2ListUnique_CR,
                                             listOfVariables2Max_CR=listOfVariables2Max_CR,
                                             listOfVariables2Min_CR=listOfVariables2Min_CR)
        end_time = time.time()
        print(f"Summarizing controlled release and detection data completed in {end_time - start_time:.6f} seconds")
    except Exception as e:
        print(f"Could not summarize controlled release and detection data due to the exception: {e}")
        CRDF = pd.DataFrame()
        DRDF = pd.DataFrame()
        cDF = pd.DataFrame()
        dDF = pd.DataFrame()
        raise

    # ### Step 8: Apply Thresholding - timeseries data ###
    try:
        start_time = time.time()
        DRDF_ts, CRDF_ts = thresholdingApplication(CRDF=CRDF,
                                                   DRDF=DRDF,
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

    # ### Step 8: Apply Thresholding - summarized data ###
    try:
        start_time = time.time()
        DRDF_sd, CRDF_sd = thresholdingApplication(CRDF=cDF,
                                                   DRDF=dDF,
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

    ### Step 8: Compare and Classify - timeseries data ###
    try:
        start_time = time.time()
        ClassifiedDF_ts, CRDF_ts, DRDF_ts = compareDataFrames(CRDF=CRDF_ts, DRDF=DRDF_ts)
        end_time = time.time()
        print(f"Pairing controlled release with detection report code executed in {end_time - start_time:.6f} seconds")
    except Exception as e:
        print(f"Could not pair controlled release with detection report due to the exception: {e}")
        raise
    try:
        start_time = time.time()
        ClassifiedDF_s, CRDF_s, DRDF_s = compareSummarizedData(CRDF_sd, DRDF_sd, tc_Detected="tc_Detection", dr_Detected="dr_Detection",
                                                               tc_Classification="tc_Classification", dr_Classification="dr_Classification")
        end_time = time.time()
        print(f"Pairing controlled release with detection report (summarized) code executed in {end_time - start_time:.6f} seconds")
    except Exception as e:
        print(f"Could not pair controlled release with detection report (summarized) due to the exception: {e}")
        raise

    ### Step 9: Final Output Preparation ###
    try: # Combine index into a single column and reset index.
        ClassifiedDF_ts = ClassifiedDF_ts.reset_index(names="Time Series")
    except Exception as e:
        print(f"ClassifiedDataFrame does not have its index in form of datetime due to {e}")
        raise

    ### Step 10: Re-initialize selected variables
    listOfVariables2ListUnique_CR.remove("tc_Classification")
    listOfVariables2ListUnique_DR.remove("dr_Classification")

    return DRDF_ts, CRDF_ts, ClassifiedDF_s
