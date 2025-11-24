import warnings
import pandas as pd
import numpy as np
from Classification import merge_intervals
from datetime import datetime, timezone

def factorsFromrelativeErrors(RE):
    """
    Converts a relative error (RE) into a multiplicative factor.
    Parameters:
        - RE (float): The relative error value.
    Returns:
        - float: The corresponding multiplicative factor.
    """
    try:
        if RE > 0:
            # Positive relative error adds directly to 1
            f = RE + 1
        elif RE < 0:
            # Negative relative error converts using the reciprocal rule
            f = -1 / (RE + 1)
        else:
            # Zero relative error means perfect accuracy (factor of 1)
            f = 1
        return f
    except ZeroDivisionError:
        print(f"ZeroDivisionError for RE: {RE}")
        return None
    except Exception as e:
        print(f"Unexpected error due to: {e}")
        return None


def totalEstimate_per_timestep(df, emissionrate, factor=1/3600, totalEmission="total_emission"):
    """
    Calculates the total emission per time step based on the specified frequency.

    Parameters:
        - df (pd.DataFrame): Input DataFrame with a datetime index.
        - emissionrate (str): Column name for the emission rate values.
        - factor (float): Conversion factor to transform rate in seconds time-basis, default is 1/3600.
        - totalEmission (str): Column name for the calculated total emissions.

    Returns:
        - pd.DataFrame: DataFrame with added columns for time differences and total emissions.
    """

    # Ensure the DataFrame is sorted by its index
    df = df.sort_index()

    # Calculate time differences based on specified frequency units in terms seconds
    df['time_diffs'] = df.index.to_series().diff().dt.total_seconds().fillna(0)

    # Modify 'time_diffs' where the emission rate is > 0 and 'Cumulative Duration' is 0
    if 'Cumulative Duration' in df.columns:
        mask = (df[emissionrate] > 0) & (df['Cumulative Duration'] == float(0))
        df.loc[mask, 'time_diffs'] = 0

    # Calculate total emission for each time step
    df[totalEmission] = df.apply(lambda row: row[emissionrate] * row['time_diffs'] * factor, axis=1)

    return df

def AnalysisCrossCheck(Original_CRDF, Original_DRDF, Analyzed_CRDF, Analyzed_DRDF, QuantData,
                       cr_releaseRate, dr_releaseRate, cr_DateTime, dr_DateTime, timefactor):
    """
    Cross-checks total emissions (mass) between original and analyzed datasets.

    Parameters
        - Original_CRDF : Original Controlled Release dataset with start and end datetime columns.
        - Original_DRDF : Original Detection Report dataset with start and end datetime columns.
        - Analyzed_CRDF : Analyzed Controlled Release dataset (index should be a DatetimeIndex).
        - Analyzed_DRDF : Analyzed Detection Report dataset (index should be a DatetimeIndex).
        - QuantData: Paired data for comparing quantification estimates with release rates
        - cr_releaseRate : Column name of release rate in CRDF datasets.
        - dr_releaseRate : Column name of release rate in DRDF datasets.
        - cr_DateTime : List containing start and end datetime column names for CRDF.
        - dr_DateTime : List containing start and end datetime column names for DRDF.
        - timefactor : Conversion factor (e.g., scaling to correct units).

    Returns None
    """

    try:
        # Ensure datetime conversion for start and end columns
        Original_CRDF[cr_DateTime[0]] = pd.to_datetime(Original_CRDF[cr_DateTime[0]])
        Original_CRDF[cr_DateTime[1]] = pd.to_datetime(Original_CRDF[cr_DateTime[1]])
        # Calculate duration (seconds) between start and end times
        Original_CRDF['time_diffs'] = (Original_CRDF[cr_DateTime[1]] - Original_CRDF[cr_DateTime[0]]).dt.total_seconds()
        # Compute total mass (rate × duration × scaling factor)
        Original_CRDF['total_emission'] = (Original_CRDF[cr_releaseRate] * Original_CRDF['time_diffs'] * timefactor)
        print(f"Original Controlled Release: Total Mass is {Original_CRDF['total_emission'].sum()}")
    except Exception as e:
        print(f"Original Controlled Release: Could not evaluate total mass due to {e}")

    try:
        # Ensure datetime conversion
        Original_DRDF[dr_DateTime[0]] = pd.to_datetime(Original_DRDF[dr_DateTime[0]])
        Original_DRDF[dr_DateTime[1]] = pd.to_datetime(Original_DRDF[dr_DateTime[1]])
        # Calculate duration (seconds) between start and end times
        Original_DRDF['time_diffs'] = (Original_DRDF[dr_DateTime[1]] - Original_DRDF[dr_DateTime[0]]).dt.total_seconds()
        # Compute total mass (rate × duration × scaling factor)
        Original_DRDF['total_emission'] = (Original_DRDF[dr_releaseRate] * Original_DRDF['time_diffs'] * timefactor)
        print(f"Original Detection Report: Total Mass is {Original_DRDF['total_emission'].sum()}")
    except Exception as e:
        print(f"Original Detection Report: Could not evaluate total mass due to {e}")

    try:
        print(f"Analyzed Controlled Release from time step data: Total Mass is {Analyzed_CRDF['tc_total_emission'].sum()}")
    except Exception as e:
        print(f"Analyzed Controlled Release from time step data: Could not evaluate total mass due to {e}")

    try:
        print(f"Analyzed Detection Report from time step data: Total Mass is {Analyzed_DRDF['dr_total_emission'].sum()}")
    except Exception as e:
        print(f"Analyzed Detection Report from time step data: Could not evaluate total mass due to {e}")

    try:
        print(f"Analyzed Controlled Release: Total Mass is {QuantData['tc_total_emission'].sum()}")
    except Exception as e:
        print(f"Analyzed Controlled Release: Could not evaluate total mass due to {e}")

    try:
        print(f"Analyzed Detection Report: Total Mass is {QuantData['dr_total_emission'].sum()}")
    except Exception as e:
        print(f"Analyzed Detection Report: Could not evaluate total mass due to {e}")

    return

def evaluateRelativeErrors(quantDF, cr_emission, dr_emission):
    """
    Evaluates relative errors between classified emissions and detection report emissions.
    Also computes multiplicative factors for these relative errors.

    Parameters:
        - quantDF (pd.DataFrame): The DataFrame of paired quantification data.
        - cr_emission (str): Column name for "classified report" emissions.
        - dr_emission (str): Column name for "detection report" emissions.

    Returns:
        - pd.DataFrame: The updated DataFrame with relative errors and multiplicative factors.
    """

    def relative_error(row):
        """
        Computes the relative error for a single row.

        Parameters:
            - row (pd.Series): A single row of the DataFrame.
        Returns:
            - float or None: The relative error, or None if values are invalid.
        """
        # Explicitly convert to scalar values
        try:
            dr_emissions = float(row[dr_emission])
            cr_emissions = float(row[cr_emission])
            classification = str(row['Classification'])
        except (ValueError, TypeError):
            return None

        # Return None if either value is null, NaN, missing, or not TP
        #if pd.isna(dr_emissions) or pd.isna(cr_emissions) or classification in ["FP", "FN", "TN"]:
        if pd.isna(dr_emissions) or pd.isna(cr_emissions) or classification in ["Maintenance Record: Data Excluded",
                                                                                "Offline Record: Data Excluded", "0"]:
            return None

        try:
            if cr_emissions == 0:
                return None
            return (dr_emissions - cr_emissions) / cr_emissions
        except Exception as e:
            print(f"Unexpected error for row {row.name}: {e}")
            return None

    # Compute relative errors for each row
    quantDF["QuantificationAccuracyRel"] = quantDF.apply(relative_error, axis=1)

    # Compute multiplicative factors for "1" classification
    mask_tp = quantDF["Classification"] == "1"
    quantDF.loc[mask_tp, "QuantificationFactors"] = quantDF.loc[mask_tp, "QuantificationAccuracyRel"].apply(
        factorsFromrelativeErrors)

    return quantDF

def identify_consecutive_runs(df, classification, RecievedTime=None):
    """
    Identifies consecutive runs of values in the specified column of a DataFrame
    and returns a new DataFrame with startTime, endTime, Classification, DetectTime, and ID.

    Parameters:
        - df (pd.DataFrame): Input DataFrame with datetime index and a column of binary or categorical values.
        - classification (1 or 0): The name of the column containing the values to identify runs (1s and 0s).
        - ReceivedTime: The datetime (UTC) at which an observation report was received by the Test Center

    Returns:
    - pd.DataFrame: A DataFrame with columns:
        - "startTime": The start time of the run.
        - "endTime": The end time of the run.
        - "Classification": The value of the run (from the column_name).
        - "ReceivedTime": The minimum value of the ReceivedTime
        - "ID": A unique incremental integer for each run.
    """
    # Ensure the DataFrame is sorted by datetime index
    df = df.sort_index()

    # Ensure the index is a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    # Add a group identifier for consecutive runs
    df['shift'] = df[classification].ne(df[classification].shift()).cumsum()

    # Reset the index to access it as a column for aggregation
    df.index.name = 'Time' # Ensures the new column created is called "Time"
    df = df.reset_index()

    # Aggregate by the 'shift' column
    if RecievedTime is not None:
        result_df = df.groupby('shift').agg(startTime=('Time', 'first'),  # Start time of the run
                                            endTime=('Time', 'last'),  # End time of the run
                                            Classification=(classification, 'first'),  # Value of the run
                                            RecievedTime=(RecievedTime, 'min') # Minimum/earliest received time
                                            ).reset_index(drop=True)
    else:
        result_df = df.groupby('shift').agg(startTime=('Time', 'first'),  # Start time of the run
                                            endTime=('Time', 'last'),  # End time of the run
                                            Classification=(classification, 'first')  # Value of the run
                                            ).reset_index(drop=True)

    # Assign unique IDs starting from 1
    result_df['ID'] = result_df.index + 1

    return result_df


def timeDifference(DRDF, CRDF, RecievedTime, tc_Detection="tc_Detection", dr_Detection="dr_Detection", timebasis=60):
    """
        Optimized calculation of time to detection (TTD) and time to alert (TTA) between CRDF and DRDF detections.

        Parameters:
            - DRDF (pd.DataFrame): DataFrame for DR detections with datetime index.
            - CRDF (pd.DataFrame): DataFrame for CR detections with datetime index.
            - RecievedTime (str): Column name for the ReceivedTime
            - tc_Detection (str): Column name for the result of thresholding result to CRDF.
            - dr_Detection (str): Column name for the result of thresholding result to DRDF.
            - timebasis (int): Time basis for normalization (default: 60 seconds).

        Returns:
            - np.ndarray: Array of time-to-detection values (TTD).
            - np.ndarray: Array of time-to-alert values (TTA).
        """
    # Create a deep copy of the DataFrame
    CRDF_copy = CRDF.copy(deep=True)
    DRDF_copy = DRDF.copy(deep=True)

    # Filter out the Null column in DetectTime, RecievedTime.
    if RecievedTime is not None:
        DRDF_copy = DRDF_copy.loc[DRDF_copy[RecievedTime] != ""]
        # Ensure all are in datetime type.
        DRDF_copy[RecievedTime] = pd.to_datetime(DRDF_copy[RecievedTime], errors='coerce')
        DRDF_copy[RecievedTime] = DRDF_copy[RecievedTime].apply(lambda dt: dt.replace(tzinfo=timezone.utc))


    # Identify consecutive runs of detections
    crDF = identify_consecutive_runs(df=CRDF_copy, classification=tc_Detection)
    drDF = identify_consecutive_runs(df=DRDF_copy, classification=dr_Detection, RecievedTime=RecievedTime)

    # Filter out rows where detection is 1
    CRDF_detect = crDF.loc[crDF["Classification"] > 0].reset_index(drop=True)
    DRDF_detect = drDF.loc[drDF["Classification"] > 0].reset_index(drop=True)

    # Extract start and end times for CRDF detections
    cr_start_times = CRDF_detect["startTime"].values[:, None]  # Shape (n, 1)
    cr_end_times = CRDF_detect["endTime"].values[:, None]  # Shape (n, 1)

    # Extract start times for DRDF detections
    dr_start_times = DRDF_detect["startTime"].values  # Shape (m,)

    # Extract the received time of the detection report associated with the detection for DRDF detections
    try:
        ReceivedTime = DRDF_detect["RecievedTime"].values  # Shape (m,)
    except Exception as e:
        print(f"Could not run the evaluation due to {e}")

    # Create a mask for overlapping time ranges for each time in dr_start_times
    overlap_mask = (dr_start_times >= cr_start_times) & (dr_start_times <= cr_end_times)

    # Evaluate TTA - Time to Alert
    try:
        time_differences_TTA = np.where(overlap_mask, (ReceivedTime - cr_start_times).astype("timedelta64[s]").astype(float)/timebasis, np.nan)
    except Exception as e:
        print(f"Could not run the evaluation due to {e}")

    # Evaluate TTD - Time to Detect
    time_differences_TTD = np.where(overlap_mask, (dr_start_times - cr_start_times).astype("timedelta64[s]").astype(float)/timebasis, np.nan)

    try:
        # Check if all columns are NaN before evaluating TTA
        if np.all(np.isnan(time_differences_TTA), axis=1).all():
            # Return an empty array if all columns are NaN
            TTA = np.array([])
        else:
            # Otherwise, calculate the minimum ignoring NaNs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                TTA = np.nanmin(time_differences_TTA, axis=1)
                TTA = TTA[~np.isnan(TTA)]
    except Exception as e:
        print(f"Could not run the evaluation due to {e}")
        TTA = np.array([])

    # Check if all columns are NaN before evaluating TTD
    if np.all(np.isnan(time_differences_TTD), axis=1).all():
        # Return an empty array if all columns are NaN
        TTD = np.array([])
    else:
        # Otherwise, calculate the minimum ignoring NaNs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            TTD = np.nanmin(time_differences_TTD, axis=1)
            TTD = TTD[~np.isnan(TTD)]

    # Add TTA and TTD columns to the detection-level DRDF_detect DataFrame
    # This section was added to display the columns of TTA and TTD in the classifiedData file
    DRDF_detect = DRDF_detect.copy()
    DRDF_detect["TTA"] = pd.Series(TTA, index=DRDF_detect.index[:len(TTA)])
    DRDF_detect["TTD"] = pd.Series(TTD, index=DRDF_detect.index[:len(TTD)])

    #return DRDF_detect
    return TTA, TTD

def concatenate_with_common_index(df1, df2):
    """
    Concatenates two DataFrames using their common indexes.
    The resulting DataFrame adopts the index of one of the input DataFrames.

    Parameters:
        - df1 (DataFrame): First DataFrame.
        - df2 (DataFrame): Second DataFrame.

    Returns:
        - pd.DataFrame: A new DataFrame concatenated using the common index.
    """
    # Determine the common indexes
    common_index = df1.index.intersection(df2.index)

    # Filter both DataFrames to only include rows with the common index
    df1_filtered = df1.loc[common_index]
    df2_filtered = df2.loc[common_index]

    # Concatenate along the columns and use the index of the specified DataFrame
    result = pd.concat([df1_filtered, df2_filtered], axis=1)

    return result


def calculate_excludedTimes(exclusionDF, col_start, col_end, start, end):
    """
    Calculates the total excluded time within a given time range by merging overlapping intervals.

    Parameters:
        - exclusionDF (DataFrame): DataFrame containing exclusion periods with start and end times.
        - col_start (str): Column name in `exclusionDF` representing the start of each interval.
        - col_end (str): Column name in `exclusionDF` representing the end of each interval.
        - start (datetime): Start time for the calculation period.
        - end (datetime): End time for the calculation period.

    Returns:
        - float: Total excluded time in seconds.
    """
    if not exclusionDF.empty:
        # Step 1: Merge overlapping intervals in the exclusion DataFrame
        exclusionDF = merge_intervals(exclusionDF=exclusionDF, start=col_start, end=col_end)

        # Step 2: Filter intervals that fall within the specified range (start to end)
        filt = (exclusionDF[col_start] >= start) & (exclusionDF[col_end] <= end)
        recordDF = exclusionDF.loc[filt]

        # Step 3: Calculate the duration of each interval in seconds
        recordDF.loc[:, ["Time Difference"]] = recordDF.apply(lambda row: (row[col_end] - row[col_start]).total_seconds(), axis=1)

        # Step 4: Calculate the total excluded time
        total_time = recordDF["Time Difference"].sum()  # Use `sum` instead of `max` for cumulative duration
    else:
        # If the exclusionDF is empty, then there is no excluded time
        total_time = 0

    return total_time


def OperationalFactorsAnalysis(CRDF, DRDF):
    """
    Analyzes operational factors for a test center and its solution based on provided records.

    Parameters:
        - CRDF (DataFrame): DataFrame containing emission data for the test center.
        - CRDF (DataFrame): DataFrame containing reported data.

    Returns:
        - test center offline fraction
        - solution offline fraction
    """

    # Calculate the operational fractions for the solution
    try:
        Total_TC_time = CRDF["time_diffs"].sum()
        # Filter valid center data when not under maintenance/offline  - data during test center online time
        Online_TestCenderData = CRDF.loc[CRDF["tc_Classification"] != "Maintenance Record: Data Excluded"]
        Online_TC_time = Online_TestCenderData["time_diffs"].sum()  # Total operational time in seconds
        TC_nonoperationtime = Total_TC_time - Online_TC_time

        # Evaluating solution online datetime
        filt = (DRDF["dr_Classification"] == "Offline Record: Data Excluded") | (DRDF["OriginalReport"] == "False")
        Online_SolData = DRDF.loc[~filt]
        Online_solTime = Online_SolData["time_diffs"].sum()
        Sol_nonoperationtime = Total_TC_time - Online_solTime

        tc_offline_frac = round(TC_nonoperationtime/Total_TC_time, 3)
        sol_offline_frac = round(Sol_nonoperationtime/Total_TC_time, 3)

    except Exception as e:
        print(f"Could not calculate solution's operational factor due to exception: {e}")
        tc_offline_frac = np.nan
        sol_offline_frac = np.nan

    return tc_offline_frac, sol_offline_frac

def reportingDelay(detectionReportDF, dr_endTime, dr_ReceivedTime, timeBasisFactor):
    """
    Calculates the reporting delay for each record in a DataFrame.

    Parameters:
        - detectionReportDF (DataFrame): DataFrame containing detection reports.
        - dr_endTime (str): Column name for the emission end time in the DataFrame.
        - dr_ReceivedTime (str): Column name for the report received time in the DataFrame.
        - timeBasisFactor (float): Factor to scale the delay time (e.g., 3600 for hours, 60 for minutes).

    Returns:
        - np.ndarray: A NumPy array containing the reporting delay for each record. Returns None for invalid rows.
    """
    # Ensure the required columns exist in the DataFrame
    if dr_endTime not in detectionReportDF.columns or dr_ReceivedTime not in detectionReportDF.columns:
        return np.array([])

    else:
        # Ensure the start and received times are datetime-like for calculations
        detectionReportDF_copy = detectionReportDF.copy(deep=True)
        detectionReportDF_copy = detectionReportDF_copy.loc[detectionReportDF_copy[dr_ReceivedTime] != ""]
        detectionReportDF_copy[dr_ReceivedTime] = pd.to_datetime(detectionReportDF_copy[dr_ReceivedTime], errors='coerce')
        detectionReportDF_copy[dr_endTime] = detectionReportDF_copy[dr_endTime].apply(lambda dt: dt.replace(tzinfo=timezone.utc))
        detectionReportDF_copy[dr_ReceivedTime] = detectionReportDF_copy[dr_ReceivedTime].apply(lambda dt: dt.replace(tzinfo=timezone.utc))

        # Calculate reporting delay with a check for null, NaN, or None values
        def calculate_delay(row):
            end_time = row[dr_endTime]
            received_time = row[dr_ReceivedTime]

            # Return None if either datetime is invalid
            if pd.isna(end_time) or pd.isna(received_time):
                return None
            return (received_time - end_time).total_seconds() / timeBasisFactor
        detectionReportDF_copy.loc[:, ["Reporting Delay"]] = detectionReportDF_copy.apply(lambda row: calculate_delay(row), axis=1)

        # Filter out None values before converting to NumPy array
        valid_delays = detectionReportDF_copy["Reporting Delay"].dropna()

        # Convert valid delays to a NumPy array and return
        reportingDelay = valid_delays.to_numpy()

        return reportingDelay

def surveytime(DF, startTime, endTime, timeBasisFactor):
    """
    Calculate the survey time for a DataFrame based on start and end time columns.

    Parameters:
        - DF (pd.DataFrame): Input DataFrame containing the start and end time columns.
        - startTime (str): Column name for the test center start time.
        - endTime (str): Column name for the performer stop time.
        - timeBasisFactor (float): Factor to convert time difference (in seconds) to desired units.

    Returns:
        - np.ndarray: Array of survey times calculated as (endTime - startTime) / timeBasisFactor.
    """

    if not DF.empty:
        # Convert start and end times to datetime format, coercing errors to NaT
        DF[startTime] = pd.to_datetime(DF[startTime], errors='coerce')
        DF[endTime] = pd.to_datetime(DF[endTime], errors='coerce')

        # Calculate survey time in seconds and divide by timeBasisFactor
        DF["Survey Time"] = (DF[endTime] - DF[startTime]).dt.total_seconds()/timeBasisFactor

        # Drop NaN values and return as a NumPy array
        surveytimes = DF["Survey Time"].dropna().to_numpy()

        # Drop column name
        DF.drop(columns="Survey Time", inplace=True)
        return surveytimes

    else:
        return np.array([])
