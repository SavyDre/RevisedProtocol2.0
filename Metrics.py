import numpy as np
import pandas as pd
from Classification import merge_intervals

def factorsFromrelativeErrors(RE):
    """
        Converts a relative error (RE) into a multiplicative factor.
        Parameters:
            - RE (float): The relative error value.
        Returns:
            - float: The corresponding multiplicative factor.
    """
    if pd.isna(RE):
        return np.nan
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

def evaluateRelativeErrors(quantDF, minTrueRate, cr_emission, dr_emission):
    """
        Filter paired rates and calculate relative errors and factors.

        Remove rows where each rate is either zero or missing, including
        mixed zero/missing pairs. For remaining rows, calculate results
        only when both rates are finite and nonzero, regardless of classification.
        The input DataFrame is not modified.

        Parameters
        ----------
        quantDF : pandas.DataFrame
            Paired data containing the columns named by cr_emission and
            dr_emission. Classification is not required or used.

        minTrueRate : float or None
            Optional CR-rate threshold in the same units as cr_emission.
            Retain only rows with CR rates strictly above this threshold.
            None or NaN disables this additional filter.

        cr_emission : str
            Controlled-release reference-rate column name.
            Numeric values provide the relative-error denominator.

        dr_emission : str
            Detection-report rate column name, using the same units as CR.

        Returns
        -------
        pandas.DataFrame
            A copied, filtered DataFrame with:
            - QuantificationAccuracyRel: (DR_rate - CR_rate) / CR_rate.
            - QuantificationFactors: output of factorsFromrelativeErrors().

            Retained rows with a missing, infinite, or zero rate receive
            None in both output columns. Previous results are replaced.

        Notes
        -----
        Unparseable rates are treated as missing.
        Finite negative rates are eligible for calculations.
        Requires factorsFromrelativeErrors() to be defined or imported.
    """
    quantDF = quantDF.copy()

    # Convert rates to numeric; unparseable values become missing.
    cr = pd.to_numeric(quantDF[cr_emission], errors="coerce")
    dr = pd.to_numeric(quantDF[dr_emission], errors="coerce")

    # Remove (0, 0), (missing, missing), (0, missing), and (missing, 0).
    remove = ((cr.eq(0) | cr.isna()) & (dr.eq(0) | dr.isna())).fillna(False)
    quantDF = quantDF.loc[~remove].copy()

    # Recalculate CR values to align with the retained rows.
    cr = pd.to_numeric(quantDF[cr_emission], errors="coerce")

    # Apply the optional minimum CR-rate filter.
    if not pd.isna(minTrueRate):
        quantDF = quantDF.loc[cr > minTrueRate].copy()

    # Align both numeric rate Series with the final retained rows.
    cr = pd.to_numeric(quantDF[cr_emission], errors="coerce")
    dr = pd.to_numeric(quantDF[dr_emission], errors="coerce")

    # Calculate results only when both rates are finite and nonzero.
    valid = (np.isfinite(cr) & np.isfinite(dr) & cr.ne(0) & dr.ne(0)).fillna(False)

    # Object dtype preserves literal None for ineligible retained rows.
    errors = pd.Series(None, index=quantDF.index, dtype=object)
    factors = pd.Series(None, index=quantDF.index, dtype=object)

    # Compute relative errors for eligible rows.
    errors.loc[valid] = ((dr.loc[valid] - cr.loc[valid]) / cr.loc[valid])

    # Compute factors using your existing helper.
    factors.loc[valid] = errors.loc[valid].apply(factorsFromrelativeErrors)

    # Replace both result columns, clearing any stale values.
    quantDF["QuantificationAccuracyRel"] = errors
    quantDF["QuantificationFactors"] = factors

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
        Return (TTA, TTD) arrays, one minimum per matched controlled-release run.

        Detection timestamps are sample-end timestamps, matching the classification
        grid. An alert requires a valid receipt time; detection time does not. UTC
        conversion preserves instants. Process each CR run independently to avoid a
        potentially huge CR-by-DR matrix. The historical RecievedTime spelling is
        retained for compatibility.
    """
    if timebasis <= 0:
        raise ValueError('timebasis must be positive.')
    cr = CRDF.copy()
    dr = DRDF.copy()
    cr.index = pd.to_datetime(cr.index, utc=True)
    dr.index = pd.to_datetime(dr.index, utc=True)

    # Exclusions must break detection runs; otherwise maintenance/offline data
    # can contribute alerts even though classification deliberately excludes it.
    for frame, flag, detection in ((cr, 'tc_Classification', tc_Detection), (dr, 'dr_Classification', dr_Detection)):
        if flag in frame:
            excluded = frame[flag].astype(str).str.contains('Data Excluded', regex=False)
            frame[detection] = frame[detection].astype(float).mask(excluded)
    has_receipt = RecievedTime is not None and RecievedTime in dr
    if has_receipt:
        dr[RecievedTime] = pd.to_datetime(dr[RecievedTime], utc=True, errors='coerce')
    cr_runs = identify_consecutive_runs(cr, tc_Detection)
    dr_runs = identify_consecutive_runs(dr, dr_Detection, RecievedTime if has_receipt else None)
    cr_runs = cr_runs.loc[cr_runs['Classification'] > 0]
    dr_runs = dr_runs.loc[dr_runs['Classification'] > 0]
    TTA, TTD = [], []
    for run in cr_runs.itertuples(index=False):
        matches = dr_runs.loc[dr_runs['startTime'].between(run.startTime, run.endTime)]
        if matches.empty:
            continue
        TTD.append((matches['startTime'].min() - run.startTime).total_seconds() / timebasis)
        if has_receipt:
            receipt = matches['RecievedTime'].min()
            if pd.notna(receipt):
                TTA.append((receipt - run.startTime).total_seconds() / timebasis)
    return np.asarray(TTA), np.asarray(TTD)

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
        Return excluded seconds within [start, end], merging overlaps once.

        Partially overlapping exclusions contribute their clipped duration instead of being discarded. The input dataframe is not modified.
    """
    if end <= start:
        raise ValueError('Exclusion window end must be later than start.')
    if exclusionDF.empty:
        return 0.0
    intervals = merge_intervals(exclusionDF, col_start, col_end)
    intervals = intervals.loc[(intervals[col_start] < end) & (intervals[col_end] > start)].copy()
    return (intervals[col_end].clip(upper=end) - intervals[col_start].clip(lower=start)).dt.total_seconds().sum()

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

    # Classification flags can be scalar strings or aggregated lists. Matching
    # their textual content handles both representations consistently.
    CRDF, DRDF = CRDF.copy(), DRDF.copy()
    CRDF['tc_Classification'] = CRDF['tc_Classification'].astype(str)
    DRDF['dr_Classification'] = DRDF['dr_Classification'].astype(str)
    # Calculate the operational fractions for the solution
    try:
        Total_TC_time = CRDF["time_diffs"].sum()
        # Filter valid center data when not under maintenance/offline  - data during test center online time
        Online_TestCenderData = CRDF.loc[~CRDF["tc_Classification"].str.contains("Maintenance Record: Data Excluded", regex=False)]
        Online_TC_time = Online_TestCenderData["time_diffs"].sum()  # Total operational time in seconds
        TC_nonoperationtime = Total_TC_time - Online_TC_time

        # Evaluating solution online datetime
        filt = (DRDF["dr_Classification"].str.contains("Offline Record: Data Excluded", regex=False)) | (DRDF["OriginalReport"] == "False")
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
        Return (receipt - report end) in the requested units, preserving UTC offsets.
        Missing receipt times are omitted. Negative delays are retained because a
        report may arrive before its declared observation period has ended.
    """
    if timeBasisFactor <= 0:
        raise ValueError('timeBasisFactor must be positive.')
    if dr_endTime not in detectionReportDF or dr_ReceivedTime not in detectionReportDF:
        return np.array([])
    received = pd.to_datetime(detectionReportDF[dr_ReceivedTime], utc=True, errors='coerce')
    end_column = f'{dr_endTime}_Original' if f'{dr_endTime}_Original' in detectionReportDF else dr_endTime
    ended = pd.to_datetime(detectionReportDF[end_column], utc=True, errors='coerce')
    return ((received - ended).dt.total_seconds() / timeBasisFactor).dropna().to_numpy()

def surveytime(DF, startTime, endTime, timeBasisFactor):
    """
        Return survey durations without changing the caller's dataframe.
    """
    if timeBasisFactor <= 0:
        raise ValueError('timeBasisFactor must be positive.')
    if DF.empty:
        return np.array([])
    starts = pd.to_datetime(DF[startTime], utc=True, errors='coerce')
    ends = pd.to_datetime(DF[endTime], utc=True, errors='coerce')
    return ((ends - starts).dt.total_seconds() / timeBasisFactor).dropna().to_numpy()