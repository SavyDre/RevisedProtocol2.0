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


def identify_consecutive_runs(df, classification, RecievedTime=None, temporal_resolution=None):
    """
        Identify consecutive runs of identical classification values in a time-indexed DataFrame.
        A new run is initiated when:
            1. The classification value changes.
            2. The current or preceding classification is missing.
            3. Consecutive timestamps are separated by an interval
               different from the specified temporal resolution.

        Parameters
        ----------
        df : pd.DataFrame - Input DataFrame with a DatetimeIndex.
        classification : str - Name of the column containing binary or categorical classification values.
        RecievedTime : str or None, optional - Name of the column containing Observation Report receipt timestamps.
                       If specified, the earliest valid receipt timestamp within each run is returned.
        temporal_resolution : str, pd.Timedelta or None, optional - Expected interval between consecutive analysis timestamps,
                            e.g., '30s' or '1min'. If None, runs are identified based on classification changes and
                            missing values without checking temporal gaps.
        Returns
        -------
        pd.DataFrame - DataFrame containing:
            startTime: Timestamp of the first sample in each run.
            endTime: Timestamp of the last sample in each run.
            Classification: Classification value associated with the run.
            RecievedTime: Earliest valid receipt timestamp within the run, if a receipt timestamp column is specified.
            ID: Unique sequential identifier assigned to each run.

        Notes
        -----
        startTime and endTime represent the first and last included sample timestamps, rather than necessarily the
        physical boundaries of the corresponding observation interval. The minimum receipt timestamp aggregated over an
        entire detection run should not be used directly to calculate TTA where the run contains both TP and FP time steps.
    """

    # Validate the input datetime index.
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    if df.index.hasnans:
        raise ValueError("DataFrame index must not contain missing timestamps.")

    if not df.index.is_unique:
        raise ValueError("DataFrame index must contain unique timestamps.")

    if classification not in df.columns:
        raise KeyError(f"Classification column '{classification}' not found.")

    # Sort chronologically and avoid modifying the input DataFrame.
    df = df.sort_index().copy()

    # Convert receipt timestamps to UTC, where applicable.
    # Invalid or missing receipt timestamps are represented by NaT.
    if RecievedTime is not None:
        if RecievedTime not in df.columns:
            raise KeyError(f"Receipt timestamp column '{RecievedTime}' not found.")
        df[RecievedTime] = pd.to_datetime(df[RecievedTime], utc=True, errors="coerce")

    # Identify changes in classification between successive rows.
    new_run = (df[classification].ne(df[classification].shift()).fillna(True))

    # Missing classifications interrupt consecutive runs. This prevents excluded or unavailable data from connecting
    # otherwise separate detection periods.
    new_run = (new_run | df[classification].isna() | df[classification].shift().isna())

    # Identify temporal discontinuities, where an expected analysis temporal resolution has been specified.
    if temporal_resolution is not None:
        resolution = pd.to_timedelta(temporal_resolution)

        if pd.isna(resolution) or resolution <= pd.Timedelta(0):
            raise ValueError("temporal_resolution must be a positive time interval.")
        time_difference = df.index.to_series().diff()

        # A temporal gap initiates a new run, even when the classification remains unchanged.
        new_run = (new_run | time_difference.ne(resolution))

    # Assign a common group identifier to samples belonging# to the same consecutive run.
    df["shift"] = new_run.cumsum()

    # Convert the datetime index into a column for aggregation.
    df.index.name = "Time"
    df = df.reset_index()

    # Define the aggregation operations for each run.
    aggregations = {"startTime": ("Time", "first"), "endTime": ("Time", "last"), "Classification": (classification, "first")}

    # Optionally retain the earliest receipt timestamp in each run.
    if RecievedTime is not None:
        aggregations["RecievedTime"] = ( RecievedTime, "min")

    # Aggregate consecutive samples into individual runs.
    result_df = (df.groupby("shift", sort=False).agg(**aggregations).reset_index(drop=True))

    # Assign unique sequential identifiers starting from one.
    result_df["ID"] = result_df.index + 1

    return result_df


def timeDifference(DRDF, CRDF, RecievedTime, tc_Detection="tc_Detection", dr_Detection="dr_Detection",
                   timebasis=60, temporal_resolution="30s", sample_end_timestamps=True):
    """
        Calculate Time to Alert (TTA) and Time to Detection (TTD) for successfully detected Test Centre detection periods.
        A Test Centre detection period is defined as a continuous interval during which the Test Centre Controlled Release
        data satisfy the specified detection threshold criteria.

        - Time to Detection (TTD): TTD is calculated as the elapsed time between the onset of a Test Centre detection
        period and the first corresponding qualifying solution detection.
        -----------------------
            A TTD of zero is assigned when a qualifying solution detection is already present at the onset of the Test
            Centre detection period.

        - Time to Alert (TTA): TTA is calculated as the elapsed time between the onset of a successfully detected Test
        Centre detection period and the earliest valid receipt timestamp associated with a corresponding TP time step.
        -------------------
            Receipt timestamps preceding the onset of the Test Centre detection period are excluded to prevent negative
            TTA values.

        Parameters
        ----------
        - DRDF (pd.DataFrame) : Time-resolved solution Observation Report data containing detection classifications and,
          where available, receipt timestamps.
        - CRDF (pd.DataFrame) : Time-resolved Test Centre Controlled Release data containing detection classifications.
        - RecievedTime (str or None) : Name of the column containing the timestamps at which Observation Reports were
          received by the Test Centre.
            If None, or if the specified column is unavailable, only TTD is calculated.
        - tc_Detection (str, optional) : Name of the Test Centre binary detection column.
        - dr_Detection (str, optional) : Name of the solution binary detection column.
        - timebasis (float, optional) : Number of seconds per output time unit.
            Examples:
                    1 : seconds
                    60 : minutes
                    3600 : hours
            Default is 60, corresponding to minutes.

        - temporal_resolution (str or pd.Timedelta, optional) : Temporal resolution of the detection classification
          analysis grid, e.g., '30s' or '1min'.
            Default is '30s'.

        - sample_end_timestamps (bool, optional) : Indicates whether the datetime indices represent the ends of the
          corresponding analysis time steps. If True, the beginning of the first qualifying time step is obtained by
          subtracting one analysis time step from its timestamp. If False, timestamps are assumed to represent the
          beginnings of the analysis time steps. Default is True.

        Returns
        -------
        TTA : np.ndarray
            Non-negative Time to Alert values for successfully
            detected Test Centre detection periods with valid
            corresponding receipt timestamps.

        TTD : np.ndarray
            Non-negative Time to Detection values for successfully
            detected Test Centre detection periods.

        Notes
        -----
        1. The function should be applied independently within
           each Analysis Spatial Extent (ASE).

        2. Test Centre data and solution Observation Reports must
           already be temporally aligned and spatially aggregated
           at the specified analysis resolution.

        3. Detection classifications must represent binary
           threshold outcomes, with 1 indicating True and
           0 indicating False. Missing or excluded classifications
           are represented by NaN.

        4. Undetected Test Centre detection periods do not
           contribute numerical TTD or TTA values.

        5. Missing or invalid receipt timestamps affect TTA
           calculations but do not affect TTD calculations.

        6. Receipt timestamps must remain associated with the
           Observation Reports contributing to the corresponding
           TP time steps.

        7. The returned TTA and TTD arrays may have different
           lengths and should not be assumed to contain paired
           measurements.
    """

    # ------------------------------------------------------------------------------------------------------------
    # 1. Validate analysis parameters.
    # ------------------------------------------------------------------------------------------------------------

    if not np.isfinite(timebasis) or timebasis <= 0:
        raise ValueError("timebasis must be finite and positive.")
    resolution = pd.to_timedelta(temporal_resolution)

    if pd.isna(resolution) or resolution <= pd.Timedelta(0):
        raise ValueError("temporal_resolution must be a positive time interval.")

    # ------------------------------------------------------------------------------------------------------------
    # 2. Prepare Test Centre and solution datasets.
    # ------------------------------------------------------------------------------------------------------------
    # Copy input DataFrames to avoid modifying the original data.
    cr = CRDF.copy()
    dr = DRDF.copy()

    # Convert timestamps to UTC and sort chronologically.
    # Timezone-aware timestamps retain their original instants.
    # Timezone-naive timestamps are assumed to already be UTC.
    cr.index = pd.to_datetime(cr.index, utc=True)
    dr.index = pd.to_datetime(dr.index, utc=True)
    cr = cr.sort_index()
    dr = dr.sort_index()

    # Validate timestamps and detection columns.
    for frame, detection, dataset in ((cr, tc_Detection, "Test Centre"), (dr, dr_Detection, "Solution")):
        if frame.index.hasnans:
            raise ValueError(f"{dataset} index contains missing timestamps.")

        if not frame.index.is_unique:
            raise ValueError(f"{dataset} index contains duplicate timestamps.")

        if detection not in frame.columns:
            raise KeyError(f"{dataset} detection column '{detection}' not found.")

    # ----------------------------------------------------------------------------------------------------------
    # 3. Account for excluded data.
    # ----------------------------------------------------------------------------------------------------------
    # Excluded observations must not contribute to detection periods or true-positive classifications.
    for frame, flag, detection in ((cr, "tc_Classification", tc_Detection), (dr, "dr_Classification", dr_Detection)):
        # Convert binary classifications to float to accommodate missing values introduced during exclusion.
        frame[detection] = frame[detection].astype(float)
        if flag in frame.columns:
            excluded = (frame[flag].astype(str).str.contains("Data Excluded", regex=False, na=False))

            # Mask excluded classifications rather than assigning False, because excluded data are not equivalent to
            # valid negative detection outcomes.
            frame[detection] = (frame[detection].mask(excluded))

        # Ensure that retained detection classifications are binary.
        valid_values = frame[detection].dropna()

        if not valid_values.isin([0, 1]).all():
            raise ValueError(f"Detection column '{detection}' must contain only binary values (0 or 1) or missing values.")

    # ------------------------------------------------------------------------------------------------------------------
    # 4. Prepare Observation Report receipt timestamps.
    # ------------------------------------------------------------------------------------------------------------------
    has_receipt = (RecievedTime is not None and RecievedTime in dr.columns)
    if has_receipt:
        # Invalid receipt timestamps are converted to NaT. These observations remain eligible for TTD evaluation.
        dr[RecievedTime] = pd.to_datetime(dr[RecievedTime], utc=True, errors="coerce")

    # -------------------------------------------------------------------------------------------------------------
    # 5. Identify Test Centre detection periods.
    # -------------------------------------------------------------------------------------------------------------
    # Consecutive positive Test Centre classifications define detection periods. Missing classifications and temporal
    # discontinuities interrupt these periods.

    cr_runs = identify_consecutive_runs(cr, tc_Detection, temporal_resolution=resolution)
    # Retain only periods satisfying the specified detection threshold criteria.
    cr_runs = cr_runs.loc[cr_runs["Classification"] == 1]

    # ----------------------------------------------------------------------------------------------------------------
    # 6. Identify true-positive time steps.
    # --------------------------------------------------------------------------------------------------------------
    # Identify timestamps at which the Test Centre Controlled Release data satisfy the detection threshold criteria.
    tc_positive_index = cr.index[cr[tc_Detection] == 1]

    # Identify qualifying solution detections.
    dr_positive = dr.loc[dr[dr_Detection] == 1]

    # A time step is classified as TP when both the Test Centre and solution detection threshold outcomes are True.
    # Restricting the analysis to these timestamps prevents FP detections, missing ground-truth data and excluded
    # observations from contributing to TTD or TTA.
    tp = dr_positive.loc[dr_positive.index.isin(tc_positive_index)]

    # ----------------------------------------------------------------------------------------------------------------
    # 7. Calculate TTD and TTA for each detection period.
    # ----------------------------------------------------------------------------------------------------------------
    TTA = []
    TTD = []

    for run in cr_runs.itertuples(index=False):
        # Locate TP time steps within the current Test Centre detection period.
        # searchsorted avoids repeatedly scanning the entire solution dataset for each detection period.
        left = tp.index.searchsorted(run.startTime, side="left")
        right = tp.index.searchsorted(run.endTime, side="right")
        matched_tp = tp.iloc[left:right]

        # If no corresponding TP time step exists, the Test Centre detection period was not successfully detected.
        # Such periods have no numerical TTD or TTA.
        if matched_tp.empty:
            continue

        # --------------------------------------------------------------------------------------------------------
        # 7a. Determine the reference onset.
        # --------------------------------------------------------------------------------------------------------
        # If timestamps represent sample ends, subtract one analysis time step to obtain the beginning of the
        # first qualifying Test Centre time step.
        if sample_end_timestamps:
            reference_onset = (run.startTime - resolution)
        else:
            reference_onset = run.startTime

        # ---------------------------------------------------------------------------------------------------------
        # 7b. Calculate Time to Detection (TTD).
        # ---------------------------------------------------------------------------------------------------------
        # Identify the first TP time step within the Test Centre detection period.
        first_detection = matched_tp.index[0]

        # Convert the first TP timestamp to the beginning of its analysis time step, using the same timestamp convention
        # applied to the Test Centre detection period.
        if sample_end_timestamps:
            first_detection = (first_detection - resolution)

        # Calculate detection latency in the specified time units. If a qualifying solution detection is already present
        # at the onset of the Test Centre detection period, the first TP time step coincides with the reference onset
        # and TTD is therefore zero.
        ttd = (first_detection - reference_onset).total_seconds() / timebasis
        TTD.append(ttd)

        # ----------------------------------------------------------------------------------------------------------
        # 7c. Calculate Time to Alert (TTA).
        # ----------------------------------------------------------------------------------------------------------
        if has_receipt:
            # Retain valid receipt timestamps associated specifically with corresponding TP time steps.
            valid_receipts = (matched_tp[RecievedTime].dropna())

            # Exclude receipt timestamps preceding the onset of the Test Centre detection period. Such timestamps would
            # produce negative TTA values and are not considered valid alert latencies for the corresponding emission period.
            valid_receipts = valid_receipts.loc[valid_receipts >= reference_onset]

            # If no valid receipt timestamp remains, omit TTA for this period without affecting its TTD.
            if not valid_receipts.empty:
                # Identify the earliest valid receipt timestamp associated with a corresponding TP time step.
                first_receipt = valid_receipts.min()

                # Calculate alert latency in the specified time units.
                tta = (first_receipt - reference_onset).total_seconds() / timebasis

                TTA.append(tta)

    # --------------------------------------------------------------------------------------------------------------
    # 8. Return latency measurements as NumPy arrays.
    # --------------------------------------------------------------------------------------------------------------
    return (np.asarray(TTA, dtype=float), np.asarray(TTD, dtype=float))

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

    # Exclude offline period
    exclude_pattern = (r"'Offline Record: Data Excluded'|" r"'Maintenance Record: Data Excluded'")
    CRDF = CRDF.loc[~CRDF["tc_Classification"].str.contains(exclude_pattern, na=False)]
    DRDF = DRDF.loc[~DRDF["dr_Classification"].str.contains(exclude_pattern, na=False)]

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