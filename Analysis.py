import ast
import time
import itertools
import pandas as pd
import numpy as np
from Metrics import timeDifference
from Metrics import OperationalFactorsAnalysis
from Metrics import reportingDelay
from Metrics import surveytime
from Metrics import AnalysisCrossCheck
from Classification import DataPairing

def extractTimeBoundaries(intervalDF, start_col, end_col):
    """
        Extract the overall temporal boundaries of a set of time intervals and calculate the total elapsed time in seconds.

        Parameters
        ----------
        intervalDF : pandas.DataFrame - DataFrame containing start and end datetime columns.
        start_col : str - Column name containing interval start datetimes.
        end_col : str - Column name containing interval end datetimes.

        Returns
        -------
        total_seconds : float - Total elapsed time in seconds between the earliest start and latest end datetimes,
        including intermediate gaps.

        Notes
        -----
        All timestamps are converted to UTC. Timezone-naive timestamps are assumed to represent UTC. If the DataFrame is
         empty or either temporal boundary is missing, the total duration is returned as 0.0.
    """

    # Handle an empty input DataFrame.
    if intervalDF.empty:
        return 0.0

    # Create a copy to avoid modifying the original DataFrame.
    intervalDF = intervalDF.copy()

    # Convert start and end columns to UTC datetime objects.
    intervalDF[start_col] = pd.to_datetime(intervalDF[start_col], utc=True)
    intervalDF[end_col] = pd.to_datetime(intervalDF[end_col], utc=True)

    # Extract the overall temporal boundaries.
    start_time = intervalDF[start_col].min()
    end_time = intervalDF[end_col].max()

    # Handle missing temporal boundaries.
    if pd.isna(start_time) or pd.isna(end_time):
        return start_time, end_time, 0.0

    # Validate the extracted temporal boundaries.
    if start_time > end_time:
        raise ValueError("Start datetime cannot be later than end datetime.")

    # Calculate the total elapsed time in seconds.
    total_seconds = (end_time - start_time).total_seconds()

    return total_seconds

def calculateTotalDuration(dataDF, start_col, end_col):
    """
        Calculate the total duration covered by a set of time intervals, excluding intermediate gaps and avoiding double
        counting of overlapping intervals.

        Parameters
        ----------
        dataDF : pandas.DataFrame - DataFrame containing start and end datetime columns.
        start_col : str - Column name containing interval start datetimes.
        end_col : str - Column name containing interval end datetimes.

        Returns: pandas.Timedelta - Total non-overlapping duration covered by the intervals.
        -------

        Notes
        -----
        Overlapping and adjacent intervals are merged before calculating the total duration. Rows with missing
        timestamps are excluded. All timestamps are converted to UTC, with timezone-naive timestamps assumed to
        represent UTC.
    """
    if dataDF.empty:
        return 0.0

    # Retain only the required columns without modifying the original data.
    df = dataDF[[start_col, end_col]].copy()

    # Convert timestamps to datetime objects.
    df[start_col] = pd.to_datetime(df[start_col], utc=True)
    df[end_col] = pd.to_datetime(df[end_col], utc=True)

    # Remove intervals with missing timestamps.
    df = df.dropna(subset=[start_col, end_col])

    # Validate interval boundaries.
    if (df[end_col] < df[start_col]).any():
        raise ValueError("End datetime cannot precede start datetime.")

    # Sort intervals chronologically by start and end times.
    df = df.sort_values(by=[start_col, end_col])

    # Return zero duration if no valid intervals exist.
    if df.empty:
        return pd.Timedelta(0)

    # Initialise the first interval.
    current_start = df[start_col].iloc[0]
    current_end = df[end_col].iloc[0]
    total_duration = pd.Timedelta(0)

    # Iterate through the remaining intervals.
    for start, end in df.iloc[1:].itertuples(index=False, name=None):
        if start <= current_end:
            # Merge overlapping or adjacent intervals.
            current_end = max(current_end, end)
        else:
            # Add the completed interval duration.
            total_duration += current_end - current_start

            # Start a new interval, excluding the intermediate gap.
            current_start = start
            current_end = end

    # Add the duration of the final merged interval.
    total_duration += current_end - current_start
    return total_duration.total_seconds()

def contains_classification(value, target):
    """
        Check whether a target classification exists as an exact element within a list or a string representation
        of a list.

        Parameters
        ----------
        value : str or list
              Classification data containing one or more classification strings.
              May be an actual Python list or a string representation of a list.
        target : str
              Classification string to search for.
        Returns
        -------
            bool: True if the target exactly matches an element in the list.
                  False if the target is absent, the input is not a list, or the input cannot be parsed.

        Notes
        -----
        Matching is case-sensitive and independent of the number or order of elements in the list.
        Partial string matches are not accepted.
    """
    try:
        # Convert a string representation of a list into a Python object. If the input is already a Python object,
        # retain it unchanged.
        records = ast.literal_eval(value) if isinstance(value, str) else value

        # Verify that records is a list and check for exact membership of the target classification within the list.
        return isinstance(records, list) and target in records

    except (ValueError, SyntaxError):
        # Return False if the input string cannot be parsed as a valid Python literal, preventing execution from
        # being interrupted.
        return False

# --------------------------------------------------------------------------------------------------------
# Helper function to safely calculate percentages
# --------------------------------------------------------------------------------------------------------
# Calculates a percentage rounded to two decimal places.
# Returns np.nan when the denominator is zero, indicating that the
# metric is undefined due to insufficient valid observations.
# Empty classification DataFrames contribute zero observations.
def percentage(numerator, denominator):
    return round((numerator * 100) / denominator, 2) if denominator > 0 else np.nan


def analysis(analyStarttime, analyEndtime, ASEs, receivedTime, cumFactor, opLogDF, op_DateTime, pob_DateTime,
             controlledReleaseDF, detectionReportDF, passiveObservationDF, dr_baselineEmissionRate,
             tc_baselineEmissionRate, cr_releaseRate, dr_releaseRate, cr_emissionrateLower, cr_emissionrateUpper,
             dr_emissionrateLower, dr_emissionrateUpper, MaintenanceRecordDF, mr_DateTime, cr_DateTime,
             dr_DateTime, OfflineRecordDF, OfflineDateTime, OnlineDateTime, dr_ID_column, tc_ID_column,
             freq, bufferTime, ThresholdDuration, detection, ThresholdRateAboveBaseline, Method, listOfVariables2Sum_CR,
             listOfVariables2Average_CR, listOfVariables2ListUnique_CR, listOfVariables2Max_CR, listOfVariables2Min_CR,
             listOfVariables2ObtainFirst_CR, listOfVariables2ObtainLast_CR, listOfVariables2Sum_DR,
             listOfVariables2Average_DR, listOfVariables2ListUnique_DR, listOfVariables2Max_DR, listOfVariables2Min_DR,
             listOfVariables2ObtainFirst_DR, listOfVariables2ObtainLast_DR, tc_PolygonCoord, tc_BoundingZ,
             tc_localization, dr_PolygonCoord, dr_BoundingZ, dr_localization, ASE_textual_mapper,
             ASE_PolygonCoord_Dict, overLappingFrac, tc_bufferDistance, dr_bufferDistance):

    """
        Analyze controlled release and detection report data.

        Parameters: - ASEs (list): List of ASEs to analyze. - receivedTime (str): Column header of the time the
        detection report was received. - cumFactor (float): Multiplicative time factor to convert the time-basis of
        controlled release and detection data to per seconds basis. - opLogDF (Dataframe): The operator log dataframe
        - op_DateTime (list): A list of the column headers of the start and end datetime of the operator log -
        controlledReleaseDF (Dataframe): The controlled release dataframe - detectionReportDF (Dataframe): The
        detection report dataframe - dr_baselineEmissionRate (float): The baseline emission rate provided by the
        solution - tc_baselineEmissionRate (float): The baseline emission rate provided by the test center -
        cr_releaseRate (str): The column header of the controlled release rate - dr_releaseRate (str): The column
        header of the detection report rate - MaintenanceRecordDF (Dataframe): The maintenance record dataframe -
        mr_DateTime (list): A list of the column headers of the start and end datetime of the maintenance record. -
        cr_DateTime (list): A list of the column headers of the start and end datetime of the controlled release data
        - dr_DateTime (list): A list of the column headers of the start and end datetime of the detection report
        data. - OfflineRecordDF (Dataframe): The offline record dataframe - OfflineDateTime (str): The column header
        of the solution's offline date time - OnlineDateTime (str): The column header of the solution's online date
        time - dr_ID_column (str): The column header of detection report ID - tc_ID_column (str): The column header
        of emission source/experiment ID - freq (str): The frequency at which data is resolved (i.e, min, hours,
        etc.. must follow a specific format by python) - bufferTime (list): The duration in minutes applied before
        and after the start and end datetime of maintenance/offline record during data exclusion. - ThresholdDuration
        (int): The duration on the time basis of freq used for thresholding. - detection (str): Column header
        indicating 1 or 0 thresholding result of detection reports. - ThresholdRateAboveBaseline (float): The
        detection threshold above specified baseline - Method (str): The method used for thresholding based on
        supplied detection criteria - listOfVariables2Sum_CR (list): The list of controlled release data columns to
        perform summation operation on during aggregation - listOfVariables2Average_CR (list): The list of controlled
        release data columns to perform averaging operation on during aggregation - listOfVariables2ListUnique_CR (
        list): The list of controlled release data columns to extract unique list during aggregation -
        listOfVariables2Max_CR (list): The list of controlled release data columns to obtain the maximum value from
        during aggregation - listOfVariables2Min_CR (list): The list of controlled release data columns to obtain the
        minimum value from during aggregation - listOfVariables2ObtainFirst_CR (list): The list of controlled release
        data columns to obtain the first occurance of a value from during aggregation - listOfVariables2ObtainLast_CR
        (list): The list of controlled release data columns to obtain the last occurance of a value from during
        aggregation - listOfVariables2Sum_DR (list): The list of detection report data columns to perform summation
        operation on during aggregation - listOfVariables2Average_DR (list): The list of detection report data
        columns to perform averaging operation on during aggregation - listOfVariables2ListUnique_DR (list): The list
        of detection report data columns to extract unique list during aggregation. - listOfVariables2Max_DR (list):
        The list of detection report data columns to obtain the maximum value from during aggregation -
        listOfVariables2Min_DR (list): The list of detection report data columns to obtain the minimum value from
        during aggregation - listOfVariables2ObtainFirst_DR (list): The list of detection report data columns to
        obtain the first occurance of a value from during aggregation - listOfVariables2ObtainLast_DR (list): The
        list of detection report data columns to obtain the last occurance of a value from during aggregation -
        tc_PolygonCoord (str): The column header of the GPS coordinates (defined as polygons) defining the bounding
        area in which a controlled release is valid - tc_BoundingZ (list): List of column headers in the controlled
        release data that defines the minimum and maximum height that defines ASE of a controlled release -
        tc_localization (str): The column header of the localization identifier/BoundingName which can be used to map
        bounding volume/ASE for each controlled release. - dr_PolygonCoord (str): The column header of the GPS
        coordinates (defined as polygons) defining the bounding area in which a detection report is valid. -
        dr_BoundingZ (list): List of column headers in the detection report data that defines the minimum and maximum
        height that defines ASE of a detection report. - dr_localization (str): The column header of the localization
        identifier/BoundingName which can be used to map bounding volume for each detection report. -
        ASE_textual_mapper (dictionary): A dictionary mapping localization identifiers/BoundingName to their WKT
        polygon,reference coordinate, and the vertical boundary. - ASE_PolygonCoord_Dict (dictionary): A dictionary
        consisting of the parameters of the ASEs defined by the test center for the analysis - overLappingFrac (
        float): Values between 0 and 1 above which intersection fraction of 2 ASEs indicate inclusion of the ASEs in
        the analysis. - tc_bufferDistance (list): A list consisting of buffer distances in meters to increase (+) or
        decrease (-) defined ASEs for controlled release data. - dr_bufferDistance (list): A list consisting of
        buffer distances in meters to increase (+) or decrease (-) defined ASEs for detection report data.

            Returns:
                Tuple containing:
                    - Agg_CRDF (DataFrame): Aggregated controlled release data.
                    - Agg_DRDF (DataFrame): Aggregated detection report data.
                    - Agg_qDF (DataFrame): Paired quantification data.
                    - Agg_ClassifiedDF (DataFrame): Classified data.
                    - Agg_TTD (list): Time to Detection (TTD).
                    - Agg_TTA (list): Time to Alert (TTA).
                    - Agg_surveyTime (list): Survey times.
                    - Agg_reportDelay (list): Reporting delays.
                    - tc_offline_frac (float): Test-center offline fraction.
                    - sol_offline_frac (float): Solution offline fraction.
    """
    # Initialize parameters
    Agg_CRDF = pd.DataFrame() # Controlled release data (timeseries with resolution specified by frequency)
    Agg_DRDF = pd.DataFrame() # Detection report data (timeseries with resolution specified by frequency)
    Agg_PairedDF = pd.DataFrame() # Paired controlled release and detection report data
    Agg_TTD = [] # A distribution of time to detection
    Agg_TTA = [] # A distribution of time to alert
    Agg_surveyTime = [] # A distribution of survey time
    Agg_reportDelay = [] # A distribution of reporting delay
    detections = {} # Initialize detection performance metrics

    try:
        print("Evaluating reporting delay in seconds..")
        ## Evaluate reporting delays in seconds
        start_time = time.time()
        reportDelay = reportingDelay(detectionReportDF=detectionReportDF,
                                     dr_endTime=dr_DateTime[1],
                                     dr_ReceivedTime=receivedTime,
                                     timeBasisFactor=1)
        end_time = time.time()
        print(f"Code for evaluating reporting delay executed in {end_time - start_time:.6f} seconds")
    except Exception as e:
        print(f"Error evaluating reporting delay due to {e}")
        reportDelay = []

    ## Append evaluated reportdelay to the aggregated files
    Agg_reportDelay = list(itertools.chain.from_iterable([Agg_reportDelay, list(reportDelay)]))

    for ase in ASEs: # Loop through the ase's defined
        print(f"Conducting Analysis for ASE: {ase}")
        print("Aggregating and pairing detection reports with controlled releases.")
        start_time = time.time()
        drDF, crDF, PairedDF = DataPairing(analyStarttime=analyStarttime, analyEndtime=analyEndtime,
                                           controlledReleaseDF=controlledReleaseDF,
                                           detectionReportDF=detectionReportDF,
                                           dr_baselineEmissionRate=dr_baselineEmissionRate,
                                           tc_baselineEmissionRate=tc_baselineEmissionRate,
                                           cr_releaseRate=cr_releaseRate,
                                           dr_releaseRate=dr_releaseRate,
                                           cr_emissionrateBounds=[cr_emissionrateLower, cr_emissionrateUpper],
                                           dr_emissionrateBounds=[dr_emissionrateLower, dr_emissionrateUpper],
                                           MaintenanceRecordDF=MaintenanceRecordDF,
                                           mr_DateTime=mr_DateTime,
                                           cr_DateTime=cr_DateTime,
                                           dr_DateTime=dr_DateTime,
                                           pob_DateTime=pob_DateTime,
                                           passiveObservationDF=passiveObservationDF,
                                           OfflineRecordDF=OfflineRecordDF,
                                           OfflineDateTime=OfflineDateTime,
                                           OnlineDateTime=OnlineDateTime,
                                           dr_ID_column=dr_ID_column,
                                           tc_ID_column=tc_ID_column,
                                           ASE=ase,
                                           freq=freq,
                                           factor=cumFactor,
                                           bufferTime=bufferTime,
                                           ThresholdDuration=ThresholdDuration,
                                           ThresholdRateAboveBaseline=ThresholdRateAboveBaseline,
                                           detection=detection,
                                           Method=Method,
                                           listOfVariables2Sum_CR=listOfVariables2Sum_CR,
                                           listOfVariables2Average_CR=listOfVariables2Average_CR,
                                           listOfVariables2ListUnique_CR=listOfVariables2ListUnique_CR,
                                           listOfVariables2Max_CR=listOfVariables2Max_CR,
                                           listOfVariables2Min_CR=listOfVariables2Min_CR,
                                           listOfVariables2ObtainFirst_CR=listOfVariables2ObtainFirst_CR,
                                           listOfVariables2ObtainLast_CR=listOfVariables2ObtainLast_CR,
                                           listOfVariables2Sum_DR=listOfVariables2Sum_DR,
                                           listOfVariables2Average_DR=listOfVariables2Average_DR,
                                           listOfVariables2ListUnique_DR=listOfVariables2ListUnique_DR,
                                           listOfVariables2Max_DR=listOfVariables2Max_DR,
                                           listOfVariables2Min_DR=listOfVariables2Min_DR,
                                           listOfVariables2ObtainFirst_DR=listOfVariables2ObtainFirst_DR,
                                           listOfVariables2ObtainLast_DR=listOfVariables2ObtainLast_DR,
                                           tc_PolygonCoord=tc_PolygonCoord,
                                           tc_BoundingZ=tc_BoundingZ,
                                           tc_localization=tc_localization,
                                           dr_PolygonCoord=dr_PolygonCoord,
                                           dr_BoundingZ=dr_BoundingZ,
                                           dr_localization=dr_localization,
                                           ASE_textual_mapper=ASE_textual_mapper,
                                           ASE_PolygonCoord_Dict=ASE_PolygonCoord_Dict,
                                           overLappingFrac=overLappingFrac,
                                           tc_bufferDistance=tc_bufferDistance,
                                           dr_bufferDistance=dr_bufferDistance)
        end_time = time.time()
        print(f"Classification Code executed in {(end_time - start_time)/3600:.6f} hours for the ASE: {ase}")

        try:
            print("Evaluating Time to Detection (TTD) and Time to Alert (TTA) in seconds with the timebasis at default value = 1 sec..")
            start_time = time.time()
            TTA, TTD = timeDifference(DRDF=drDF, CRDF=crDF, RecievedTime=receivedTime, tc_Detection="tc_Detection",
                                      dr_Detection="dr_Detection", timebasis=1, temporal_resolution=freq,
                                      sample_end_timestamps=True)
            end_time = time.time()
            print(f"Code for evaluating TTA and TTD executed in {end_time - start_time:.6f} seconds")
        except Exception as e:
            print(f"Error evaluating TTA and TTD due to {e}")
            TTA, TTD = [], []

        try:
            print("Evaluate the test center and solution downtime.")
            # ----------------------------------------------------------------------------------------------------------
            # Total test center and solution downtime period.
            # ----------------------------------------------------------------------------------------------------------
            # The total controlled release period
            TC_total_seconds = extractTimeBoundaries(intervalDF=controlledReleaseDF,
                                                     start_col=cr_DateTime[0], end_col=cr_DateTime[1])
            # The total observation period
            DR_total_seconds = extractTimeBoundaries(intervalDF=detectionReportDF,
                                                     start_col=dr_DateTime[0], end_col=dr_DateTime[1])
            # Total test center downtime period
            tc_DT = calculateTotalDuration(dataDF=MaintenanceRecordDF, start_col=mr_DateTime[0], end_col=mr_DateTime[1])
            # Total solution downtime period
            dr_DT = calculateTotalDuration(dataDF=OfflineRecordDF, start_col=OfflineDateTime, end_col=OnlineDateTime)

            detections["TCDT_Total"] = percentage(tc_DT, TC_total_seconds)
            print(f"The test center downtime fraction: {detections['TCDT_Total']}%")
            # ----------------------------------------------------------------------------------------------------------
            # Total solution downtime period.
            # ----------------------------------------------------------------------------------------------------------
            detections["SDT_Total"] = percentage(dr_DT, DR_total_seconds)
            print(f"The solution downtime fraction: {detections['SDT_Total']}%")
        except Exception as e:
            print(f"The test center and solution downtime could not be evaluated due to {e}")
            detections["TCDT_Total"] = 0
            detections["SDT_Total"] = 0

        ## Append evaluated time to detection and alert to the aggregated files
        Agg_TTA = list(itertools.chain.from_iterable([Agg_TTA, list(TTA)]))
        Agg_TTD = list(itertools.chain.from_iterable([Agg_TTD, list(TTD)]))

        # Retain provenance when concatenating multiple spatial extents.
        for frame in (crDF, drDF, PairedDF):
            frame['ASE'] = ase

        # Append classified data and quantification data obtained for this ASE to the master dataframe
        Agg_PairedDF = pd.concat([Agg_PairedDF, PairedDF])
        Agg_PairedDF = Agg_PairedDF.reset_index(drop=True)

        # Rename the old index column for CRDF
        crDF.reset_index(inplace=True)
        crDF.rename(columns={"index": "Time"}, inplace=True)

        # Rename the old index column for DRDF
        drDF.reset_index(inplace=True)
        drDF.rename(columns={"index": "Time"}, inplace=True)

        # Concatenate to master dataframe
        Agg_CRDF = pd.concat([Agg_CRDF, crDF])
        Agg_DRDF = pd.concat([Agg_DRDF, drDF])

    try:
        print("Evaluate survey times in seconds")
        start_time = time.time()
        surveyTime = surveytime(DF=opLogDF,
                                startTime=op_DateTime[0],
                                endTime=op_DateTime[1],
                                timeBasisFactor=1)
        end_time = time.time()
        print(f"Code for survey times executed in {end_time - start_time:.6f} seconds")
    except Exception as e:
        print(f"Error evaluating survey times due to {e}")
        surveyTime = []

    try:
        print("Analysis cross-check: Quantification")
        AnalysisCrossCheck(Original_CRDF=controlledReleaseDF,
                           Original_DRDF=detectionReportDF,
                           QuantData=Agg_PairedDF,
                           Analyzed_CRDF=Agg_CRDF,
                           Analyzed_DRDF=Agg_DRDF,
                           cr_releaseRate=cr_releaseRate,
                           dr_releaseRate=dr_releaseRate,
                           cr_DateTime=cr_DateTime,
                           dr_DateTime=dr_DateTime,
                           timefactor=cumFactor)
    except Exception as e:
        print(f"Analysis cross-check: Quantification due to {e}")

    ## Append evaluated surveytimes to the aggregated files
    Agg_surveyTime = list(itertools.chain.from_iterable([Agg_surveyTime, list(surveyTime)]))

    # ---------------------------------------------------------------------------------------------------------
    # Extract classification data: Controlled-release classifications
    # ---------------------------------------------------------------------------------------------------------
    tpcrdf = Agg_CRDF.loc[Agg_CRDF["tc_Classification"] == "TP"]
    fncrdf = Agg_CRDF.loc[Agg_CRDF["tc_Classification"] == "FN"]
    tncrdf = Agg_CRDF.loc[Agg_CRDF["tc_Classification"] == "TN"]
    excrdf = Agg_CRDF.loc[Agg_CRDF["tc_Classification"] == "Below threshold to be classified an FN"]

    # --------------------------------------------------------------------------------------------------------
    # Extract classification data: Solution-reporting classifications
    # --------------------------------------------------------------------------------------------------------
    tpdrdf = Agg_DRDF.loc[Agg_DRDF["dr_Classification"] == "TP"]
    fpdrdf = Agg_DRDF.loc[Agg_DRDF["dr_Classification"] == "FP"]
    tndrdf = Agg_DRDF.loc[Agg_DRDF["dr_Classification"] == "TN"]
    exdrdf = Agg_DRDF.loc[Agg_DRDF["dr_Classification"] == "Below threshold to be classified an FP"]

    # --------------------------------------------------------------------------------------------------------
    # Calculate total valid observation intervals
    # --------------------------------------------------------------------------------------------------------
    # Each row represents a 30-second observation interval.
    # Because all intervals have equal duration, the number of rows is proportional to the total observation time.
    # total_cr: Total valid controlled-release observation intervals.
    total_cr = len(tpcrdf) + len(fncrdf) + len(tncrdf) + len(excrdf)
    # total_dr: Total valid solution-reporting observation intervals.
    total_dr = len(tpdrdf) + len(fpdrdf) + len(tndrdf) + len(exdrdf)

    # =========================================================================================================
    # CONDITIONAL DETECTION PERFORMANCE METRICS
    # =========================================================================================================
    # ---------------------------------------------------------------------------------------------------------
    # True Positive Time Fraction (TPTF)
    # ---------------------------------------------------------------------------------------------------------
    # Definition: The proportion of valid controlled-release time during which releases meeting the specified
    # detection threshold were correctly detected. This metric evaluates the solution's ability to correctly detect
    # controlled releases that satisfy the detection threshold criteria.
    # Formula: TPTF = TP_CR / (TP_CR + FN_CR) * 100; A higher TPTF indicates better detection sensitivity.
    detections["TPTF"] = percentage(len(tpcrdf), len(tpcrdf) + len(fncrdf))
    print(f"The number of controlled release time-steps classified as TP: {len(tpcrdf)}")
    print(f"The True Positive Time Fraction: {detections['TPTF']}%")

    # ---------------------------------------------------------------------------------------------------------
    # False Negative Time Fraction (FNTF)
    # ---------------------------------------------------------------------------------------------------------
    # Definition: The proportion of valid controlled-release time during which releases meeting the specified detection
    # threshold were not correctly detected. This metric evaluates the solution's failure to detect controlled releases
    # that satisfy the detection threshold criteria. Formula: FNTF = FN_CR / (TP_CR + FN_CR) * 100; A lower FNTF
    # indicates better detection performance. When defined, TPTF + FNTF = 100%.
    detections["FNTF"] = percentage(len(fncrdf), len(tpcrdf) + len(fncrdf))
    print(f"The number of controlled release time-steps classified as FN: {len(fncrdf)}")
    print(f"The False Negative Time Fraction: {detections['FNTF']}%")

    # ----------------------------------------------------------------------------------------------------------
    # False Positive Time Fraction (FPTF)
    # ----------------------------------------------------------------------------------------------------------
    # Definition: The proportion of valid solution-reported detection time during which the solution incorrectly
    # reported a detection when no controlled release was present or when the controlled release did not meet the
    # specified detection threshold criteria. This metric evaluates the proportion of reported detections that
    # were false positives rather than correctly identified detections. Formula: FPTF = FP_DR / (FP_DR + TP_DR) * 100
    # This definition corresponds mathematically to the False Discovery Proportion (FDP), rather than the conventional
    # false positive rate. A lower FPTF indicates greater reliability of reported detections.
    detections["FPTF"] = percentage(len(fpdrdf), len(fpdrdf) + len(tpdrdf))
    print(f"The number of observation report time-steps classified as FP: {len(fpdrdf)}")
    print(f"The False Positive Time Fraction: {detections['FPTF']}%")

    # ------------------------------------------------------------------------------------------------------------
    # True Negative Time Fraction (TNTF)
    # -------------------------------------------------------------------------------------------------------------
    # Definition: The proportion of valid non-detectable solution-observation time during which the solution correctly
    # reported no detection, either because no controlled release was present or because the controlled release did not
    # meet the specified detection threshold criteria. This metric evaluates the solution's ability to correctly identify
    # conditions that do not qualify as detectable releases. Formula: TNTF = TN_DR / (FP_DR + TN_DR) * 100
    # A higher TNTF indicates better performance in correctly identifying
    # non-detectable conditions.
    detections["TNTF"] = percentage(len(tndrdf), len(fpdrdf) + len(tndrdf))
    print(f"The number of observation report time-steps classified as TN: {len(tndrdf)}")
    print(f"The True Negative Time Fraction: {detections['TNTF']}%")

    # =============================================================================================================
    # TOTAL OBSERVATION TIME FRACTIONS
    # =============================================================================================================
    # -------------------------------------------------------------------------------------------------------------
    # Total True Positive Time Fraction (TPTF_Total)
    # -------------------------------------------------------------------------------------------------------------
    # Definition: The proportion of total valid controlled-release observation time during which releases meeting
    # the specified detection threshold were correctly detected. Unlike conditional TPTF, this metric considers all
    # valid emissions observation time, including true negatives and below-threshold observations.
    # Formula: TPTF_Total = TP_CR / Total_CR * 100. This metric indicates the percentage of the overall controlled-release
    # observation period classified as true positive.
    detections["TPTF_Total"] = percentage(len(tpcrdf), total_cr)
    print(f"The True Positive Time Fraction (Total Time): {detections['TPTF_Total']}%")

    # --------------------------------------------------------------------------------------------------------------
    # Total False Negative Time Fraction (FNTF_Total)
    # --------------------------------------------------------------------------------------------------------------
    # Definition:The proportion of total valid emissions observation time during which releases meeting the
    # specified detection threshold were not correctly detected. Unlike conditional FNTF, this metric includes all
    # valid emissions observation time in the denominator. Formula: FNTF_Total = FN_CR / Total_CR * 100. This metric
    # indicates the percentage of the overall controlled-release observation period classified as false negative.
    detections["FNTF_Total"] = percentage(len(fncrdf), total_cr)
    print(f"The False Negative Time Fraction (Total Time): {detections['FNTF_Total']}%")

    # ---------------------------------------------------------------------------------------------------------------
    # Total False Positive Time Fraction (FPTF_Total)
    # ---------------------------------------------------------------------------------------------------------------
    # Definition: The proportion of total valid solution-reporting observation time during which the solution
    # incorrectly reported a detection when no controlled release was present or when the controlled release did not
    # meet the specified detection threshold criteria. Unlike conditional FPTF, this metric considers all valid
    # solution-reporting intervals, including true positives, true negatives, and below-threshold observations.
    # Formula: FPTF_Total = FP_DR / Total_DR * 100. This metric indicates the percentage of the overall solution-reporting
    # observation period classified as false positive.
    detections["FPTF_Total"] = percentage(len(fpdrdf), total_dr)
    print(f"The False Positive Time Fraction (Total Time): {detections['FPTF_Total']}%")

    # ---------------------------------------------------------------------------------------------------------------
    # Total True Negative Time Fraction (TNTF_Total)
    # ---------------------------------------------------------------------------------------------------------------
    # Definition: The proportion of total valid controlled-release observation time during which the observation
    # was classified as a true negative. Unlike conditional TNTF, this metric considers all valid controlled-release
    # intervals, including true positives, false negatives, and below-threshold observations.
    # Formula: TNTF_Total = TN_CR / Total_CR * 100. This metric indicates the percentage of the overall controlled-release
    # observation period classified as true negative.
    detections["TNTF_Total"] = percentage(len(tncrdf), total_cr)
    print(f"The True Negative Time Fraction (Total Time): {detections['TNTF_Total']}%")

    # ---------------------------------------------------------------------------------------------------------------
    # Count of original and resulting controlled release and observation reports.
    # ---------------------------------------------------------------------------------------------------------------
    print(f"The original number of controlled releases: {len(controlledReleaseDF)}")
    detections["Orig_ControlledRelease"] = len(controlledReleaseDF)
    print(f"The original number of detection report: {len(detectionReportDF)}")
    detections["Orig_DetectionReport"] = len(detectionReportDF)
    print(f"The resulting number of effective controlled releases: {len(Agg_PairedDF.loc[Agg_PairedDF['Classification'].isin(['TP', 'FN'])])}")
    detections["Eff_ControlledRelease"] = len(Agg_PairedDF.loc[Agg_PairedDF['Classification'].isin(['TP', 'FN'])])
    print(f"The resulting number of effective detection report: {len(Agg_PairedDF.loc[Agg_PairedDF['Classification'].isin(['TP', 'FP'])])}")
    detections["Eff_DetectionReport"] = len(Agg_PairedDF.loc[Agg_PairedDF['Classification'].isin(['TP', 'FP'])])

    return Agg_CRDF, Agg_DRDF, Agg_PairedDF, Agg_TTD, Agg_TTA, Agg_surveyTime, Agg_reportDelay, detections
