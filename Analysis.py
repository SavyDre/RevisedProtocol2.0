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

        Parameters:
            - ASEs (list): List of ASEs to analyze.
            - receivedTime (str): Column header of the time the detection report was received.
            - cumFactor (float): Multiplicative time factor to convert the time-basis of controlled release and detection
              data to per seconds basis.
            - opLogDF (Dataframe): The operator log dataframe
            - op_DateTime (list): A list of the column headers of the start and end datetime of the operator log
            - controlledReleaseDF (Dataframe): The controlled release dataframe
            - detectionReportDF (Dataframe): The detection report dataframe
            - dr_baselineEmissionRate (float): The baseline emission rate provided by the solution
            - tc_baselineEmissionRate (float): The baseline emission rate provided by the test center
            - cr_releaseRate (str): The column header of the controlled release rate
            - dr_releaseRate (str): The column header of the detection report rate
            - MaintenanceRecordDF (Dataframe): The maintenance record dataframe
            - mr_DateTime (list): A list of the column headers of the start and end datetime of the maintenance record.
            - cr_DateTime (list): A list of the column headers of the start and end datetime of the controlled release data
            - dr_DateTime (list): A list of the column headers of the start and end datetime of the detection report data.
            - OfflineRecordDF (Dataframe): The offline record dataframe
            - OfflineDateTime (str): The column header of the solution's offline date time
            - OnlineDateTime (str): The column header of the solution's online date time
            - dr_ID_column (str): The column header of detection report ID
            - tc_ID_column (str): The column header of emission source/experiment ID
            - freq (str): The frequency at which data is resolved (i.e, min, hours, etc.. must follow a specific format by python)
            - bufferTime (list): The duration in minutes applied before and after the start and end datetime of maintenance/offline record during data exclusion.
            - ThresholdDuration (int): The duration on the time basis of freq used for thresholding.
            - detection (str): Column header indicating 1 or 0 thresholding result of detection reports.
            - ThresholdRateAboveBaseline (float): The detection threshold above specified baseline
            - Method (str): The method used for thresholding based on supplied detection criteria
            - listOfVariables2Sum_CR (list): The list of controlled release data columns to perform summation operation on during aggregation
            - listOfVariables2Average_CR (list): The list of controlled release data columns to perform averaging operation on during aggregation
            - listOfVariables2ListUnique_CR (list): The list of controlled release data columns to extract unique list during aggregation
            - listOfVariables2Max_CR (list): The list of controlled release data columns to obtain the maximum value from during aggregation
            - listOfVariables2Min_CR (list): The list of controlled release data columns to obtain the minimum value from during aggregation
            - listOfVariables2ObtainFirst_CR (list): The list of controlled release data columns to obtain the first occurance of a value from during aggregation
            - listOfVariables2ObtainLast_CR (list): The list of controlled release data columns to obtain the last occurance of a value from during aggregation
            - listOfVariables2Sum_DR (list): The list of detection report data columns to perform summation operation on during aggregation
            - listOfVariables2Average_DR (list): The list of detection report data columns to perform averaging operation on during aggregation
            - listOfVariables2ListUnique_DR (list): The list of detection report data columns to extract unique list during aggregation.
            - listOfVariables2Max_DR (list): The list of detection report data columns to obtain the maximum value from during aggregation
            - listOfVariables2Min_DR (list): The list of detection report data columns to obtain the minimum value from during aggregation
            - listOfVariables2ObtainFirst_DR (list): The list of detection report data columns to obtain the first occurance of a value from during aggregation
            - listOfVariables2ObtainLast_DR (list): The list of detection report data columns to obtain the last occurance of a value from during aggregation
            - tc_PolygonCoord (str): The column header of the GPS coordinates (defined as polygons) defining the bounding area in which a controlled release is valid
            - tc_BoundingZ (list): List of column headers in the controlled release data that defines the minimum and maximum height that defines ASE of a controlled release
            - tc_localization (str): The column header of the localization identifier/BoundingName which can be used to map bounding volume/ASE for each controlled release.
            - dr_PolygonCoord (str): The column header of the GPS coordinates (defined as polygons) defining the bounding area in which a detection report is valid.
            - dr_BoundingZ (list): List of column headers in the detection report data that defines the minimum and maximum height that defines ASE of a detection report.
            - dr_localization (str): The column header of the localization identifier/BoundingName which can be used to map bounding volume for each detection report.
            - ASE_textual_mapper (dictionary): A dictionary mapping localization identifiers/BoundingName to their WKT polygon,reference coordinate, and the vertical boundary.
            - ASE_PolygonCoord_Dict (dictionary): A dictionary consisting of the parameters of the ASEs defined by the test center for the analysis
            - overLappingFrac (float): Values between 0 and 1 above which intersection fraction of 2 ASEs indicate inclusion of the ASEs in the analysis.
            - tc_bufferDistance (list): A list consisting of buffer distances in meters to increase (+) or decrease
              (-) defined ASEs for controlled release data.
            - dr_bufferDistance (list): A list consisting of buffer distances in meters to increase (+) or decrease
              (-) defined ASEs for detection report data.

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
    Agg_CRDF = pd.DataFrame() # Time series CRDF
    Agg_DRDF = pd.DataFrame() # Time series DRDF
    Agg_ClassifiedDF = pd.DataFrame() # Paired Detection Data
    Agg_TTD = []
    Agg_TTA = []
    Agg_surveyTime = []
    Agg_reportDelay = []

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
        drDF, crDF, ClassifiedDF = DataPairing(analyStarttime=analyStarttime, analyEndtime=analyEndtime,
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
            TTA, TTD = timeDifference(DRDF=drDF,
                                      CRDF=crDF,
                                      RecievedTime=receivedTime,
                                      tc_Detection="tc_Detection",
                                      dr_Detection="dr_Detection",
                                      timebasis=1)
            end_time = time.time()
            print(f"Code for evaluating TTA and TTD executed in {end_time - start_time:.6f} seconds")
        except Exception as e:
            print(f"Error evaluating TTA and TTD due to {e}")
            TTA, TTD = [], []

        ## Append evaluated time to detection and alert to the aggregated files
        Agg_TTA = list(itertools.chain.from_iterable([Agg_TTA, list(TTA)]))
        Agg_TTD = list(itertools.chain.from_iterable([Agg_TTD, list(TTD)]))

        # Retain provenance when concatenating multiple spatial extents.
        for frame in (crDF, drDF, ClassifiedDF):
            frame['ASE'] = ase

        # Append classified data and quantification data obtained for this ASE to the master dataframe
        Agg_ClassifiedDF = pd.concat([Agg_ClassifiedDF, ClassifiedDF])
        Agg_ClassifiedDF = Agg_ClassifiedDF.reset_index(drop=True)

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
        print("Evaluate the solutions (opp_sol) and test center (opp_tc) operational factor..")
        start_time = time.time()
        tc_offline_frac, sol_offline_frac = OperationalFactorsAnalysis(CRDF=Agg_CRDF, DRDF=Agg_DRDF)
        end_time = time.time()
        print(f"Code for operational factor executed in {end_time - start_time:.6f} seconds")
    except Exception as e:
        print(f"Error evaluating operational factor due to {e}")
        tc_offline_frac, sol_offline_frac = np.nan, np.nan

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
                           QuantData=Agg_ClassifiedDF,
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

    return Agg_CRDF, Agg_DRDF, Agg_ClassifiedDF, Agg_TTD, Agg_TTA, Agg_surveyTime, Agg_reportDelay, tc_offline_frac, sol_offline_frac
