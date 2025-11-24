import sys
import pytz
import time
import pathlib
import datetime
import argparse
from Analysis import analysis
from Plotting import plotting
from LoadData import loadData
from Dict_to_JSON_to_Dict import load_dict_from_json
from SupportingCodes import run_script_file_and_call
from SupportingCodes import validate_frequency
from writeFiles import writeOutputData

def main(args):
    """
    This code implements the analysis section of the protocol found on the webpage: https://metec.colostate.edu/aded-2-0/

    The code accepts controlled release data, detection report data, maintenance record data, offline record data, and
    operator log to perform detection classification, assess performance metrics stated in the protocol, and plots
    associated figures. Before using, read the instruction below and the definitions in the InputVariables.py file for
    the guidelines and meaning of the input variables. The code comes with an arbitrary data for demonstration. Please,
    if you identify any bug or incorrect logic, contact the METEC team for correction. The code is constantly
    being improved, so expect regular updates and improvements.

    : General info: During the analysis, controlled release data and reported detections are broken into time steps based
     on stipulated frequency (args.freq with default value of 30 seconds). Stipulated frequency must be 30 seconds or less.
     The code will end abruptly is frequencies outside this range is supplied.

    ****Assumptions****
    1. The controlled release data must contain columns of at least controlled release rate, event ID, start datetime,
       end datetime, and localization (either as BoundingName or (BoundingXY, BoundingZMinimum, and BoundingZMaximum)).
    2. The detection report data must contain columns of at least reported emission rate/detection, start datetime,
       end datetime, detection report ID, received time, and localization (either as BoundingName or BoundingXY,
       BoundingZMinimum, and BoundingZMaximum).
    3. The maintenance, offline, operator record should contain at least start and end datetimes.
    4. To run analysis for different configurations simultaneously, a unique input variable script must be created for each configuration.

    ****Cautions****
    1. Ensure that the dataframes do not have missing values. All reported data must be complete.
    2. Ensure the rates in the controlled release and detection report dataframes have the same unit.
    3. Ensure that each row of the detection report dataframe indicates a unique detection report that has a unique ID.
    4. Ensure that each row of the controlled release dataframe indicates a unique controlled release that has a unique ID.
    5. Ensure that all datetimes in the dataframes are in UTC with a common format. The only exception is the start/end datetime in the reported detection and offline report which must be in the form:  <start>/<end> - <yyyy-mm-ddThh:mm:ssZ>/<yyyy-mm-ddThh:mm:ssZ>
    6. When using BoundingName, provide a mapper in form of a JSON file (args.emissionSourceMapper) linking BoundingName to BoundingXY, BoundingZMinimum, and BoundingZMaximum

    ****Notes*****
    1. The code is structured to limit modifications of the code itself (except to modify the logic or plotting parameters like fontsize, figsize, etc).
       Every input parameter (from column headers, filepaths, and other required analysis information) not provided in the args is supplied in the "InputVariables.py" file.
       This includes plotting parameters that depend on the input data. Therefore, DO NOT modify the maincode to run the analysis. The InputVariables.py can be renamed.
    2. The detection classification process can perform analysis on a modular level i.e., break down a test facility
       into analysis spatial extents (ASEs) as long as they are listed in args.ASEs and also included in the dictionary data["ASE"] in InputVariables.py
    3. The figures plot the aggregated metrics analysed for each ASE listed in args.ASEs.

    ****Application****
    1. To avoid spurious false negative and false positive classifications, scanning/imaging (unless monitoring the whole test area)
       should define ASE's based on the assets/field of view (FoV) monitored at a time but provide offline records accounting for every minute
       (could be seconds depending on the value of the freq parameter) that FoV is not being monitored.
    2. Similarly, survey LDAQ solutions/solutions that require human supervision to operate must report offline periods for
       anytime the ASE defined for the analysis/survey is not being actively monitored or surveyed.
    3. Solutions must report detections for every monitoring/survey.

    :param args:
    - args.ASEs: A list of applied spatial extent (ASE) names corresponding to keys of the dictionary data["ASE"] in InputVariables.py
    - args.bufferDistance (default [0, 0]): A list consisting of buffer distances in meters to increase (+) or decrease
      (-) defined ASEs for controlled release data and detection report, respectively.
    - args.overlappingFraction (default 0): Values between 0 and 1 above which intersection fraction of 2 ASEs indicate inclusion of the ASEs in the analysis.
    - args.bufferTime (default [0, 0]): The buffer times in minutes applied before the start datetime and after the end
      datetimes of maintenance and offline records during data exclusion.
    - args.freq (default min): The frequency at which to subdivide/resolve controlled release and detection report data.
    - args.timezone (default US/Mountain): The timezone of the args.startDateTime and args.startDateTime.
    - args.startDateTime: The startdate time of the analysis supplied in your local timezone in the format: YYYYMMDDHHMM.
    - args.endDateTime: The enddate time of the analysis supplied in your local timezone in the format: YYYYMMDDHHMM.
    - args.outputFilePath: The path to the output/result folder.
    - args.performer: The name of the solution/performer being analysed.
    - args.detectionReportPath: The filepath to the csv/xlsx file of detection report.
    - args.controlledReleasePath: The filepath to the csv/xlsx file of controlled release.
    - args.maintenanceRecordPath: The filepath to the csv/xlsx file of maintenance record.
    - args.offlineFilePath: The filepath to the csv/xlsx file of offline record.
    - args.operatorLogFilePath: The filepath to the csv/xlsx file of operator record.
    - args.emissionSourceMapper: The filepath to the JSON file used to map BoundingName to BoundingXY,
      BoundingZMinimum, and BoundingZMaximum. The keys of the JSON file must be same as the elements of the
      BoundingName column. The values of the JSON file must be of the same format as the value of the dictionary
      data["ASE"] in InputVariables.py
    - args.format: A string of the datetime format for datetimes in the controlled release data, maintenance record, offline record, and operator log.
    - args.inputVariablePath: The filepath to the inputVariablePath script.

    :return:
    1. Classified data.
    2. Processed controlled release data.
    3. Processed detection report data.
    4. POD curves.
    5. Histogram of Time to Alert, Detection Time, Reporting Delay, and Survey Time.
    6. A scatterplot of reported estimates against controlled release rates.
    7. CDFs comparing controlled release rate with reported estimates.
    """

    # Initialize dictionary for major variables.
    paperDict = {}

    print("Read and define input variables...")
    data = run_script_file_and_call(script_path=args.inputVariablePath,
                                    function_name='inputVariables')
    OfflineLogVar = data["OfflineRecord"] # Extract offline data variables.
    OperatorLogVar = data["OperatorLog"] # Extract operator log data variables.
    MaintenanceLogVar = data["MaintenanceRecord"] # Extract maintenance record data variables.
    DetectionReportLogVar = data["DetectionReportData"] # Extract detection report data variables.
    ControlledReleaseLogVar = data["ControlledReleaseData"] # Extract controlled release data variables.
    plottingVar = data["Plotting"]
    ASEs = args.ASEs.strip('][').split(', ')
    bufferDistance = [float(d) for d in args.bufferDistance.strip('][').split(', ')]
    overlappingFraction = float(args.overlappingFraction)
    bufferTime = [float(t) for t in args.bufferTime.strip('][').split(', ')]
    freq = str(args.freq)
    runstatus = validate_frequency(suppliedfreq=freq,
                                   referencefreq="30s") # Check if supplied frequency is allowed for the analysis
    if not runstatus:
        sys.exit()

    print("Reformat analysis datetime settings...")
    tz = pytz.timezone(args.timezone)
    startDateTime = tz.localize(datetime.datetime.strptime(args.startDateTime, "%Y%m%d%H%M")).astimezone(pytz.utc)  # Analysis start time in UTC
    endDateTime = tz.localize(datetime.datetime.strptime(args.endDateTime, "%Y%m%d%H%M")).astimezone(pytz.utc) # Analysis end time in UTC

    start_time = time.time()
    print("Setup directories for saving results...")
    saveDir = pathlib.PurePath(args.outputFilePath, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + args.performer)
    pathlib.Path(saveDir).mkdir(parents=True,
                                exist_ok=True)
    figurefilepath = saveDir / 'figures'
    datafilepath = saveDir / 'data'
    pathlib.Path(figurefilepath).mkdir(parents=True,
                                       exist_ok=True)
    pathlib.Path(datafilepath).mkdir(parents=True,
                                     exist_ok=True)

    print("Loading detection report data with associated datetimes in UTC...")
    dr_datetimes = [DetectionReportLogVar["StartDateTime"], DetectionReportLogVar["EndDateTime"], DetectionReportLogVar["ReceivedTime"]]
    DRDF = loadData(Path=args.detectionReportPath,
                    DateTimes=dr_datetimes,
                    StartTime=startDateTime,
                    EndTime=endDateTime,
                    Format=args.format)  # Detection report data
    print("Loading controlled release data with associated datetimes in UTC...")
    cr_datetimes = [ControlledReleaseLogVar["StartDateTime"], ControlledReleaseLogVar["EndDateTime"]]
    CRDF = loadData(Path=args.controlledReleasePath,
                    DateTimes=cr_datetimes,
                    StartTime=startDateTime,
                    EndTime=endDateTime,
                    Format=args.format)  # Controlled release data
    print("Loading maintenance report data with associated datetimes in UTC...")
    mr_datetimes = [MaintenanceLogVar["StartDateTime"], MaintenanceLogVar["EndDateTime"]]
    MaintDF = loadData(Path=args.maintenanceRecordPath,
                       DateTimes=mr_datetimes,
                       StartTime=startDateTime,
                       EndTime=endDateTime,
                       Format=args.format)  # Maintenance record data
    print("Loading offline report data with associated datetimes in UTC...")
    of_datetimes = [OfflineLogVar["OfflineDateTime"], OfflineLogVar["OnlineDateTime"]]
    OfflineDF = loadData(Path=args.offlineFilePath,
                         DateTimes=of_datetimes,
                         StartTime=startDateTime,
                         EndTime=endDateTime,
                         Format=args.format)  # Offline record data
    print("Loading operator log data with associated datetimes in UTC...")
    op_datetimes = [OperatorLogVar["StartDateTime"], OperatorLogVar["EndDateTime"]]
    OplogDF = loadData(Path=args.operatorLogFilePath,
                       DateTimes=op_datetimes,
                       StartTime=startDateTime,
                       EndTime=endDateTime,
                       Format=args.format)  # Operator record data
    # A dictionary mapping emission sources to their WKT polygon,reference coordinate, and the vertical boundary.
    Dict_EPmapper = load_dict_from_json(file_path=args.emissionSourceMapper)

    print("Evaluate data classification and performance metrics...")
    CRDF, DRDF, quantDF, ClassifiedDF, runs_cDF, TTD, TTA, SurveyTime, ReportDelay, tc_offline_frac, sol_offline_frac = \
        analysis(ASEs=ASEs,
                 receivedTime=DetectionReportLogVar["ReceivedTime"],
                 cumFactor=plottingVar["AggQuant"]["timeFactor"],
                 opLogDF=OplogDF,
                 op_DateTime=[OperatorLogVar["StartDateTime"], OperatorLogVar["EndDateTime"]],
                 controlledReleaseDF=CRDF,
                 detectionReportDF=DRDF,
                 tc_baselineEmissionRate=ControlledReleaseLogVar["baselineEmissionRate"],
                 dr_baselineEmissionRate=DetectionReportLogVar["baselineEmissionRate"],
                 cr_releaseRate=ControlledReleaseLogVar["EmissionRate"],
                 dr_releaseRate=DetectionReportLogVar["EmissionRate"],
                 cr_emissionrateLower=ControlledReleaseLogVar["EmissionRateLower"],
                 cr_emissionrateUpper=ControlledReleaseLogVar["EmissionRateUpper"],
                 dr_emissionrateLower=DetectionReportLogVar["EmissionRateLower"],
                 dr_emissionrateUpper=DetectionReportLogVar["EmissionRateUpper"],
                 MaintenanceRecordDF=MaintDF,
                 mr_DateTime=[MaintenanceLogVar["StartDateTime"], MaintenanceLogVar["EndDateTime"]],
                 cr_DateTime=[ControlledReleaseLogVar["StartDateTime"], ControlledReleaseLogVar["EndDateTime"]],
                 dr_DateTime=[DetectionReportLogVar["StartDateTime"], DetectionReportLogVar["EndDateTime"]],
                 OfflineRecordDF=OfflineDF,
                 OfflineDateTime=OfflineLogVar["OfflineDateTime"],
                 OnlineDateTime=OfflineLogVar["OnlineDateTime"],
                 dr_ID_column=DetectionReportLogVar["ReportIDs"],
                 tc_ID_column=ControlledReleaseLogVar["SourceID"],
                 freq=freq,
                 bufferTime=bufferTime,
                 detection=DetectionReportLogVar["detection"],
                 ThresholdDuration=DetectionReportLogVar["ThresholdDuration"],
                 ThresholdRateAboveBaseline=DetectionReportLogVar["ThresholdRateAboveBaseline"],
                 nSampling=DetectionReportLogVar["nSampling"],
                 Method=DetectionReportLogVar["Method"],
                 listOfVariables2Sum_CR=ControlledReleaseLogVar["listOfVariables2Sum"],
                 listOfVariables2Average_CR=ControlledReleaseLogVar["listOfVariables2Average"],
                 listOfVariables2ListUnique_CR=ControlledReleaseLogVar["listOfVariables2ListUnique"],
                 listOfVariables2Max_CR=ControlledReleaseLogVar["listOfVariables2Max"],
                 listOfVariables2Min_CR=ControlledReleaseLogVar["listOfVariables2Min"],
                 listOfVariables2ObtainFirst_CR=ControlledReleaseLogVar["listOfVariables2ObtainFirst"],
                 listOfVariables2ObtainLast_CR=ControlledReleaseLogVar["listOfVariables2ObtainLast"],
                 listOfVariables2Sum_DR=DetectionReportLogVar["listOfVariables2Sum"],
                 listOfVariables2Average_DR=DetectionReportLogVar["listOfVariables2Average"],
                 listOfVariables2ListUnique_DR=DetectionReportLogVar["listOfVariables2ListUnique"],
                 listOfVariables2Max_DR=DetectionReportLogVar["listOfVariables2Max"],
                 listOfVariables2Min_DR=DetectionReportLogVar["listOfVariables2Min"],
                 listOfVariables2ObtainFirst_DR=DetectionReportLogVar["listOfVariables2ObtainFirst"],
                 listOfVariables2ObtainLast_DR=DetectionReportLogVar["listOfVariables2ObtainLast"],
                 tc_PolygonCoord=ControlledReleaseLogVar["BoundingXY"],
                 tc_BoundingZ=[ControlledReleaseLogVar["BoundingZMinimum"], ControlledReleaseLogVar["BoundingZMaximum"]],
                 tc_localization=ControlledReleaseLogVar["BoundingName"],
                 dr_PolygonCoord=DetectionReportLogVar["BoundingXY"],
                 dr_BoundingZ=[DetectionReportLogVar["BoundingZMinimum"], DetectionReportLogVar["BoundingZMaximum"]],
                 dr_localization=DetectionReportLogVar["BoundingName"],
                 ASE_textual_mapper=Dict_EPmapper,
                 ASE_PolygonCoord_Dict=ControlledReleaseLogVar["ASE"],
                 overLappingFrac=overlappingFraction,
                 tc_bufferDistance=bufferDistance,
                 dr_bufferDistance=bufferDistance)

    print("Plotting performance metrics..")
    plotting(figurefilepath=figurefilepath,
             quantDF=quantDF,
             Agg_classifiedDF=runs_cDF,
             TTD=TTD,
             TTA=TTA,
             CRDF=CRDF,
             DRDF=DRDF,
             SurveyTime=SurveyTime,
             ReportDelay=ReportDelay,
             tc_baselineEmissionRate=ControlledReleaseLogVar["baselineEmissionRate"],
             singleQuant_xCol=plottingVar["SingleQuant"]["xCategory"],
             singleQuant_yCol=plottingVar["SingleQuant"]["yCategory"],
             singleQuant_xLowerCol=plottingVar["SingleQuant"]["xCategoryLower"],
             singleQuant_xUpperCol=plottingVar["SingleQuant"]["xCategoryUpper"],
             singleQuant_yLowerCol=plottingVar["SingleQuant"]["yCategoryLower"],
             singleQuant_yUpperCol=plottingVar["SingleQuant"]["yCategoryUpper"],
             singleQuant_xScalingFact=plottingVar["SingleQuant"]["xScalingFactor"],
             singleQuant_yScalingFact=plottingVar["SingleQuant"]["yScalingFactor"],
             singleQuant_xunit=plottingVar["SingleQuant"]["xunit"],
             singleQuant_yunit=plottingVar["SingleQuant"]["yunit"],
             singleQuant_limit=plottingVar["SingleQuant"]["limits"],
             singleQuant_quantFactor=plottingVar["SingleQuant"]["quantificationFactor"],
             singleQuant_xScale=plottingVar["SingleQuant"]["xScale"],
             singleQuant_yScale=plottingVar["SingleQuant"]["yScale"],
             singleQuant_legendloc=plottingVar["SingleQuant"]["legendloc"],
             aggQuant_xlimit=plottingVar["AggQuant"]["xlimit"],
             aggQuant_ylimit=plottingVar["AggQuant"]["ylimit"],
             aggQuant_xunit=plottingVar["AggQuant"]["xunit"],
             aggQuant_yunit=plottingVar["AggQuant"]["yunit"],
             aggQuant_yScalingFact=plottingVar["AggQuant"]["scalingFactor"],
             aggQuant_legendloc=plottingVar["AggQuant"]["legendloc"],
             aggQuant_all=plottingVar["AggQuant"]["IncludeAll"],
             TTD_timeFact=plottingVar["TTD"]["timeFactor"],
             TTD_nBins=plottingVar["TTD"]["bins"],
             TTD_xunit=plottingVar["TTD"]["unit"],
             TTD_xlimit=plottingVar["TTD"]["xlimit"],
             TTD_ylimit=plottingVar["TTD"]["ylimit"],
             TTD_legendloc=plottingVar["TTD"]["legendloc"],
             TTA_timeFact=plottingVar["TTA"]["timeFactor"],
             TTA_nBins=plottingVar["TTA"]["bins"],
             TTA_xunit=plottingVar["TTA"]["unit"],
             TTA_xlimit=plottingVar["TTA"]["xlimit"],
             TTA_ylimit=plottingVar["TTA"]["ylimit"],
             TTA_legendloc=plottingVar["TTA"]["legendloc"],
             RD_timeFact=plottingVar["ReportDelay"]["timeFactor"],
             RD_nBins=plottingVar["ReportDelay"]["bins"],
             RD_xunit=plottingVar["ReportDelay"]["unit"],
             RD_xlimit=plottingVar["ReportDelay"]["xlimit"],
             RD_ylimit=plottingVar["ReportDelay"]["ylimit"],
             RD_legendloc=plottingVar["ReportDelay"]["legendloc"],
             ST_nBins=plottingVar["SurveyTime"]["bins"],
             ST_xunit=plottingVar["SurveyTime"]["unit"],
             ST_xlimit=plottingVar["SurveyTime"]["xlimit"],
             ST_ylimit=plottingVar["SurveyTime"]["ylimit"],
             ST_legendloc=plottingVar["SurveyTime"]["legendloc"],
             ST_timeFact=plottingVar["SurveyTime"]["timeFactor"],
             desiredLDLFraction=plottingVar["POD-CF"]["desiredLDLFraction"],
             Nbootstrap=plottingVar["POD-CF"]["Nbootstrap"],
             LR_xCol=plottingVar["POD-LR"]["xCategory"],
             LR_xScalingFact=plottingVar["POD-LR"]["scalingFactor"],
             LR_xunit=plottingVar["POD-LR"]["unit"],
             LR_xlimit=plottingVar["POD-LR"]["xlimit"],
             LR_legendloc=plottingVar["POD-LR"]["legendloc"],
             LR_xScale=plottingVar["POD-LR"]["xScale"],
             LR_yScale=plottingVar["POD-LR"]["yScale"],
             LR_xLabel=plottingVar["POD-LR"]["xlabel"],
             LR_nSamples=plottingVar["POD-LR"]["nSamples"],
             CF_nSamples=plottingVar["POD-CF"]["nSamples"],
             CF_xCol=plottingVar["POD-CF"]["xCategory"],
             CF_xScalingFact=plottingVar["POD-CF"]["scalingFactor"],
             CF_xunit=plottingVar["POD-CF"]["unit"],
             CF_xlimit=plottingVar["POD-CF"]["xlimit"],
             CF_step=plottingVar["POD-CF"]["xstep"],
             CF_legendloc=plottingVar["POD-CF"]["legendloc"],
             CF_xScale=plottingVar["POD-CF"]["xScale"],
             CF_yScale=plottingVar["POD-CF"]["yScale"],
             CF_xLabel=plottingVar["POD-CF"]["xlabel"],
             bounds=plottingVar["POD-CF"]["bounds"],
             a0s=plottingVar["POD-CF"]["a0s"],
             b0s=plottingVar["POD-CF"]["b0s"],
             plot_singlequant=plottingVar["SingleQuant"]["plot"],
             plot_TTD=plottingVar["TTD"]["plot"],
             plot_TTA=plottingVar["TTA"]["plot"],
             plot_RD=plottingVar["ReportDelay"]["plot"],
             plot_ST=plottingVar["SurveyTime"]["plot"],
             plot_LR=plottingVar["POD-LR"]["plot"],
             plot_CF=plottingVar["POD-CF"]["plot"],
             plot_aggquant=plottingVar["AggQuant"]["plot"])

    print(f"The test center was offline/not operating {tc_offline_frac * 100}% of the test duration")
    print(f"The solution was offline/not operating {sol_offline_frac * 100}% of the test duration")

    print("Writing data to files..")
    writeOutputData(saveDir,
                    datapath=datafilepath,
                    classifiedDF=ClassifiedDF,
                    DetectionData=DRDF,
                    TestCenterData=CRDF,
                    quantDF=quantDF,
                    paperDict=paperDict)

    end_time = time.time()
    print(f"Code executed in {(end_time - start_time)/3600:.6f} hours")
if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Data analysis using the Protocol.")
    p.add_argument('-p', '--performer', help="The name of performer.", default="Solution")
    p.add_argument('-s', '--startDateTime', help="Start date and time for analysis. Format - YYYYMMDDhhmm.")
    p.add_argument('-e', '--endDateTime', help="End date and time for analysis. Format - YYYYMMDDhhmm.")
    p.add_argument('-tz', '--timezone', help="Timezone of the start and end datetime", default="US/Mountain")
    p.add_argument('-d', '--detectionReportPath', help="File path to detection report data.")
    p.add_argument('-c', '--controlledReleasePath', help="File path to controlled release data.")
    p.add_argument('-m', '--maintenanceRecordPath', help="File path to test center's maintenance record.")
    p.add_argument('-l', '--ASEs', help="A list of test center's analysis spatial extents (ASE) to aggregate.")
    p.add_argument('-bd', '--bufferDistance', help="list of ASE buffer distance in meters.", default="[0, 0]")
    p.add_argument('-ov', '--overlappingFraction', help="The fraction of overlapping between two ASEs.", default=0)
    p.add_argument('-mp', '--emissionSourceMapper', help=" Path to JSON file mapping emission sources to WKT polygons.")
    p.add_argument('-of', '--offlineFilePath', help="File path to reported offline data.")
    p.add_argument('-op', '--operatorLogFilePath', help="File path to the test center's operator log")
    p.add_argument('-o', '--outputFilePath', help="Directory to save all output data to.")
    p.add_argument('-bt', '--bufferTime', help="List of buffer times in minutes to apply to exclusion records.", default="[0, 0]")
    p.add_argument('-f', '--freq', help="The frequency of temporary resolution.", default="min")
    p.add_argument('-fm', '--format', help="The string format of datetime.", default="%Y%m%d %H:%M:%S")
    p.add_argument('-ivp', '--inputVariablePath', help="Path to input variable.")

    args = p.parse_args()
    main(args)