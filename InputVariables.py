import numpy as np
from Dict_to_JSON_to_Dict import save_dict_to_json

def inputVariables():
    """
    A comprehensive description of the input variables for the classification analysis and performance metrics.
    Note: Ensure that all datetimes in the dataframes are in UTC with a common format. The only exception is the start/end datetime in the reported detection and offline report which must be in the form:  <start>/<end> - <yyyy-mm-ddThh:mm:ssZ>/<yyyy-mm-ddThh:mm:ssZ>
    - data[OfflineRecord] contains data required from the offline record provided by the solution.
        + OfflineDateTime: The column header of start date time of the offline period. It already has a default value which must not be changed.
        + OnlineDateTime: The column header of end date time of the offline period. It already has a default value which must not be changed.

    - data[OperatorLog] contains data required from the operator log provided by the test center.
        + StartDateTime: The column header of start date time of the operator log.
        + EndDateTime: The column header of end date time of the operator log.
        + NosOfAssetsSurveyed (optional): The column header of the number of assets covered during a survey.

    - data[MaintenanceRecord] contains data required from the maintenance record provided by the test center.
        + StartDateTime: The column header of start date time of the maintenance record.
        + EndDateTime: The column header of end date time of the maintenance record.

    - data[DetectionReportData] contains relevant data from the detection report provided by the solutions.
        + StartDateTime: The column header of the start date time of the detection report. It already has a default value which must not be changed.
        + EndDateTime: The column header of the end date time of the detection report. It already has a default value which must not be changed.
        + EmissionRate: The column header of the emission rate of the detection report.
        + ReportIDs: The column header of the observation report ID.
        + BoundingXY: The column header of the GPS coordinates (defined as WKT polygons) defining the bounding area in which a detection report is valid.
          The polygon must form a closed boundary (i.e., the first and last coordinates in the polygon must be the same) and and of this format depending in the number of vertices.
          This examples shows an example of a polygon of 3 vertices: "POLYGON ((40.59508012712645 -105.1400239508514, 40.59516024495923 -105.1401361785956, 40.59507806685163 -105.1402325404461, 40.59508012712645 -105.1400239508514))"
        + BoundingZMinimum: The column header of the minimum height forming a controlled volume in a detection report.
        + BoundingZMaximum: The column header of the maximum height forming a controlled volume in a detection report.
        + BoundingName: The column header of the localization identifier which can be used to map bounding volumes for each detection report.
        + ReceivedTime: The column header of the time stamp when the observation report was received by the test center.
        + detection: The column header of the detection report for mode 2 reporting (yes/no).
        + Method: Pre-defined methods used for applying thresholding. As at now, 3 methods have been defined.
            * Above Threshold Across Duration: Produces a thresholding value of 1 when the cumulative emissions duration at
              a timestep/instance > specified thresholding duration (default = 0), and controlled release/estimated emission
              rate > baseline estimate + detection threshold above baseline. If condition is not met, thresholding value is 0.
            * Rolling Average: Produces a thresholding value of 1 when controlled release/estimated emission rate at a given
              timestep is > the rolling average rate/estimate evaluated over a specified thresholding duration. If condition
              is not met, thresholding value is 0.
        + baselineEmissionRate: A float of the average baseline emission rate by the solution. It must have the same unit as
          the estimated emission rate.
        + ThresholdDuration:  An integer of the thresholding duration specified by the solution. Must have the same unit as
          the time resolution frequency (default is minutes).
        + ThresholdRateAboveBaseline: A float of the detection threshold above the average baseline emission rate by the
          solution. It must have the same unit as the estimated emission rate.
        + listOfVariables2Sum (default is None): A list of column headers in the detection report of which the sum of
          the data with the same time step (when aggregated) is of interest. For mode 1 reporting, the list must contain
          the column header of emission rate in the detection report.
        + listOfVariables2Average (default is None): A list of column headers in the detection report of which the mean
          of the data with the same time step (when aggregated) is of interest. Note: Only the mean of data in each column
          occurring at the same time step is evaluated.
        + listOfVariables2ListUnique (default is None): A list of column headers in the detection report of which a list
          of unique values for data with the same time step (when aggregated) is of interest. Note: Only the unique list
          of data in each column occurring at the same time step is evaluated.
        + listOfVariables2Max (default is None): A list of column headers in the detection report of which the maximum
          value for data with the same time step (when aggregated) is of interest. Note: Only the maximum value of data
          in each column occurring at the same time step is evaluated.
        + listOfVariables2Min (default is None): A list of column headers in the detection report of which the minimum
          value for data with the same time step (when aggregated) is of interest. Note: Only the minimum value of data
          in each column occurring at the same time step is evaluated.
        + listOfVariables2ObtainFirst (default is None): A list of column headers in the detection report of which the
          data associated with the lowest ReportIDs value for the same time step (when aggregated) is evaluated.
        + listOfVariables2ObtainLast (default is None): A list of column headers in the detection report of which the
          data associated with the largest ReportIDs value for that time step (when aggregated) is of interest.
          The list must contain at minimum the column headers for ReceivedTime.

    - data[ControlledReleaseData] contains relevant data from the controlled release data provided by the test center.
        + StartDateTime: The column header of the start date time of the controlled release data.
        + EndDateTime: The column header of the end date time of the controlled release data.
        + EmissionRate: The column header of the emission rate of the controlled release data.
        + SourceIDs: The column header of the unique source IDs associated with controlled releases.
        + BoundingXY: The column header of the GPS coordinates (defined as WKT polygons) defining the bounding area in
          which a controlled release is valid. The polygon must form a closed boundary (i.e., the first and last
          coordinates in the polygon must be the same) and of this format depending on the number of vertices. This
          examples shows an example of a polygon of 3 vertices: "POLYGON ((40.59508012712645 -105.1400239508514,
          40.59516024495923 -105.1401361785956, 40.59507806685163 -105.1402325404461, 40.59508012712645 -105.1400239508514))"
        + BoundingZMinimum: The column header of the minimum height forming a controlled volume for controlled release.
        + BoundingZMaximum: The column header of the maximum height forming a controlled volume for controlled release.
        + BoundingName: The column header of the localization identifier which can be used to map bounding volume for
          each controlled release.
        + baselineEmissionRate: A float of the average baseline emission rate by the test center. It must have the same
          unit as the controlled release rate.
        + listOfVariables2Sum (default is None): A list of column headers in the controlled release data of which the
          sum of the data with the same time step (when aggregated) is of interest. For mode 1 reporting, the list must
          contain the column header of controlled release rate in the controlled release.
        + listOfVariables2Average (default is None): A list of column headers in the controlled release data of which
          the mean of the data with the same time step (when aggregated) is of interest. Note: Only the mean of data in
          each column occurring at the same time step is evaluated.
        + listOfVariables2ListUnique (default is None): A list of column headers in the controlled release data of which
          a list of unique values for data with the same time step (when aggregated) is of interest. Note: Only the
          unique list of data in each column occurring at the same time step is evaluated.
        + listOfVariables2Max (default is None): A list of column headers in the controlled release data of which the
          maximum value for data with the same time step (when aggregated) is of interest. Note: Only the maximum value
          of data in each column occurring at the same time step is evaluated.
        + listOfVariables2Min (default is None): A list of column headers in the controlled release data of which the
          minimum value for data with the same time step (when aggregated) is of interest. Note: Only the minimum value
          of data in each column occurring at the same time step is evaluated.
        + listOfVariables2ObtainFirst (default is None): A list of column headers in the controlled release data of which
          the data associated with the lowest SourceID value for the same time step (when aggregated) is evaluated.
        + listOfVariables2ObtainLast (default is None): A list of column headers in the controlled release of which the
          data associated with the largest SourceID value for that time step (when aggregated) is of interest.
          The list must contain at minimum the column headers for ReceivedTime.

    - data["ASE"] is a dictionary of a dictionary containing the BoundingXY, BoundingZ, and the reference GPS coordinates
      of the applied spatial extent (ASE) defined for a subset of the test facility for analysis. The format and example
      of the inner dictionary is:
      ASEsubset: {"BoundingXY": "POLYGON ((40.59552772212062 -105.1396852051655, 40.59601267917348 -105.139683347761, 40.59600301434005 -105.1405294847709, 40.59554745867481 -105.1405484620506, 40.59552772212062 -105.1396852051655))",
                  "BoundingZ": [0.0, 25], "ReferenceCoord": [40.59577187772668, -105.14010720572513]}.

    -data["Plotting"] summarizes the plotting parameters for evaluating and plotting the primary performance metrics.
        plot: A boolean indicating if the plot should be made.
        + TTA: Time to Alert.
            * timeFactor (default 1/3600): A multiplicative factor to convert from seconds to the time basis specified by unit.
            * unit (default hours): The unit of time (e.g., minutes, hours, etc.)
            * bins (default 30): The number of bins.
            * xlimit (default None): The x-axis limit specified in unit of time specified.
            * ylimit (default None): The y-axis limit specified in unit of time specified.
            * legendloc (default upper right): The location of the plot legend.

        + TTD: Time to Detection.
            * timeFactor (default 1/3600): A multiplicative factor to convert from seconds to the time basis specified by unit.
            * unit (default hours): The unit of time (e.g., minutes, hours, etc.)
            * bins (default 30): The number of bins.
            * xlimit (default None): The x-axis limit specified in unit of time specified.
            * ylimit (default None): The y-axis limit specified in unit of time specified.
            * legendloc (default upper right): The location of the plot legend.

        + ReportDelay: Reporting delay
            * timeFactor (default 1/86400): A multiplicative factor to convert from seconds to the time basis specified by unit.
            * unit (default days): The unit of time (e.g., minutes, hours, etc.)
            * bins (default 30): The number of bins.
            * xlimit (default None): The x-axis limit specified in unit of time specified.
            * ylimit (default None): The y-axis limit specified in unit of time specified.
            * legendloc (default upper right): The location of the plot legend.

        + SurveyTime: Survey time
            * timeFactor (default 1/3600): A multiplicative factor to convert from seconds to the time basis specified by unit.
            * unit (default hours): The unit of time (e.g., minutes, hours, etc.)
            * bins (default 30): The number of bins.
            * xlimit (default None): The x-axis limit specified in unit of time specified.
            * ylimit (default None): The y-axis limit specified in unit of time specified.
            * legendloc (default upper right): The location of the plot legend.

        + POD-LR: Probability of Detection Curve Using Binary Logistic Regression
            * scalingFactor (default 1/1000): The multiplicative factor to scale x-axis data to basis specified by unit.
            * xCategory: The column header of the x-axis data.
            * unit (default r"$\mathrm{kg~CH_4/h}$"): The x-axis data unit.
            * xlimit: The lower and upper limit of the x-axis data specified on the basis of the unit specified.
            * xstep (default 0.0001): The x-axis data interval/step.
            * xScale (default linear): The type of scale applied on the x-axis.
            * yScale (default linear): The type of scale applied on the y-axis.
            * desiredLDLFraction (default 0.9): The y-axis data at which to evaluate probability of detection.
            * Nbootstrap (default 500): An integer representing the number of bootstrapping
            * xlabel: The x-axis label.
            * legendloc: (default lower right): The location of the plot legend.
            * nSamples: Sample size of the x-axis value.

        + POD-CF: Probability of Detection Curve By Curve-fitting Exponential Function
            * scalingFactor (default 1/1000): The multiplicative factor to scale x-axis data to basis specified by unit.
            * xCategory: The column header of the x-axis data.
            * unit (default r"$\mathrm{kg~CH_4/h}$"): The x-axis data unit.
            * xlimit: The lower and upper limit of the x-axis data specified on the basis of the unit specified.
            * xstep (default 0.0001): The x-axis data interval/step.
            * xScale (default linear): The type of scale applied on the x-axis.
            * yScale (default linear): The type of scale applied on the y-axis.
            * desiredLDLFraction (default 0.9): The y-axis data at which to evaluate probability of detection.
            * Nbootstrap (default 500): An integer representing the number of bootstrapping
            * xlabel: The x-axis label
            * legendloc (default lower right): The location of the plot legend.
            * a0s (default list(np.linspace(0.001, 2, 7))): An array of initial guesses for the curve-fitting parameter (a)
            * b0s (default list(np.linspace(0.1, 10, 7))): An array of initial guesses for the curve-fitting parameter (b)
            * bounds (default ([1e-6, 1e-6], [+np.inf, +np.inf])): Bounds of the curve fitting parameters.
            * nSamples: Sample size of the x-axis value.

        + SingleQuant: Compares controlled release rates with reported emission estimates
            * xCategory: The column header of the x-axis data.
            * yCategory: The column header of the y-axis data.
            * xScalingFactor (default 1/1000): The multiplicative factor to scale x-axis data to basis specified by unit.
            * yScalingFactor (default 1/1000): The multiplicative factor to scale y-axis data to basis specified by unit.
            * quantificationFactor (default 3): The quantification factor applied to the analysis.
            * xunit (default r"$\mathrm{kg~CH_4/h}$"): The x-axis data unit.
            * yunit (default r"$\mathrm{kg~CH_4/h}$"): The y-axis data unit.
            * xScale (default linear): The type of scale applied on the x-axis.
            * yScale (default linear): The type of scale applied on the y-axis.
            * limits: The lower and upper limit of the x-axis and y-axis data specified on the basis of the unit specified.
            * legendloc (default lower left): The location of the plot legend.

        + AggQuant:
            * xunit (default days): The x-axis data unit
            * yunit (default r"$\mathrm{Mg~CH_4}$"): The y-axis data unit.
            * scalingFactor (default 1/1000000): The multiplicative factor to scale x-axis data to basis specified by unit.
            * duration (default daily): The period of data summation/aggregation
            * timeFactor (default 1/3600): A multiplicative factor to convert the time basis of the controlled release and detection report data to seconds.
            * xlimit: The lower and upper limit of the x-axis data specified on the basis of the unit specified.
            * ylimit: The lower and upper limit of the y-axis data specified on the basis of the unit specified.
            * legendloc (default lower right): The location of the plot legend.
    """

    data = {
        "OfflineRecord": {
            "OfflineDateTime": 'StartTime', # This must not be changed. Ensure that no column titled 'StartTime' exists in the file.
            "OnlineDateTime": 'EndTime', # This must not be changed. Ensure that no column titled 'EndTime' exists in the file.
        },
        "PassiveObservation": {
            "StartDateTime": 'StartTime', # This must not be changed. Ensure that no column titled 'StartTime' exists in the file.
            "EndDateTime": 'EndTime', # This must not be changed. Ensure that no column titled 'EndTime' exists in the file.
        },
        "OperatorLog": {
            "StartDateTime": None,
            "EndDateTime": None,
            "NosOfAssetsSurveyed": None,
        },
        "MaintenanceRecord": {
            "StartDateTime": 'StartTime', # Column name for the start time in UTC
            "EndDateTime": 'EndTime', # Column name for the end time in UTC
        },

        "DetectionReportData": {
            "StartDateTime": 'StartTime', # This must not be changed. Ensure that no column titled 'StartTime' exists in the file.
            "EndDateTime": 'EndTime', # This must not be changed. Ensure that no column titled 'EndTime' exists in the file.
            "EmissionRate": "EmissionRate",
            "ReportIDs": "ObservationReportID",
            "EmissionRateLower": None,
            "EmissionRateUpper": None,
            "BoundingXY": None,
            "BoundingZMinimum": None,
            "BoundingZMaximum": None,
            "BoundingName": None,
            "ReceivedTime": None,
            "detection": None,
            "Method": "Above Threshold Across Duration",
            "baselineEmissionRate": 896,
            "ThresholdDuration": 0, # use 0 if not provided
            "ThresholdRateAboveBaseline": 400,
            "listOfVariables2Sum": ['EmissionRate'],
            "listOfVariables2Average": [], # if "detection" is reported, then it must be contained in this list.
            "listOfVariables2ListUnique": ["ObservationReportID"],
            "listOfVariables2Max": None,
            "listOfVariables2Min": None,
            "listOfVariables2ObtainFirst": None, # if "ReceivedTime" is available, then it must be contained in this list. ["ReceivedTime"]
            "listOfVariables2ObtainLast": None,
        },

        "ControlledReleaseData": {
            "StartDateTime": 'UTCStart',
            "EndDateTime": 'UTCEnd',
            "EmissionRate": 'ControlledReleaseRate',
            "EmissionRateLower": None,
            "EmissionRateUpper": None,
            "SourceID": None,
            "BoundingXY": None,
            "BoundingZMinimum": None,
            "BoundingZMaximum": None,
            "BoundingName": None,
            "baselineEmissionRate": 900,
            "listOfVariables2Sum": ['ControlledReleaseRate'],
            "listOfVariables2Average": ["WindSpeed"],
            "listOfVariables2ListUnique": ["tc_EquipmentGroupID"],
            "listOfVariables2Max": None,
            "listOfVariables2Min": None,
            "listOfVariables2ObtainFirst": None,
            "listOfVariables2ObtainLast": None,
            "ASE": { # This is current configured for the set up at METEC, can be modified to suite the needs of the user.,
                "WellPad4": {
                    "BoundingXY": "POLYGON ((40.59552772212062 -105.1396852051655, 40.59601267917348 -105.139683347761, 40.59600301434005 -105.1405294847709, 40.59554745867481 -105.1405484620506, 40.59552772212062 -105.1396852051655))",
                    "BoundingZ": [0.0, 25],
                    "ReferenceCoord": [40.59577187772668, -105.14010720572513]
                                },
                    "WellPad5": {
                    "BoundingXY": "POLYGON ((40.59552898227727 -105.1391338458152, 40.59598255230748 -105.1391436886777, 40.59601744264521 -105.1396897500801, 40.59552202309123 -105.1397587083881, 40.59552898227727 -105.1391338458152))",
                    "BoundingZ": [0.0, 25],
                    "ReferenceCoord": [40.59575771876613, -105.13943630798387]
                               },
                    "WellPad4and5": {
                    "BoundingXY": "POLYGON ((40.59554798291955 -105.1405845434814, 40.59553007903099 -105.1391294662506, 40.59598459722168 -105.139142322344, 40.59601515540727 -105.1405610779183, 40.59554798291955 -105.1405845434814))",
                    "BoundingZ": [0.0, 25],
                    "ReferenceCoord": [40.595768545092895, -105.13985781011445]
                              },
                    "WellPad1": {
                    "BoundingXY": "POLYGON ((40.5950404507719 -105.1394195954563, 40.59514262360668 -105.1394109020159, 40.59513547272269 -105.1395560152809, 40.59503981142013 -105.139563442192, 40.5950404507719 -105.1394195954563))",
                    "BoundingZ": [0.0, 25],
                    "ReferenceCoord": [40.59508965301302, -105.13948669515618]
                              },
                    "WellPad2": {
                    "BoundingXY": "POLYGON ((40.59508012712645 -105.1400239508514, 40.59516024495923 -105.1401361785956, 40.59507806685163 -105.1402325404461, 40.59500615590143 -105.1401391989829, 40.59508012712645 -105.1400239508514))",
                    "BoundingZ": [0.0, 25],
                    "ReferenceCoord": [40.59508186397796, -105.14013136593624]
                              },
                    "WellPad3": {
                    "BoundingXY": "POLYGON ((40.59546032996206 -105.1407263590843, 40.5960509623022 -105.1407190263477, 40.59603621605287 -105.1409414695913, 40.59544591739934 -105.1409382382479, 40.59546032996206 -105.1407263590843))",
                    "BoundingZ": [0.0, 25],
                    "ReferenceCoord": [40.59575074961323, -105.14083127651611]
                               },
                    "DummyPad": {
                        "BoundingXY": None,
                        "BoundingZ": [None, None],
                        "ReferenceCoord": [None, None]
                                }
                    }
                },
        "Plotting": {
            "TTA": {
                "plot": True, # A boolean indicating if the plot should be made.
                "timeFactor": 1/3600,
                "unit": "hours",
                "bins": 0.2, # Bin width (i.e., 0.2 hours increment)
                "xlimit": None,
                "ylimit": None,
                "legendloc": 'upper right'
            },
            "TTD": {
                "plot": True, # A boolean indicating if the plot should be made.
                "timeFactor": 1/3600,
                "unit": "hours",
                "bins": 0.2, # Bin width (i.e., 0.2 hours increment)
                "xlimit": None,
                "ylimit": None,
                "legendloc": 'upper right'
            },
            "ReportDelay": {
                "plot": True, # A boolean indicating if the plot should be made.
                "timeFactor": 1/86400,
                "unit": "days",
                "bins": 0.2, # Bin width (i.e., 0.2 hours increment)
                "xlimit": None,
                "ylimit": None,
                "legendloc": 'upper center'
            },
            "SurveyTime": {
                "plot": True, # A boolean indicating if the plot should be made.
                "timeFactor": 1/3600,
                "unit": "hours",
                "bins": 0.2,
                "xlimit": None,
                "ylimit": None,
                "legendloc": 'upper right'
            },
            "POD-LR": {# plotting POD curve using logistic regression
                "plot": True, # A boolean indicating if the plot should be made.
                "scalingFactor": 1/1000, # 1/1000
                "xCategory": "ControlledReleaseRate",
                "unit": r"$\mathrm{kg~CH_4/h}$",
                "xlimit": (0.001, 100), # for xScale being log: (0.000001, 100000)
                "xScale": "log",
                "yScale": "linear",
                "desiredLDLFraction": 0.9,
                "nSamples": 1000,
                "Nbootstrap": 1000,
                "xlabel": "Release Rate above baseline",
                "legendloc": 'center left'
            },
            "SingleQuant": {
                "plot": True, # A boolean indicating if the plot should be made.
                "xCategory": "ControlledReleaseRate", # Test center release rate.
                "colorBar": None,
                "colorBarLabel": None,
                "xCategoryLower": None,
                "xCategoryUpper": None,
                "yCategory": "EmissionRate", # Reported release rate
                "yCategoryLower": None,
                "yCategoryUpper": None,
                "xScalingFactor": 1, # 1/1000
                "yScalingFactor": 1,
                "quantificationFactor": 2,
                "xunit": r"$\mathrm{g~CH_4/h}$", # r"$\mathrm{kg~CH_4/h}$"
                "yunit": r"$\mathrm{g~CH_4/h}$", # r"$\mathrm{kg~CH_4/h}$"
                "xScale": "log",
                "yScale": "log",
                "limits": (1, 50000), # (0.01, 100),
                "legendloc": 'lower right',
                "minTrueRate": None},

            "AggQuant": {
                "plot": True, # A boolean indicating if the plot should be made.
                "xunit": "days",
                "yunit": r"$\mathrm{kg~CH_4}$",
                "scalingFactor": 1/1000,
                "timeFactor": 1/3600,
                "xlimit": None,
                "ylimit": None,
                "legendloc": 'upper left',
                "IncludeAll": False,
            }

        }
    }

    return data