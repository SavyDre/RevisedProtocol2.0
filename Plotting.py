import time
import matplotlib.pyplot as plt
from FiguresBuilder import LOGISTICREGRESSION
from FiguresBuilder import POD_curve
from FiguresBuilder import densityPlot
from FiguresBuilder import quant_cummulative
from FiguresBuilder import histogram
from FiguresBuilder import scatter_plot
from FiguresBuilder import withAfactor

def plotting(figurefilepath, quantDF, TTD, TTA, SurveyTime, ReportDelay, tc_baselineEmissionRate, singleQuant_xCol,
             singleQuant_xLowerCol, singleQuant_xUpperCol, singleQuant_yCol, singleQuant_yLowerCol,
             singleQuant_yUpperCol, singleQuant_xScalingFact, singleQuant_yScalingFact, CRDF, DRDF, singleQuant_xunit,
             singleQuant_yunit, singleQuant_limit, singleQuant_quantFactor, singleQuant_xScale, singleQuant_yScale,
             singleQuant_legendloc, aggQuant_xlimit, aggQuant_ylimit, aggQuant_xunit, aggQuant_yunit, aggQuant_all,
             aggQuant_yScalingFact, aggQuant_legendloc,  TTD_timeFact, TTD_nBins, TTD_xunit, TTD_xlimit, TTD_ylimit,
             TTD_legendloc, TTA_timeFact, TTA_nBins, TTA_xunit, TTA_xlimit, TTA_ylimit, TTA_legendloc, RD_timeFact,
             RD_nBins, RD_xunit, RD_xlimit, RD_ylimit, RD_legendloc, ST_nBins, ST_xunit, ST_xlimit, ST_ylimit,
             ST_legendloc, ST_timeFact, desiredLDLFraction, Nbootstrap, Agg_classifiedDF, LR_xCol, LR_xScalingFact,
             LR_xunit, LR_xlimit, LR_legendloc, LR_nSamples, LR_xScale, LR_yScale, LR_xLabel, CF_nSamples, CF_xCol,
             CF_xScalingFact, CF_xunit, CF_xlimit, CF_step, CF_legendloc, CF_xScale, CF_yScale, CF_xLabel, bounds,
             a0s, b0s, plot_singlequant, plot_TTD, plot_TTA, plot_RD, plot_ST, plot_LR, plot_CF, plot_aggquant):
    """
        Generate various plots for the analysis.
        Returns:
            None
        """

    if plot_singlequant:
        try:
            print("Plot scatter plot of quantification estimations- with density plots")
            start_time = time.time()
            TPDF = quantDF.loc[quantDF['Classification'] == "1"]
            # Calculate the percentage of estimates within reported emissions estimates within the factor: singleQuant_quantFactor
            percentage = withAfactor(quantDF=quantDF,
                                     factor=singleQuant_quantFactor)
            scatter_plot(data=TPDF,
                         x_column=singleQuant_xCol,
                         y_column=singleQuant_yCol,
                         x_columnLower=singleQuant_xLowerCol,
                         x_columnUpper=singleQuant_xUpperCol,
                         y_columnLower=singleQuant_yLowerCol,
                         y_columnUpper=singleQuant_yUpperCol,
                         xFactor=singleQuant_xScalingFact,
                         yFactor=singleQuant_yScalingFact,
                         edge_color="k",
                         xunit=singleQuant_xunit,
                         yunit=singleQuant_yunit,
                         limit=singleQuant_limit,
                         sColor="green",
                         lColor="red",
                         lw=1,
                         fig=None,
                         figsize=(5, 4),
                         axes=None,
                         slabel="Data Points",
                         factor=(singleQuant_quantFactor, percentage),
                         xlabel="Test Center Rates",
                         ylabel="Reported Estimates",
                         xScale=singleQuant_xScale,
                         yScale=singleQuant_yScale,
                         fontsize=9,
                         fileName="Scatter_QuantificationAccuracyDenP.png",
                         outputFilePath=figurefilepath,
                         legendloc=singleQuant_legendloc,
                         dpi=400)
            end_time = time.time()
            print(f"Plot of scatter plot of quantification estimations with density plots executed in {end_time - start_time:.6f} seconds")
        except Exception as e:
            print(f"Could not plot scatter plot of quantification estimations due to {e}")

    if plot_TTD:
        try:
            print("Plot histogram data for time to detection (TTD)")
            start_time = time.time()
            histogram(data=TTD,
                      unit=TTD_xunit,
                      fig=None,
                      axes=None,
                      ci=(2.5, 97.5),
                      factor=TTD_timeFact,
                      figsize=(4.5, 3.5),
                      bins=TTD_nBins,
                      color='teal',
                      edgecolor='black',
                      lw=1.5,
                      fontsize=9,
                      xlabel="Time to Detection",
                      ylabel="Count",
                      alpha=0.1,
                      xlimit=TTD_xlimit,
                      ylimit=TTD_ylimit,
                      fileName="Hist_TimetoDetection.png",
                      outputFilePath=figurefilepath,
                      legendloc=TTD_legendloc,
                      dpi=400)
            end_time = time.time()
            print(f"Plot of histogram data for time to detection (TTD) executed in {end_time - start_time:.6f} seconds")
        except Exception as e:
            print(f"Could not plot histogram data for time to detection (TTD) due to {e}")

    if plot_TTA:
        try:
            print("Plot histogram data for time to alert (TTA)")
            start_time = time.time()
            histogram(data=TTA,
                      unit=TTA_xunit,
                      fig=None,
                      axes=None,
                      ci=(2.5, 97.5),
                      factor=TTA_timeFact,
                      figsize=(4.5, 3.5),
                      bins=TTA_nBins,
                      color='teal',
                      edgecolor='black',
                      lw=1.5,
                      fontsize=9,
                      xlabel="Time to Alert",
                      ylabel="Count",
                      alpha=0.1,
                      xlimit=TTA_xlimit,
                      ylimit=TTA_ylimit,
                      fileName="Hist_TimetoAlert.png",
                      outputFilePath=figurefilepath,
                      legendloc=TTA_legendloc,
                      dpi=400)
            end_time = time.time()
            print(f"Plot of histogram data for time to alert (TTA) executed in {end_time - start_time:.6f} seconds")
        except Exception as e:
            print(f"Could not plot histogram data for time to alert (TTA) due to {e}")

    if plot_RD:
        try:
            print("Plot histogram data for Reporting time")
            start_time = time.time()
            histogram(data=ReportDelay,
                      unit=RD_xunit,
                      fig=None,
                      axes=None,
                      ci=(2.5, 97.5),
                      figsize=(4.5, 3.5),
                      factor=RD_timeFact,
                      bins=RD_nBins,
                      color='teal',
                      edgecolor='black',
                      lw=1.5,
                      fontsize=9,
                      xlabel="Reporting Delay",
                      ylabel="Count",
                      alpha=0.1,
                      xlimit=RD_xlimit,
                      ylimit=RD_ylimit,
                      fileName="Hist_ReportingDelay.png",
                      outputFilePath=figurefilepath,
                      legendloc=RD_legendloc,
                      dpi=400)
            end_time = time.time()
            print(f"Plot of histogram data for Reporting time executed in {end_time - start_time:.6f} seconds")
        except Exception as e:
            print(f"Could not plot histogram data for Reporting time due to {e}")

    if plot_ST:
        try:
            print("Plot histogram data for survey time")
            start_time = time.time()
            histogram(data=SurveyTime,
                      unit=ST_xunit,
                      fig=None,
                      axes=None,
                      ci=(2.5, 97.5),
                      figsize=(4.5, 3.5),
                      factor=ST_timeFact,
                      bins=ST_nBins,
                      color='teal',
                      edgecolor='black',
                      lw=1.5,
                      fontsize=9,
                      xlabel="Survey Time",
                      ylabel="Count",
                      alpha=0.1,
                      xlimit=ST_xlimit,
                      ylimit=ST_ylimit,
                      fileName="Hist_ReportingDelay.png",
                      outputFilePath=figurefilepath,
                      legendloc=ST_legendloc,
                      dpi=400)
            end_time = time.time()
            print(f"Plot of histogram data for survey time executed in {end_time - start_time:.6f} seconds")
        except Exception as e:
            print(f"Could not plot histogram data for survey time due to {e}")

    if plot_aggquant:
        try:
            print(f"Ploting cummulative emissions sum comparison of test center and reported emissions")
            start_time = time.time()
            quant_cummulative(CRDF=CRDF,
                              DRDF=DRDF,
                              time="Time",
                              cr_emission="tc_total_emission",
                              dr_emission="dr_total_emission",
                              fig=None,
                              axes=None,
                              figsize=(5.5, 3.5),
                              lw=1.5,
                              fontsize=9,
                              xlabel="Date",
                              ylabel="Cumulative Emissions",
                              alpha=0.1,
                              xlimit=aggQuant_xlimit,
                              ylimit=aggQuant_ylimit,
                              fileName="CummulativeQuantAll.png",
                              outputFilePath=figurefilepath,
                              legendloc=aggQuant_legendloc,
                              dpi=400,
                              xunit=aggQuant_xunit,
                              xrotation=45,
                              yunit=aggQuant_yunit,
                              color1="orange",
                              color2="blue",
                              factor=aggQuant_yScalingFact,
                              aggQuant_all=aggQuant_all)
            end_time = time.time()
            print(f"Plot of cummulative emissions sum executed in {end_time - start_time:.6f} seconds")
        except Exception as e:
            print(f"Could not plot cummulative emissions sum comparison of test center and reported emissions due to {e}")

    if plot_LR:
        try:
            print("Plot POD curve using Logistic Regression")
            start_time = time.time()

            LOGISTICREGRESSION(Df=Agg_classifiedDF,
                               filePath=figurefilepath,
                               fileName="LogisticRegressionWithDensity.png",
                               desiredLDLFraction=desiredLDLFraction,
                               Nbootstrap=Nbootstrap,
                               fontsize=9,
                               xCategory=LR_xCol,
                               yCategory='Classification',
                               xlabel=LR_xLabel,
                               xunits=LR_xunit,
                               figsize=(4.5, 3.5),
                               ScaleFactor=LR_xScalingFact,
                               borderpad=0.4,
                               eps=1e-6,
                               xScale=LR_xScale,
                               yScale=LR_yScale,
                               xmax=LR_xlimit[1],
                               digits=1,
                               xmin=LR_xlimit[0],
                               N=LR_nSamples,
                               mColors='b',
                               bColors='#B0C4DE',
                               loc="lower right",
                               baseline=tc_baselineEmissionRate,
                               dpi=400)

            # Adjust layout
            plt.tight_layout()

            end_time = time.time()
            print(f"Plot of POD curve with combined density plot executed in {end_time - start_time:.6f} seconds")
        except Exception as e:
            print(f"Could not plot POD curve using Logistic Regression due to {e}")


    # if plot_CF:
    #     try:
    #         print("Plot POD curve using curve fitting: sigmoid with combined density plot")
    #         start_time = time.time()
    #         fig = plt.figure(figsize=(4.5, 4.5))  # Reduced height since we only have 2 panels
    #         gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.07)  # Only 2 panels now
    #
    #         # Create subplots
    #         ax_top = plt.subplot(gs[0])  # Combined density plot (TP + FN)
    #         ax_main = plt.subplot(gs[1])  # Main POD curve plot
    #
    #         POD_curve(Df=Agg_classifiedDF, outputFilePath=figurefilepath, fileName="CurveFitting_POD_Density.png",
    #                   detectionlimit=desiredLDLFraction, Nbootstrap=Nbootstrap, xScaleFactor=CF_xScalingFact,
    #                   xData=CF_xCol, cData='Classification', xlabel=CF_xLabel, xunit=CF_xunit, alpha=0.1,
    #                   fig=fig, axes=(ax_main, ax_top), digits=2, varPrefix="", legendLOC=CF_legendloc,
    #                   # Now passing 2 axes
    #                   dpi=400, figsize=(4.5, 3.5), fontsize=9, bounds=bounds, step=CF_step, a0s=a0s, b0s=b0s,
    #                   bsColor='#B0C4DE', mColor='darkblue', N=CF_nSamples, xScale=CF_xScale, yScale=CF_yScale,
    #                   xlim=CF_xlimit, baseline=tc_baselineEmissionRate, paperDict=None)
    #
    #         # Adjust layout
    #         plt.tight_layout()
    #
    #         end_time = time.time()
    #         print(f"Plot of POD curve with combined density plot executed in {end_time - start_time:.6f} seconds")
    #     except Exception as e:
    #         print(f"Could not plot POD curve using curve fitting due to {e}")

    return