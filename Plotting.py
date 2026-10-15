"""Dispatch configured performance figures; report individual plot failures."""
import time
import matplotlib.pyplot as plt
import pandas as pd
from FiguresBuilder import LOGISTICREGRESSION
from FiguresBuilder import quant_cummulative
from FiguresBuilder import histogram
from FiguresBuilder import scatter_plot
from FiguresBuilder import scatter
from FiguresBuilder import withAfactor
from Metrics import evaluateRelativeErrors

def plotting(figurefilepath, TTD, TTA, SurveyTime, ReportDelay, tc_baselineEmissionRate, ThresholdRateAboveBaseline,
             singleQuant_xCol, singleQuant_xLowerCol, singleQuant_xUpperCol, singleQuant_yCol, singleQuant_yLowerCol,
             singleQuant_yUpperCol, singleQuant_xScalingFact, singleQuant_yScalingFact, CRDF, DRDF, singleQuant_xunit,
             singleQuant_yunit, singleQuant_limit, singleQuant_quantFactor, singleQuant_xScale, singleQuant_yScale,
             singleQuant_legendloc, singleQuant_cbar, singleQuant_cbar_lab, minTrueRate, aggQuant_xlimit, aggQuant_ylimit,
             aggQuant_xunit, aggQuant_yunit, aggQuant_yScalingFact, aggQuant_legendloc,  TTD_timeFact, TTD_nBins,
             TTD_xunit, TTD_xlimit, TTD_ylimit, TTD_legendloc, TTA_timeFact, TTA_nBins, TTA_xunit, TTA_xlimit,
             TTA_ylimit, TTA_legendloc, RD_timeFact, RD_nBins, RD_xunit, RD_xlimit, RD_ylimit, RD_legendloc,
             ST_nBins, ST_xunit, ST_xlimit, ST_ylimit, ST_legendloc, ST_timeFact, desiredLDLFraction,
             ClassifiedDF, LR_xCol, LR_xScalingFact, LR_xunit, LR_xlimit, LR_legendloc, LR_nSamples, LR_xScale,
             LR_yScale, LR_xLabel, plot_singlequant, plot_TTD, plot_TTA, plot_RD, plot_ST, plot_LR, plot_aggquant,
             random_state=0, LR_Nbootstrap=None):
    """
        Generate various plots for the analysis.
        Returns: None
    """
    tpdf = ClassifiedDF.loc[ClassifiedDF['Classification'] == "TP"]
    fndf = ClassifiedDF.loc[ClassifiedDF['Classification'] == "FN"]
    tndf = ClassifiedDF.loc[ClassifiedDF['Classification'] == "TN"]
    fpdf = ClassifiedDF.loc[ClassifiedDF['Classification'] == "FP"]

    def percentage(count, denominator):
        # An absent reference population has an undefined fraction, not zero.
        return round(count * 100 / denominator, 2) if denominator else float('nan')

    print(f"TPF: {percentage(len(tpdf), len(tpdf) + len(fndf))}")
    print(f"FNF: {percentage(len(fndf), len(tpdf) + len(fndf))}")
    print(f"TNF: {percentage(len(tndf), len(tndf) + len(fpdf))}")
    print(f"FPF: {percentage(len(fpdf), len(fpdf) + len(tpdf))}")

    if plot_singlequant:
        try:
            print("Plot scatter plot of quantification estimations- with density plots")
            start_time = time.time()
            # Calculate the percentage of estimates within reported emissions estimates within the factor: singleQuant_quantFactor
            quantDF = ClassifiedDF.loc[ClassifiedDF["Classification"].isin(["TP", "FN", "FP", "TN"])]
            quantDF = evaluateRelativeErrors(quantDF=quantDF, minTrueRate=minTrueRate,
                                             cr_emission=singleQuant_xCol,
                                             dr_emission=singleQuant_yCol)
            percentage = withAfactor(quantDF=quantDF, factor=singleQuant_quantFactor)
            # Keep rows with numeric relative errors; exclude None and invalid values.
            numeric_errors = pd.to_numeric(quantDF["QuantificationAccuracyRel"], errors="coerce")
            numericDF = quantDF.loc[numeric_errors.notna()].copy()
            print(f"The fraction of estimates within a factor of 2: {withAfactor(quantDF=quantDF, factor=2)}")
            print(f"The empirical 95% CI are {numericDF['QuantificationAccuracyRel'].quantile(0.025) * 100}% and {numericDF['QuantificationAccuracyRel'].quantile(0.975) * 100}%")
            scatter_plot(data=quantDF,
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
                         slabel=None,
                         factor=(singleQuant_quantFactor, percentage),
                         xlabel="Test Center Rates",
                         ylabel="Reported Estimates",
                         xScale=singleQuant_xScale,
                         yScale=singleQuant_yScale,
                         fontsize=11,
                         fileName="Scatter_QuantificationAccuracyDenP.png",
                         outputFilePath=figurefilepath,
                         legendloc=singleQuant_legendloc,
                         dpi=400,
                         color_column=singleQuant_cbar,
                         colorbar_label=singleQuant_cbar_lab,
                         colorbar_ticks=6,
                         cmap="viridis")
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
                      fontsize=11,
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
                      fontsize=11,
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
                      fontsize=11,
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
                      fontsize=11,
                      xlabel="Survey Time",
                      ylabel="Count",
                      alpha=0.1,
                      xlimit=ST_xlimit,
                      ylimit=ST_ylimit,
                      fileName="Hist_SurveyTime.png",
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
                              fontsize=11,
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
                              factor=aggQuant_yScalingFact)
            end_time = time.time()
            print(f"Plot of cummulative emissions sum executed in {end_time - start_time:.6f} seconds")
        except Exception as e:
            print(f"Could not plot cummulative emissions sum comparison of test center and reported emissions due to {e}")

    if plot_LR:
        try:
            print("Plot POD curve using Logistic Regression")
            start_time = time.time()
            LOGISTICREGRESSION(Df=ClassifiedDF,
                               filePath=figurefilepath,
                               fileName="LogisticRegression.png",
                               desiredLDLFraction=desiredLDLFraction,
                               Nbootstrap=LR_Nbootstrap,
                               fontsize=11,
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
                               mColors='b',
                               bColors='#B0C4DE',
                               loc=LR_legendloc,
                               baseline=tc_baselineEmissionRate,
                               ThresholdRateAboveBaseline=ThresholdRateAboveBaseline,
                               dpi=400, random_state=random_state, n_curve_points=LR_nSamples)
            # Adjust layout
            plt.tight_layout()
            end_time = time.time()
            print(f"Plot of POD curve with combined density plot executed in {end_time - start_time:.6f} seconds")
        except Exception as e:
            print(f"Could not plot POD curve using Logistic Regression due to {e}")

        if 'Capture fraction' in ClassifiedDF.columns:
            try:
                print("Plot the release rate vs capture fraction")
                start_time = time.time()
                scatter(df=ClassifiedDF, xCategory=LR_xCol, yCategory="Capture fraction",
                        classificationCategory="Classification", tp_only=True, xlabel="Controlled release rate",
                        xunits="g CH4/h", ylabel="Fraction of Duration Detected ", xScale="linear", xmin=None, xmax=None,
                        figsize=(4.5, 3.5), fontsize=9, marker_size=13, alpha=0.8, ax=None, filePath=figurefilepath)
                end_time = time.time()
                print(f"Plot release rate vs capture fraction executed in {end_time - start_time:.6f} seconds")
            except Exception as e:
                print(f"Could not plot release rate vs capture fraction due to {e}")
    
    return