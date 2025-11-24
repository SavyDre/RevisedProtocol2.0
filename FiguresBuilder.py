import os
import math
import warnings
import itertools
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import statistics
from warnings import warn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from matplotlib.font_manager import FontProperties
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2
from scipy.stats import gaussian_kde
from Classification import DataPairing
from numpy.typing import NDArray

class ROC_curves:
    """
    A class to generate and visualize ROC (Receiver Operating Characteristic) curves
    and calculate AUC (Area Under the Curve).

    Attributes:
    - fileName (str): The name of the file to save the ROC plot.
    - outputFilePath (str): The directory to save the ROC plot.
    - figsize (tuple): The size of the figure (width, height).
    - xlimit (tuple): The x-axis limits for the plot.
    - ylimit (tuple): The y-axis limits for the plot.
    - alpha (float): Transparency of the plot line.
    - lw (float): Line width of the plot.
    - ls (str): Line style for the plot.
    - dpi (int): Specifies the resolution of the saved image, measured in dots per inch
    - fontsize (int): Font size for labels and legend.
    - color (str): Line color of the ROC curve.
    - legendloc (str): Location of the legend on the plot.
    - fig, axes (matplotlib objects): Custom figure and axes for plotting.
    - marker_size (int): Size of scatterplot markers.
    - c (str): Color of scatterplot markers.
    - xlabel, ylabel (str): Labels for the x-axis and y-axis.
    """

    def __init__(self, fileName, outputFilePath, figsize, xlimit=(0.0, 1.0), ylimit=(0.0, 1.05), alpha=0.1, lw=1, ls="-",
                 fontsize=10, color='orange', legendloc='upper left', fig=None, axes=None, marker_size=15, c='black',
                 dpi=400, xlabel="Model's False Positive Rate (FPR)", ylabel="Model's True Positive Rate (TPR)"):
        """
        Initializes the ROC_curves class with configuration parameters for the plot.

        Parameters:
        - fileName (str): The name of the file to save the plot.
        - outputFilePath (str): Directory to save the plot.
        - figsize (tuple): Figure size (width, height).
        - xlimit (tuple): X-axis limits (default: (0.0, 1.0)).
        - ylimit (tuple): Y-axis limits (default: (0.0, 1.05)).
        - alpha (float): Transparency of the line (default: 0.1).
        - lw (float): Line width (default: 1).
        - ls (str): Line style (default: '-').
        - dpi (int): Specifies the resolution of the saved image, measured in dots per inch
        - fontsize (int): Font size for labels and legend (default: 10).
        - color (str): Line color (default: 'orange').
        - legendloc (str): Legend location (default: 'upper left').
        - fig, axes (matplotlib objects): Custom figure and axes for plotting.
        - marker_size (int): Size of markers for scatter points.
        - c (str): Color of scatter points.
        - xlabel, ylabel (str): Labels for the x-axis and y-axis.
        """
        self.fileName = fileName
        self.outputFilePath = outputFilePath
        self.figsize = figsize
        self.xlimit = xlimit
        self.ylimit = ylimit
        self.alpha = alpha
        self.lw = lw
        self.ls = ls
        self.fontsize = fontsize
        self.color = color
        self.legendloc = legendloc
        self.dpi = dpi
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fig = fig
        self.axes = axes
        self.marker_size = marker_size
        self.c = c

    def curvesUsingTrueAndPredictedValues(self, y_true, y_score, pos_label=1):
        """
        Plots the ROC curve and calculates AUC from true labels and predicted scores.

        Parameters:
        - y_true (list or np.array): True binary labels (0 or 1).
        - y_score (list or np.array): Predicted probabilities or scores (0 to 1).
        - pos_label (int): The positive class label (default: 1).

        Returns:
        - ROC_ParametersDF (pd.DataFrame): DataFrame with thresholds, TPR, and FPR.
        - fig (matplotlib Figure): The generated plot figure.
        - ax (matplotlib Axes): The generated plot axes.
        """
        # Setup figure and axes
        if self.fig is None:
            fig, ax = plt.subplots(figsize=self.figsize) if self.figsize else plt.subplots()
        else:
            ax = self.axes
            fig = self.fig

        # Calculate ROC curve metrics
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score, pos_label=pos_label)
        ROC_ParametersDF = pd.DataFrame({"Threshold": thresholds, "TPR": tpr, "FPR": fpr})

        # Calculate AUC
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve and scatter points
        ax.scatter(fpr, tpr, s=self.marker_size, c=self.c, edgecolors='black', marker='o')
        ax.plot(fpr, tpr, color=self.color, lw=self.lw, ls=self.ls, alpha=self.alpha, label=f"AUC: {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5)  # Diagonal line

        # Set plot labels and limits
        ax.set_xlim(self.xlimit)
        ax.set_ylim(self.ylimit)
        ax.set_xlabel(self.xlabel, fontsize=self.fontsize)
        ax.set_ylabel(self.ylabel, fontsize=self.fontsize)

        # Add legend
        fontP = FontProperties()
        fontP.set_size(self.fontsize)
        leg = ax.legend(loc=self.legendloc, prop=fontP)
        leg.get_frame().set_edgecolor('black')
        fig.tight_layout()

        # Save the plot
        if self.fileName:
            path = os.path.join(self.outputFilePath or os.getcwd(), self.fileName)
            print(f"Saving ROC curve to: {path}")
            plt.savefig(path, dpi=self.dpi)

        return ROC_ParametersDF, roc_auc, fig, ax

    def curvesUsingTPRandFPR(self, tpr, fpr, thresholds):
        """
        Plots the ROC curve and calculates AUC from precomputed TPR, FPR, and thresholds.

        Parameters:
        - tpr (list or np.array): True Positive Rates.
        - fpr (list or np.array): False Positive Rates.
        - thresholds (list or np.array): Decision thresholds used in the analysis.

        Returns:
        - ROC_ParametersDF (pd.DataFrame): DataFrame with thresholds, TPR, and FPR.
        - fig (matplotlib Figure): The generated plot figure.
        - ax (matplotlib Axes): The generated plot axes.
        """
        # Setup figure and axes
        if self.fig is None:
            fig, ax = plt.subplots(figsize=self.figsize) if self.figsize else plt.subplots()
        else:
            ax = self.axes
            fig = self.fig

        # Create DataFrame and calculate AUC
        ROC_ParametersDF = pd.DataFrame({"Threshold": thresholds, "TPR": tpr, "FPR": fpr})
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve and scatter points
        ax.scatter(fpr, tpr, s=self.marker_size, c=self.c, edgecolors='black', marker='o')
        ax.plot(fpr, tpr, color=self.color, lw=self.lw, ls=self.ls, alpha=self.alpha, label=f"AUC: {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5)  # Diagonal line

        # Set plot labels and limits
        ax.set_xlim(self.xlimit)
        ax.set_ylim(self.ylimit)
        ax.set_xlabel(self.xlabel, fontsize=self.fontsize)
        ax.set_ylabel(self.ylabel, fontsize=self.fontsize)

        # Add legend
        fontP = FontProperties()
        fontP.set_size(self.fontsize)
        leg = ax.legend(loc=self.legendloc, prop=fontP)
        leg.get_frame().set_edgecolor('black')
        fig.tight_layout()

        # Save the plot
        if self.fileName:
            path = os.path.join(self.outputFilePath or os.getcwd(), self.fileName)
            print(f"Saving ROC curve to: {path}")
            plt.savefig(path, dpi=self.dpi)

        return ROC_ParametersDF, roc_auc, fig, ax


def withAfactor(quantDF, factor):
    """
    Calculates the percentage of true positives (TPs) that fall within a given multiplicative factor.

    Parameters:
    - quantDF (pd.DataFrame): The DataFrame containing paired quantification data.
    - factor (float): The multiplicative factor threshold.

    Returns:
    - int: Percentage of TPs within the specified factor.
    """
    # Filter DataFrame to include only 1's
    TPDF = quantDF.loc[quantDF["Classification"] == "1"]

    # Further filter TPs to those within the specified factor range
    withinAfactorDF = TPDF.loc[(TPDF["QuantificationFactors"] >= -factor) & (TPDF["QuantificationFactors"] <= factor)]

    # Calculate the percentage of qualifying TPs
    percentage = int((len(withinAfactorDF) * 100) / len(TPDF))

    return percentage

def scatter_plot(data, x_column, x_columnLower, x_columnUpper, y_column, y_columnLower, y_columnUpper, xFactor=1,
                 yFactor=1, edge_color="k", xunit="", yunit="", limit=None, sColor="blue", lColor="red",
                 rColor="green", lw=1.5, fig=None, figsize=None, axes=None, slabel="Data", factor=None,
                 xlabel="", ylabel="", xScale="log", yScale="log", fontsize=10, fileName=None, outputFilePath=None,
                 legendloc='upper left', dpi=400):
    """
    Creates a scatter plot with an optional linear regression line, R^2 calculation, 1:1 line, and bounded factors.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing data to plot.
    - x_column (str): Column name for x-axis values.
    - x_columnLower (str): Column name for the lower x-axis values.
    - x_columnUpper (str): Column name for the upper x-axis values.
    - y_column (str): Column name for y-axis values.
    - y_columnLower (str): Column name for the lower y-axis values.
    - y_columnUpper (str): Column name for the upper y-axis values.
    - xFactor (float): Scaling factor for x-axis values (default: 1).
    - yFactor (float): Scaling factor for y-axis values (default: 1).
    - edge_color (str): Edge color for scatter points.
    - xunit (str): Unit label for the x-axis.
    - yunit (str): Unit label for the y-axis.
    - limit (tuple): Range for x and y limits (default: None, auto-calculated).
    - marker_shape (str): Marker shape for scatter plot.
    - marker_size (int): Size of the scatter plot markers.
    - sColor (str): Color of scatter points.
    - lColor (str): Color for the 1:1 line.
    - rColor (str): Color for the regression line.
    - lw (int): Line width for the lines.
    - fig (matplotlib.figure.Figure): Existing figure to use (default: None, creates a new one).
    - figsize (tuple): Size of the figure in inches (width, height).
    - axes (matplotlib.axes.Axes): Existing axes to use (default: None, creates a new one).
    - slabel (str): Label for scatter plot legend.
    - factor (tuple): factor[0] (int) = quantification factor; factor[1] (float) = quantification relative error
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - xScale (str): Scale for the x-axis ('linear', 'log', etc.).
    - yScale (str): Scale for the y-axis ('linear', 'log', etc.).
    - fontsize (int): Font size for labels and legend.
    - fileName (str): File name for saving the plot (default: None, does not save).
    - outputFilePath (str): Directory path for saving the file (default: None, uses current directory).
    - legendloc (str): Location of the legend in the plot.
    - dpi (int): Resolution in dots per inch for saved figure.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object.
    - ax (matplotlib.axes.Axes): The axes object.
    """
    # Setup figure and axes
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize) if figsize else plt.subplots()
    else:
        ax = axes if axes is not None else plt.gca()

    # Drop rows with NA values
    data = data.dropna(axis=0, how='any', subset=[x_column, y_column], inplace=False)

    # Scale data by factors: xFactor and yFactor.
    xVal = data[x_column].mul(xFactor)  # Scale x-values by xFactor
    yVal = data[y_column].mul(yFactor)  # Scale y-values by yFactor
    xerror_bars = None # Initialization
    yerror_bars = None  # Initialization

    # Scale data for the lower and upper bounds if they are provided.
    if (x_columnLower is not None) and (x_columnUpper is not None) and (x_columnLower in data.columns) and (x_columnUpper in data.columns):
        xVal_lower = data[x_columnLower].mul(xFactor)
        xVal_upper = data[x_columnUpper].mul(xFactor)
        xlower_errors = np.abs(xVal.to_numpy() - xVal_lower.to_numpy())
        xupper_errors = np.abs(xVal_upper.to_numpy() - xVal.to_numpy())
        xerror_bars = [xlower_errors, xupper_errors]
    if (y_columnLower is not None) and (y_columnUpper is not None) and (y_columnLower in data.columns) and (y_columnUpper in data.columns):
        yVal_lower = data[y_columnLower].mul(yFactor)
        yVal_upper = data[y_columnUpper].mul(yFactor)
        ylower_errors = np.abs(yVal.to_numpy() - yVal_lower.to_numpy())
        yupper_errors = np.abs(yVal_upper.to_numpy() - yVal.to_numpy())
        yerror_bars = [ylower_errors, yupper_errors]

    # Plot error bars
    ax.errorbar(x=xVal, y=yVal, yerr=yerror_bars, xerr=xerror_bars, fmt='o', color=sColor, ecolor=edge_color,
                elinewidth=0.5, capsize=5, mew=0.5, markerfacecolor='none', label=slabel)

    # Fit a linear regression model to calculate R^2
    X = np.array(xVal).reshape(-1, 1)  # Extract x-values as 2D array
    y = yVal.values  # Extract y-values
    model = LinearRegression().fit(X=X, y=y)  # Fit linear regression model
    r_squared = model.score(X=X, y=y)  # Calculate R^2 score
    y_pred = model.predict(np.array(xVal).reshape(-1, 1))  # Predict y-values using the model
    ax.plot(xVal, y_pred, ls=":", color=rColor, linewidth=lw, label=f"$R^2 = {r_squared:.2f}$")  # Plot regression line

    # Plot 1:1 line
    if limit is None:
        limit = (0, np.max([xVal.max(), yVal.max()]))  # Calculate limit if not provided
    ax.plot(limit, limit, ls="-", color=lColor, linewidth=lw, label="1:1 Line")  # 1:1 reference line

    # Plot bounds with factors
    if factor is not None:
        ax.plot(limit, [lim / factor[0] for lim in limit], ls='--', color='black', linewidth=lw,
                label=f"Factor of {factor[0]} = {factor[1]}%")  # Lower bound line
        ax.plot(limit, [lim * factor[0] for lim in limit], ls='--', color='black', linewidth=lw)  # Upper bound line

    # Set axis scale
    ax.set_xscale(xScale)
    ax.set_yscale(yScale)

    # Set limits
    ax.set_xlim(limit)
    ax.set_ylim(limit)

    # Add axis labels
    ax.set_xlabel(xlabel + " (" + xunit + ")", fontsize=fontsize)
    ax.set_ylabel(ylabel + " (" + yunit + ")", fontsize=fontsize)

    # Add legend
    fontP = FontProperties()
    fontP.set_size(fontsize)
    leg = ax.legend(loc=legendloc, prop=fontP)
    leg.get_frame().set_edgecolor('black')  # Add a border to legend box
    fig.tight_layout()

    # Save the plot
    if fileName:
        # Determine file save path
        path = os.path.join(outputFilePath or os.getcwd(), fileName)
        print(f"Saving the scatter plot to: {path}")
        # Save figure with specified dpi
        plt.savefig(path, dpi=dpi)

    return fig, ax

def histogram(data, unit="", fig=None, axes=None, ci=(2.5, 97.5), figsize=(3.54331, 3.54331/1.5), bins=30,
              color='skyblue', edgecolor='black', lw=1.5, fontsize=10, xlabel="", ylabel="", alpha=0.1, xlimit=None,
              ylimit=None, fileName=None, outputFilePath=None, legendloc='upper left', factor=1, dpi=400):
    """
    Plots a histogram with optional confidence interval (CI) lines and mean.

    Parameters:
    - data (array-like): The data to be plotted.
    - unit (str, optional): Unit of the x-axis variable (displayed in axis label).
    - fig (matplotlib.figure.Figure, optional): Existing figure object.
    - axes (matplotlib.axes.Axes, optional): Existing axes object.
    - ci (tuple, optional): Confidence interval (default: 2.5th and 97.5th percentiles).
    - figsize (tuple, optional): Figure size in inches (default: (3.54331, 3.54331/1.5)).
    - bins (int, optional): Number of histogram bins (default: 30).
    - color (str, optional): Color of the histogram bars (default: 'skyblue').
    - edgecolor (str, optional): Color of the bar edges (default: 'black').
    - lw (float, optional): Line width for CI and mean lines (default: 1.5).
    - fontsize (int, optional): Font size for labels and legend (default: 10).
    - xlabel (str, optional): Label for the x-axis.
    - ylabel (str, optional): Label for the y-axis.
    - alpha (float, optional): Transparency of the histogram bars (default: 0.1).
    - xlimit (tuple, optional): Limit for the x-axis (default: None).
    - ylimit (tuple, optional): Limit for the y-axis (default: None).
    - fileName (str, optional): Name of the file to save the plot (default: None).
    - outputFilePath (str, optional): Directory to save the plot (default: None).
    - legendloc (str, optional): Location of the legend (default: 'upper left').
    - factor (int, float): A factor used to scale the data
    - dpi (int, optional): Resolution of the saved figure (default: 400).

    Returns:
    - fig (matplotlib.figure.Figure): The figure object of the plot.
    - ax (matplotlib.axes.Axes): The axes object of the plot.
    """

    if (np.array(data).size == 0) or (np.all(np.isnan(np.array(data)))):
        return None

    # Setup figure and axes
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize) if figsize else plt.subplots()
    else:
        ax = axes if axes is not None else plt.gca()

    # Scale the data by factor: factor
    data = factor * np.array(data)

    try:
        # Calculate confidence interval (CI) and mean
        ci_lower = np.percentile(data, ci[0])  # Lower bound of CI
        mean_val = np.mean(data, dtype=np.float64)              # Mean value
        ci_upper = np.percentile(data, ci[1])  # Upper bound of CI
        vline_positions = [ci_lower, mean_val, ci_upper]

        # Plot CI and mean lines
        linestyle = ["--", ":", "--"]
        # Labels for the vertical lines
        labels = [f"95% CI: [{round(ci_lower, 2)}, {round(ci_upper, 2)}]", f"Mean: {round(mean_val, 2)}", None]
        for position, ls, label in zip(vline_positions, linestyle, labels):
            ax.axvline(x=position, color='red', linewidth=lw, linestyle=ls, label=label)
        # Plot the histogram
        ax.hist(data, bins=bins, color=color, edgecolor=edgecolor, label="Data")

        # Set axis labels
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlabel(f'{xlabel} ({unit})', fontsize=fontsize)

        # Set axis limits if provided
        if xlimit:
            ax.set_xlim(xlimit)
        if ylimit:
            ax.set_ylim(ylimit)

        # Set grid alpha
        ax.grid(which='major', alpha=alpha)

        # Add legend
        fontP = FontProperties()
        fontP.set_size(fontsize)
        leg = ax.legend(loc=legendloc, prop=fontP)
        leg.get_frame().set_edgecolor('black')  # Add a border to the legend box
        fig.tight_layout()

        # Save the plot if fileName is specified
        if fileName:
            path = os.path.join(outputFilePath or os.getcwd(), fileName)  # Determine file save path
            print(f"Saving the histogram to: {path}")
            plt.savefig(path, dpi=dpi)  # Save figure with specified dpi

        return fig, ax

    except Exception as e:
        print(f"Could not build histogram due to exception: {e}")
        return None

def quant_cummulative(CRDF, DRDF, time, cr_emission, dr_emission, fig=None, axes=None, figsize=(3.54331, 3.54331/1.5),
                      factor=1, lw=1.5, fontsize=10, xlabel="Date", ylabel="Cumulative Emissions", alpha=0.1,
                      xlimit=None, ylimit=None, fileName=None, outputFilePath=None, legendloc='upper left', dpi=400,
                      xunit="Days", yunit=r"$\mathrm{kg~CH_4}$", color1="blue", color2="orange", xrotation=45,
                      aggQuant_all=False):
    """
    Plot cumulative emissions over time for two data sources and save the result.

    Parameters:
    - CRDF: pandas DataFrame containing time and emission data for controlled releases.
    - DRDF: pandas DataFrame containing time and emission data for detection reports.
    - time: Column name for the time variable.
    - cr_emission: Column name for test center emissions.
    - dr_emission: Column name for reported emissions.
    - fig, axes: Existing figure and axes objects (optional).
    - figsize: Tuple specifying the figure size.
    - factor: float or integer for scaling variables
    - lw: Line width for the plot.
    - fontsize: Font size for labels and legend.
    - xlabel, ylabel: Labels for the x and y axes.
    - alpha: Opacity for the lines.
    - xlimit, ylimit: Limits for x and y axes.
    - fileName: File name for saving the plot (optional).
    - outputFilePath: Directory path for saving the plot (optional).
    - legendloc: Location of the legend.
    - dpi: Resolution for saving the plot.
    - xunit, yunit: Units for the x and y axes.
    - color1, color2: Colors for the two emission lines.
    - xrotation: The orientation of the x-ticks label
    - aggQuant_all: Indicates if to include all data or exclude flagged data

    Returns:
    - fig, ax: The figure and axis objects for further customization.
    """
    # Setup figure and axes
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize) if figsize else plt.subplots()
    else:
        ax = axes if axes is not None else plt.gca()

    # Include or exclude data
    if not aggQuant_all:
        CRDF = CRDF.loc[~CRDF["tc_Classification"].isin(["[Maintenance Record: Data Excluded]", "[Offline Record: Data Excluded]"])]
        DRDF = DRDF.loc[~DRDF["dr_Classification"].isin(["[Maintenance Record: Data Excluded]", "[Offline Record: Data Excluded]"])]

    # Preparing data
    CRDF[time] = pd.to_datetime(CRDF[time], errors="coerce")
    DRDF[time] = pd.to_datetime(DRDF[time], errors="coerce")
    CRDF = CRDF.sort_values(by=time)
    DRDF = DRDF.sort_values(by=time)
    CRDF = CRDF.dropna(subset=[time, cr_emission])
    DRDF = DRDF.dropna(subset=[time, dr_emission])

    # Scale the variables by factor: factor
    CRDF[cr_emission] = CRDF[cr_emission].apply(lambda x: x * factor)
    DRDF[dr_emission] = DRDF[dr_emission].apply(lambda x: x * factor)

    # Compute cumulative sums for the two emission sources
    tc_emission_cummulative = CRDF[cr_emission].cumsum()
    dr_emission_cummulative = DRDF[dr_emission].cumsum()

    # Calculate totals for the legend
    tc_total = tc_emission_cummulative.iloc[-1]
    dr_total = dr_emission_cummulative.iloc[-1]

    # Calculate percent difference
    percent_difference = ((dr_total - tc_total) / tc_total) * 100

    # Plot cumulative emissions
    ax.plot(CRDF[time].tolist(), tc_emission_cummulative, label=f"Test center total: {tc_total:.1f} {yunit}",
            linewidth=lw, linestyle="--", color=color1)
    ax.plot(DRDF[time].tolist(), dr_emission_cummulative, label=f"Estimated total: {dr_total:.1f} {yunit}",
            linewidth=lw, linestyle="--", color=color2)

    # Set axis limits if specified
    if xlimit:
        ax.set_xlim(xlimit)
    if ylimit:
        ax.set_ylim(ylimit)

    # Add axis labels
    ax.set_xlabel(f"{xlabel} ({xunit})", fontsize=fontsize)
    ax.set_ylabel(f"{ylabel} ({yunit})", fontsize=fontsize)

    # Set xtick rotation angle
    ax.tick_params(axis='x', rotation=xrotation)

    # Add legend
    fontP = FontProperties()
    fontP.set_size(fontsize)
    leg = ax.legend(loc=legendloc, prop=fontP, title=f'Percent Difference: {percent_difference:.2f}%')
    leg.get_frame().set_edgecolor('black')  # Add a border to the legend box
    fig.tight_layout()

    # Set grid alpha
    ax.grid(which='major', alpha=alpha)

    # Save the plot if a file name is provided
    if fileName:
        path = os.path.join(outputFilePath or os.getcwd(), fileName)  # Determine file save path
        print(f"Saving the plot to: {path}")
        plt.savefig(path, dpi=dpi)  # Save the figure

    return fig, ax


# Define sigmoid model
def mSigmoidModel(x, a, b):
    return 1 - np.exp(-a * (x ** b))

def InversemSigmoidModel(y, a, b):
    return (-(np.log(1 - y)) / a) ** (1 / b)

# Helper to round up to nearest step
def round_up_to_nearest(x, step):
    return np.ceil(x / step) * step

def perform_curve_fit(releaserate, classification, p0, bounds):
    """
    Perform curve fitting using a sigmoid model.

    Parameters:
    - releaserate (array-like): X-axis data (release rates).
    - classification (array-like): Y-axis data (classifications: 1 for TP, 0 for FN).
    - p0 (list): Initial guess for the parameters [a, b].
    - bounds (tuple): Bounds for curve fitting as ([lower_a, lower_b], [upper_a, upper_b]).

    Returns:
    - params (list): Fitted parameters [a, b].
    - covariance (array): Covariance matrix of the fit.
    """
    try:
        ## Attempt to fit the sigmoid model to the data
        params, covariance = curve_fit(mSigmoidModel, xdata=releaserate, ydata=classification, p0=p0, bounds=bounds,
                                       method="trf", maxfev=10000)
        return params, covariance

    except Exception as e:
        ## Handle errors during curve fitting
        warn(f"Curve fitting failed: {e}")
        return [np.nan, np.nan], np.full((2, 2), np.nan)


def optimizeCurve(a_values, b_values, bounds, releaserate, classification):
    """
    Optimize the curve-fitting parameters for the sigmoid model.

    Parameters:
    - a_values (iterable): Range of initial guesses for parameter 'a'.
    - b_values (iterable): Range of initial guesses for parameter 'b'.
    - bounds (tuple): Bounds for curve fitting as ([lower_a, lower_b], [upper_a, upper_b]).
    - releaserate (array-like): X-axis data (release rates).
    - classification (array-like): Y-axis data (classifications: 1 for TP, 0 for FN).

    Returns:
    - best_params (list): Optimal parameters [a, b].
    - best_covariance (array): Covariance matrix of the best fit.
    """
    best_params = None
    best_covariance = None
    best_residual = np.inf  # Initialize best residual sum of squares (RSS) to infinity

    ## Iterate over all combinations of a_values and b_values
    for a, b in itertools.product(a_values, b_values):
        p0 = [a, b]  # Initial parameter guess
        params, covariance = perform_curve_fit(releaserate=releaserate, classification=classification, p0=p0,
                                               bounds=bounds)
        ## Skip invalid parameter sets
        if np.any(np.isnan(params)) or params[0] <= 0 or params[1] <= 0:
            continue

        ## Calculate residual sum of squares (RSS)
        classification_fitted = mSigmoidModel(releaserate, params[0], params[1])
        # residual = [a - b for a, b in zip(classification_fitted, classification)]
        # rss = np.sum([e ** 2 for e in residual])
        residuals = classification_fitted - classification
        rss = np.sum(residuals ** 2)

        ## Update the best parameters if current RSS is lower
        if rss < best_residual:
            best_residual = rss
            best_params = params
            best_covariance = covariance

    return best_params, best_covariance


def POD_curve(Df, outputFilePath=None, fileName=None, detectionlimit=0.90, Nbootstrap=500, xScaleFactor=1,
              xData='tc_EPBFE', cData='tc_Classification', xlabel="", xunit='slpm', alpha=0.1, fig=None,
              axes=None, digits=3, varPrefix="", legendLOC='lower right', dpi=400, figsize=(8, 6), fontsize=18,
              bounds=([0, 0], [np.inf, np.inf]), step=0.5, a0s=None, b0s=None, bsColor='#B0C4DE', N=1000,
              mColor='darkblue', xScale="linear", yScale="linear", xlim=None, paperDict=None, baseline=0):

    """
        Generate and plot the Probability of Detection (POD) curve.

        Parameters:
            Df (DataFrame): Input data containing release rate and classification data.
            outputFilePath (str): Directory to save the plot (optional).
            fileName (str): Name of the file to save the plot (optional).
            detectionlimit (float): Threshold for detection (e.g., 0.90 for 90% detection).
            Nbootstrap (int): Number of bootstrap iterations for uncertainty estimation.
            xScaleFactor (float): Scaling factor applied to the x-axis data.
            xData (str): Column name for release rate data.
            cData (str): Column name for classification data.
            xlabel (str): Label for the x-axis (optional).
            xunit (str): Unit for x-axis data (optional, e.g., 'slpm').
            alpha (float): Transparency for gridlines.
            baseline (float): baseline emission rate by the test center
            ... (other parameters for customization and fitting).

        Returns:
            fig (matplotlib.figure.Figure): The created figure object.
            ax (matplotlib.axes.Axes): The axes of the plot.
            paperDict (dict): Dictionary updated with detection limits for documentation.
    """

    # Set up the plotting environment
    plt.rcParams.update({'font.size': fontsize})
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = axes

    # Preprocess the data
    print("Preprocessing data...")
    CRDF = Df.loc[Df[cData].isin(['TP', 'FN'])].dropna(subset=[xData])
    CRDF['Detection'] = CRDF[cData].map({'TP': 1, 'FN': 0})
    CRDF[xData] = CRDF[xData].apply(lambda x: (x - baseline) * xScaleFactor)
    CRDF = CRDF.loc[CRDF[xData] >= 0]  # Extract only positive release rates

    # Determine x-axis limits
    xMax = round_up_to_nearest(CRDF[xData].max(), step)
    xlim = xlim or [0, xMax]

    # Extract releaserate and classification arrays
    releaserate = np.array(CRDF[xData].values)
    classification = np.array(CRDF['Detection'].values)

    # Perform curve fitting
    print("Performing curve fitting...")
    param, covariance = optimizeCurve(a_values=a0s, b_values=b0s, bounds=bounds, releaserate=releaserate,
                                      classification=classification)

    # Perform bootstrap analysis
    print("Performing bootstrap analysis...")

    bootstrap_results = [optimizeCurve(a0s, b0s, bounds, *resample(releaserate, classification))[0]
                         for _ in range(Nbootstrap) if
                         not np.isnan(optimizeCurve(a0s, b0s, bounds, *resample(releaserate, classification))[0]).any()]

    # Preparing labels
    percentage = "{:.0%}".format(detectionlimit)
    denominator = '1 - e^{{%5.2f*x}^{%5.2f}}' % (-1 * param[0], param[1])
    label = r'curve: ${%s}$' % (denominator)

    print("Plotting results...")
    # Main curves
    x_values = np.linspace(0, xlim[1], N)
    ax.plot(x_values, mSigmoidModel(x_values, param[0], param[1]), color=mColor, linewidth=1, label=label, zorder=3)
    ax.scatter(releaserate, classification, label='TP (1) & FN (0)', s=5, facecolors='none', zorder=3, edgecolors='k')

    # Plot the bootstraps
    for params in bootstrap_results:
        ax.plot(x_values, mSigmoidModel(x_values, *params), color=bsColor, alpha=alpha)

    # Convert the results to a numpy array for easier manipulation
    bootstrap_results = np.array(bootstrap_results)

    # Calculate detection limits
    DL90s = InversemSigmoidModel(detectionlimit, bootstrap_results[:, 0], bootstrap_results[:, 1])
    DL5bound = round(np.percentile(DL90s, 5), digits) if np.percentile(DL90s, 5) >= 0 else 0
    DL95bound = round(np.percentile(DL90s, 95), digits) if np.percentile(DL90s, 95) > 0 else "INF"
    DL90 = round(InversemSigmoidModel(detectionlimit, param[0], param[1]), digits) if InversemSigmoidModel(detectionlimit, param[0], param[1]) > 0 else "NA"

    # Plot the DL90 in the figure.
    try:
        ax.axvline(DL90, color='k', linestyle=':', lw=0.8, zorder=2,
                   label=f"{percentage} POD : {DL90:.{digits}f} [{DL5bound:.{digits}f}, {DL95bound:.{digits}f}]")
        ax.axhline(detectionlimit, color='k', linestyle=':', lw=0.8)
    except Exception as e:
        print(f'Could not plot the ldl due to exception: {e}')
        ax.axhline(detectionlimit, color='k', linestyle=':', lw=0.8, label=f"{percentage} POD : out of tested range")

    # Update the paperDict
    if paperDict is not None:
        paperDict.update({f"{varPrefix}ldl": DL90, f"{varPrefix}lowerLDL": DL5bound, f"{varPrefix}upperLDL": DL95bound})

    # Final plot formatting
    ax.set_xscale(xScale)
    ax.set_yscale(yScale)
    ax.set_xlim(xlim)
    ax.set_ylim([0, 1])
    ax.set_xlabel(f"{xlabel} ({xunit})", fontsize=fontsize)
    ax.set_ylabel("Probability of Detection (POD)", fontsize=fontsize)

    # Set grid alpha
    ax.grid(which='major', alpha=alpha)

    # Add legend
    fontP = FontProperties()
    fontP.set_size(fontsize)
    leg = ax.legend(loc=legendLOC, prop=fontP)
    leg.get_frame().set_edgecolor('black')  # Add a border to the legend box
    fig.tight_layout()

    # Step 10: Save the plot if requested
    if fileName:
        path = os.path.join(outputFilePath or os.getcwd(), fileName)
        print(f"Saving POD curve by fitting to: {path}")
        plt.savefig(path, dpi=dpi)

    return fig, ax, paperDict


def filteringBootstraps(Bmc, beta_hat, ci=0.95):
    """
    Filters bootstrap parameter estimates (beta0, beta1) by removing outliers
    using Mahalanobis distance and a chi-square cutoff.

    Parameters
        - Bmc : 2 x N array of bootstrap parameter estimates, where:
        - row 0 = beta0 values
        - row 1 = beta1 values

    beta_hat (list/tuple) : parameters of the main curve
    confidence : Confidence level for the chi-square cutoff (default = 0.95)

    Returns
        - keep : Boolean mask (length N) showing which bootstrap samples are kept.
    """

    # Extract beta0 and beta1 from bootstrap samples
    beta0 = Bmc[0, :]
    beta1 = Bmc[1, :]

    # Compute the standard deviation of each parameter
    s0, s1 = beta0.std(ddof=1), beta1.std(ddof=1)

    # Standardize each parameter (z-score like scaling)
    z0 = (beta0 - beta_hat[0]) / s0
    z1 = (beta1 - beta_hat[1]) / s1

    # Compute squared Mahalanobis-like distance
    D2 = z0**2 + z1**2   # because we assume no correlation between beta0 and beta1

    # Compute chi-square cutoff for 2 parameters (degrees of freedom = 2)
    cutoff = chi2.ppf(ci, df=2)

    # Keep samples within the cutoff (inside the confidence ellipse)
    keep = D2 <= cutoff

    return keep

def logisticRegression(coeff, x):  # Logistic Function
    """
        Logistic Function:
        Computes the logistic function for a given input value `x` based on coefficients `B0` and `B1`.

        Parameters:
            - B0 (float): Intercept of the logistic regression.
            - B1 (float): Coefficient (slope) for the independent variable.
            - x (float): Independent variable value.

        Returns:
            - y (float): Probability value between 0 and 1 calculated using the logistic function.
                         If there's an error (e.g., overflow), it returns 0 as a fallback.

        Formula:
            - y = 1 / (1 + exp(B0 + B1 * x))
        """
    try:
        # Calculate the logistic function value
        y = 1 / (1 + math.exp(-coeff[0] - coeff[1] * x))
    except:
        # Handle overflow or other numerical errors by returning 0
        y = 0
    return y

def invlogisticRegression(p, B):  # invlogistic Function
    """
        Inverse Logistic Function:
        Determines the value of `x` corresponding to a given probability `p`
        using the inverse of the logistic function.

        Parameters:
            - p (float): Desired probability (must be in the range (0, 1)).
            - B0 (float): Intercept of the logistic regression.
            - B1 (float): Coefficient (slope) for the independent variable.

        Returns:
            - x (float): The value of `x` corresponding to the given probability `p`.

        Formula:
            x = (log(1 / p - 1) - B[0]) / B[1]
        """
    try:
        # Calculate the inverse logistic function value
        x = - (math.log((1/p) - 1) + B[0]) / B[1]
    except:
        # Return NaN in case of any error (e.g., division by zero)
        x = float('nan')
    return math.exp(x)

def latex_float(f):
    float_str = str(f)
    if "e" in float_str:
        astr = '{:.2E}'.format(f)
        return '$' + astr.replace('E', ' \\times 10^{') + '} $'
    else:
        return float_str

def signed_fmt(x):
    return f'+{x:5.3f}' if x >= 0 else f'{x:5.3f}'


def LOGISTICREGRESSION(Df, filePath=None, fileName=None, desiredLDLFraction=0.90, Nbootstrap=500, fontsize=7,
                       xCategory='tc_EPBFE', yCategory='', xlabel='Whole gas emission rate', xunits='slpm',
                       figsize=None, fig=None, axes=None, ScaleFactor=1, ylabel="Probability of Detection (POD)",
                       borderpad=0.4, eps=1e-6, xScale="linear", yScale="linear", xmax=None, digits=1, xmin=0, N=1000,
                       mColors='b', bColors='#B0C4DE', loc='lower right', baseline=0, dpi=400, ci=0.20):
    """
        Perform logistic regression analysis with bootstrapping to estimate the Probability of Detection (POD).

        Parameters:
        - Df (pd.DataFrame): Input data containing classification and independent variable.
        - outputFilePath (str): Directory to save the plot.
        - fileName (str): File name for saving the plot.
        - desiredLDLFraction (float): Target detection probability for determining LDL.
        - Nbootstrap (int): Number of bootstrap iterations for uncertainty estimation.
        - xScaleFactor (float): Scaling factor for the independent variable.
        - xCategory (str): Column name for the independent variable.
        - xlabel (str): Label for the x-axis.
        - xunit (str): Unit for the x-axis.
        - xmax, xmin (float): Maximum and minimum x-axis values.
        - figsize (tuple): Figure size for the plot.
        - fig, axes: Existing figure and axes for plotting.
        - digits (int): Number of decimal places for LDL values.
        - fontsize (int): Font size for plot labels.
        - dpi (int): Resolution for saving the plot.
        - legendloc (str): Location of the plot legend.
        - alpha (float): Transparency for bootstrap curves.
        - cData (str): Column name for classification data ('TP', 'FN').
        - bsColor (str): Color for bootstrap curves.
        - color (str): Color for the main logistic regression curve.
        - xScale (str): Scale for the x-axis ('linear', 'log', etc.).
        - yScale (str): Scale for the y-axis ('linear', 'log', etc.).
        - baseline (float): baseline emission rate by the test center
        - ci: Confidence interval

        Returns:
        - fig, ax: The figure and axes objects for the generated plot.
    """

    # Axes declaration
    plt.rcParams.update({'font.size': fontsize})
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize) if figsize else plt.subplots()
    else:
        ax = axes if axes is not None else plt.gca()

    # Filtering out the desired columns from the classified dataframe and classifying detections as 1 or 0
    CRDF = Df.loc[Df[yCategory].isin(['TP', 'FN'])] # Filter only rows classified as TP and FN.
    CRDF = CRDF.reset_index(drop=True) # Prevent duplicate index issues
    CRDF = CRDF.loc[:, ~CRDF.columns.duplicated()] # Prevent duplicate column issues
    logisticDf = CRDF.filter([xCategory, yCategory], axis=1) # Filter only columns required for the analysis.
    logisticDf['Detection'] = logisticDf[yCategory].apply(lambda x: 1 if x == 'TP' else 0) # Create a column "Detection" where TP is 1 otherwise 0.
    logisticDf[xCategory] = logisticDf[xCategory].apply(lambda x: (x - baseline) * ScaleFactor) # Subtract baseline and apply scaling factor.

    # Developing the logistic regression model
    xDatafield = logisticDf[xCategory].values # Extract x-axis values as arrays.
    minFlowRate = logisticDf[xCategory].min() # Extract minimum x-axis values
    maxFlowRate = logisticDf[xCategory].max() # Extract maximum x-axis values
    X = np.log(xDatafield + eps).reshape(-1, 1) # Convert x-axis values to 2-D array for logistic regression assessment
    Y = logisticDf['Detection'].values # Extract y-axis values as arrays.
    logReg = LogisticRegression(solver='lbfgs') # Develop the logistic regression model
    logReg.fit(X, Y) # Use the model to fit the dataset.
    modelCoef = logReg.coef_[0, 0] # Extract the coefficient of the model fit
    modelIntercept = logReg.intercept_[0] # Extract the intercept of the model fit
    param = [modelIntercept, modelCoef]
    ldl = invlogisticRegression(p=desiredLDLFraction, B=param) # Estimate the value of x-value given y-value: desiredLDLFraction.

    # Data for main plot
    xMain = np.logspace(np.log10(np.clip(xDatafield[xDatafield > 0].min(), eps, None)) - 15, np.log10(xDatafield.max()) + 15, N)
    xMain_log = np.log(xMain + eps).reshape(-1, 1) # Evaluate the log of the x-axis data.
    yMain = logReg.predict_proba(xMain_log)[:, 1] # Predict the y-axis values.

    # Create a bootstrapped curve for the main curve.
    idx = np.random.randint(1, len(logisticDf), size=(len(logisticDf), Nbootstrap))
    LDLmc: NDArray[np.float64] = np.zeros((1, Nbootstrap)) # Container for the bootstrapped lower detection limit.
    Bmc: NDArray[np.float64] = np.zeros((2, Nbootstrap)) # Container for the bootstrapped coefficient and intercept.
    Ymc: NDArray[np.float64] = np.zeros((N, Nbootstrap)) # Container for the bootstrapped predicted POD values.

    # Bootstrapping analysis - repeating everything done on the main curve using bootstrapped data
    for i in list(range(Nbootstrap)):
        # For each iteration, check if the selected detection values contain at least TP and FN, otherwise, model fitting will fail or produce meaningless results.
        if len(np.unique(logisticDf['Detection'].iloc[idx[:, i]].values)) < 2:
            continue
        xbot_rs = logisticDf[xCategory].iloc[idx[:, i]]
        xbot = np.log(np.array(xbot_rs) + eps).reshape(-1, 1)
        ybot = logisticDf['Detection'].iloc[idx[:, i]].values
        botmodel = LogisticRegression(solver='lbfgs')
        botmodel.fit(xbot, ybot)
        botCoef = botmodel.coef_[0, 0]
        botIntercept = botmodel.intercept_[0]
        botB = [botIntercept, botCoef]
        # Write
        try:
            LDLmc[0, i] = invlogisticRegression(p=desiredLDLFraction, B=botB)
            Bmc[0, i] = botIntercept
            Bmc[1, i] = botCoef
            Ymc[:, i] = botmodel.predict_proba(xMain_log)[:, 1]
        except:
            continue

    ## Evaluating the bootstrapped curve
    keep = filteringBootstraps(Bmc=Bmc, beta_hat=param, ci=ci) # Filtering bootstrapp based on the Mahalanobis distance.
    # The Mahalanobis distance analysis was applied to retain only those bootstrapped curves within the
    # ci confidence ellipse of the curve-fitting parameter distribution.
    LDLmc = LDLmc[keep.reshape(1, -1)] # Keep the only LDL that corresponds to the filtered bootstrapped curves.
    UpperCL = np.max(LDLmc) # Extract the maximum LDL value.
    LowerCL = np.min(LDLmc) # Extract the minimum LDL value.

    # Equations and label definition
    denominator = '1 + e^{(%s%s*\\ln x)}' % (signed_fmt(-param[0]), signed_fmt(-param[1]))
    equation = r'$\frac{1}{%s}$' % (denominator)
    if (ldl < maxFlowRate) and (ldl > minFlowRate):
        lowLDL = latex_float(round(LowerCL, digits))
        upLDL = latex_float(round(UpperCL, digits))
        dl = latex_float(round(ldl, digits))
        lab = f'{dl} [{lowLDL}, {upLDL}]'
        label = ('DL{percent}: {lb}').format(percent=f'{int(desiredLDLFraction * 100)}', lb=lab)
    else:
        label = ('DL{percent} outside tested range').format(percent=f'{int(desiredLDLFraction * 100)}')

    # Plotting the main curve
    ax.scatter(x=logisticDf[xCategory].tolist(), y=logisticDf['Detection'], label='TP & FN', s=5, facecolors='none',
               edgecolors='k')
    ax.plot(xMain, yMain, ls='solid', lw=1, c=mColors, marker='None', label=equation, zorder=2)

    # Plotting the bootstraps
    for p in list(range(Nbootstrap)):  # To plot each individual bootstrapped curve
        try:
            # Only plot the filtered curves
            if keep[p]:
                ax.plot(xMain, Ymc[p], ls='solid', lw=0.5, c=bColors, marker='None', zorder=1)
        except:
            continue

    # Plot the detection threshold - horizontal line
    ax.plot([0, ldl], [desiredLDLFraction, desiredLDLFraction], ls=':', lw=0.5, c='k', marker='None', zorder=2)
    # Plot the detection threshold - vertical line
    try:
        ax.plot([ldl, ldl], [0, desiredLDLFraction], ls=':', lw=1.0, c='k', marker='None', zorder=2, label=label)
    except:
        pass

    # Plotting the grid lines
    a_min = int(np.ceil(np.log10(xmin)))
    a_max = int(np.floor(np.log10(xmax)))
    values = [10 ** a for a in range(a_min, a_max + 1)]
    for x in values:
        ax.axvline(x=x, color='k', linestyle='--', linewidth=0.03)
    for y in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ax.axhline(y=y, color='k', linestyle='--', linewidth=0.03)

    # Finishing Touches
    ax.set_xscale(xScale) # Set scale for x-axis
    ax.set_yscale(yScale) # Set scale for y-axis
    ax.set_ylim([-0.05, 1.05]) # Set y-axis limits
    # Set x-axis limits
    if xmax:
        ax.set_xlim([xmin, xmax])
    else:
        ax.set_xlim([0, maxFlowRate])
    # Set y-axis label
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    # Set x-axis label
    if xlabel:
        ax.set_xlabel(xlabel + " (" + xunits + ")", fontsize=fontsize)
    # Setting the grid size
    ax.grid(axis='both', alpha=0.05)
    # Setting the font size and properties
    fontP = FontProperties()
    fontP.set_size(fontsize)
    leg = ax.legend(loc=loc, prop=fontP, borderpad=borderpad)
    leg.get_frame().set_linewidth(0.5)

    # save data
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=dpi)

    return


def densityPlot(DF,xCategory, classification, fig=None, axes=None, fontsize=12, figsize=(6, 4), N=1000,
                eps=1e-9, xScale="linear", xlimit=None, loc="upper right", fileName=None, filePath=None,
                dpi=300):
    """
    Plot normalized KDE density curves for TP and FN classes.

    Parameters:
        - DF (DataFrame): Input dataframe containing classification and xCategory data
        - xCategory (str): Name of the numeric column used on x-axis
        - classification (str): Column name with 'TP' and 'FN' labels
        - fig : Existing figure handle
        - axes : Existing axes handle
        - fontsize : Font size for plot
        - figsize : Figure size if creating a new figure
        - N : Number of points used for x-axis sampling in KDE
        - eps : Small constant to avoid log(0)
        - xScale (str) : 'linear' or 'log'
        - xlimit : limits of the x-axis
        - loc : Legend location
        - fileName : Name of output image file
        - filePath : Output directory
        - dpi : Image resolution
    """

    # Figure and axis setup
    plt.rcParams.update({'font.size': fontsize})

    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = axes if axes is not None else plt.gca()

    # Split data into categories
    positive_data = DF[DF[classification] == 'TP'][xCategory].dropna().values
    negative_data = DF[DF[classification] == 'FN'][xCategory].dropna().values

    # Extract x-axis values as arrays.
    xDatafield = DF[xCategory].dropna().values
    if xScale == "log":
        xMain = np.logspace(np.log10(np.clip(xDatafield[xDatafield > 0].min(), eps, None)) - 15,
                            np.log10(xDatafield.max()) + 15, N)
    else:
        xMain = np.linspace(0, int(xDatafield.max()) + 5, N)

    # Plot TP density
    if len(positive_data) > 1:
        try:
            # Estimate the probability density function (PDF) of positive_data
            # using a non-parametric kernel density estimator.
            kde_pos = gaussian_kde(positive_data)

            # Evaluate the KDE over the predefined x-axis values (xMain)
            kde_pos_values = kde_pos(xMain)

            # Normalize the KDE so that the maximum value equals 1
            # This makes it easier to compare relative density shapes
            kde_pos_norm = kde_pos_values / kde_pos_values.max()

            # Plot the normalized KDE curve
            ax.plot(xMain, kde_pos_norm, color='blue', linewidth=1.5, label='TP Density', alpha=0.8)
        except Exception as e:
            print("Could not plot TP density:", e)

    # Plot FN density
    if len(negative_data) > 1:
        try:
            # Estimate the probability density function (PDF) of positive_data
            # using a non-parametric kernel density estimator.
            kde_neg = gaussian_kde(negative_data)

            # Evaluate the KDE over the predefined x-axis values (xMain)
            kde_neg_values = kde_neg(xMain)

            # Normalize the KDE so that the maximum value equals 1
            # This makes it easier to compare relative density shapes
            kde_neg_norm = kde_neg_values / kde_neg_values.max()

            # Plot the normalized KDE curve
            ax.plot(xMain, kde_neg_norm, color='red', linewidth=1.5, linestyle='--', label='FN Density', alpha=0.8)
        except Exception as e:
            print("Could not plot FN density:", e)

    # Axis formatting
    ax.set_ylabel("Normalized Density", fontsize=fontsize)
    if xlimit:
        ax.set_xlim(xlimit)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.15)

    # Legend formatting
    fontP = FontProperties(size=fontsize)
    legend = ax.legend(loc=loc, prop=fontP)
    legend.get_frame().set_linewidth(0.5)

    # Save figure if requested
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)

        print("Saving plot to:", path)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")

    return fig, ax