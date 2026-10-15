"""Figure builders and statistical fitting helpers for ADED performance analysis."""
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
from scipy.stats import ks_2samp

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
        Calculate the percentage of calculated factors within ±factor.

        Parameters
        ----------
        quantDF : pandas.DataFrame
            Data containing "QuantificationFactors". Rows with missing or
            nonnumeric factors are excluded from both numerator and denominator.

        factor : float
            Finite, nonnegative threshold. Both endpoints are included.

        Returns
        -------
        int or None
            Percentage truncated to an integer, preserving the original behavior.
            Returns None when no numeric factors are available.
    """
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("factor must be finite and nonnegative.")

    # Keep only rows where a numeric factor was calculated.
    calculated = pd.to_numeric(quantDF["QuantificationFactors"], errors="coerce").dropna()

    if calculated.empty:
        return None

    # Infinite calculated factors remain in the denominator but fall outside any finite threshold.
    within = calculated.between(-factor, factor, inclusive="both")

    return int(within.sum() * 100 / len(calculated))


# def scatter_plot(data, x_column, x_columnLower=None, x_columnUpper=None, y_column=None, y_columnLower=None,
#                  y_columnUpper=None, xFactor=1, yFactor=1, edge_color="k", xunit="", yunit="", limit=None,
#                  sColor="blue", lColor="red", rColor="green", lw=1.5, fig=None, figsize=None, axes=None,
#                  slabel="Data", factor=None, xlabel="", ylabel="", xScale="log", yScale="log", fontsize=10,
#                  fileName=None, outputFilePath=None, legendloc='upper left', dpi=400):
#
#     """
#         Creates a scatter plot with:
#         - error bars
#         - regression line
#         - R-squared
#         - 1:1 line
#         - optional factor bounds
#     """
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # SETUP FIGURE AND AXES
#     # ------------------------------------------------------------------------------------------------------------------
#
#     if axes is not None:
#         ax = axes
#         fig = ax.figure
#     else:
#         fig, ax = plt.subplots(figsize=figsize)
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # CLEAN AND SCALE DATA
#     # ------------------------------------------------------------------------------------------------------------------
#     required_cols = [x_column, y_column]
#     data = data.dropna(subset=required_cols).copy()
#     xVal = data[x_column] * xFactor
#     yVal = data[y_column] * yFactor
#
#     # ----------------------------------------------
#     # Remove non-positive values for log scaling
#     # ----------------------------------------------
#     if xScale == "log":
#         mask = xVal > 0
#         xVal = xVal[mask]
#         yVal = yVal[mask]
#         data = data.loc[mask]
#
#     if yScale == "log":
#         mask = yVal > 0
#         xVal = xVal[mask]
#         yVal = yVal[mask]
#         data = data.loc[mask]
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # ERROR BARS
#     # ------------------------------------------------------------------------------------------------------------------
#     xerror_bars = None
#     yerror_bars = None
#
#     # X error bars
#     if (x_columnLower is not None and x_columnUpper is not None and x_columnLower in data.columns and x_columnUpper in data.columns):
#         xVal_lower = data[x_columnLower] * xFactor
#         xVal_upper = data[x_columnUpper] * xFactor
#         xlower_errors = np.abs(xVal - xVal_lower)
#         xupper_errors = np.abs(xVal_upper - xVal)
#         xerror_bars = [xlower_errors, xupper_errors]
#
#     # Y error bars
#     if (y_columnLower is not None and y_columnUpper is not None and y_columnLower in data.columns and y_columnUpper in data.columns):
#         yVal_lower = data[y_columnLower] * yFactor
#         yVal_upper = data[y_columnUpper] * yFactor
#         ylower_errors = np.abs(yVal - yVal_lower)
#         yupper_errors = np.abs(yVal_upper - yVal)
#         yerror_bars = [ylower_errors, yupper_errors]
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # SCATTER PLOT
#     # ------------------------------------------------------------------------------------------------------------------
#     ax.errorbar(x=xVal, y=yVal, xerr=xerror_bars, yerr=yerror_bars, fmt='o', color=sColor, ecolor=edge_color,
#                 elinewidth=0.5, capsize=3, mew=0.5, markerfacecolor='none', label=slabel, alpha=0.8)
#     # ------------------------------------------------------------------------------------------------------------------
    # REGRESSION + R-squared
#     # ------------------------------------------------------------------------------------------------------------------
#     X = np.array(xVal).reshape(-1, 1)
#     y = np.array(yVal)
#     model = LinearRegression()
#     model.fit(X, y)
#     r_squared = model.score(X, y)
#
#     # Sort before plotting line
#     sort_idx = np.argsort(xVal)
#     ax.plot(np.array(xVal)[sort_idx], model.predict(X)[sort_idx], ls="-.", color=rColor, linewidth=lw,
#            label=fr"$R^2 = {r_squared:.2f}$")
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # LIMITS
#     # ------------------------------------------------------------------------------------------------------------------
#     if limit is None:
#         positive_vals = np.concatenate([np.array(xVal), np.array(yVal)])
#         positive_vals = positive_vals[positive_vals > 0]
#         min_lim = positive_vals.min()
#         max_lim = positive_vals.max()
#         limit = (min_lim, max_lim)
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # 1:1 LINE
#     # ------------------------------------------------------------------------------------------------------------------
#     line_vals = np.array(limit)
#     ax.plot(line_vals, line_vals, ls='--', color=lColor, linewidth=lw, label='1:1 Line')
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # FACTOR BOUNDS
#     # ------------------------------------------------------------------------------------------------------------------
#     if factor is not None:
#         factor_value = factor[0]
#         # ax.plot(line_vals, line_vals / factor_value, ls=':', color='black', linewidth=lw,
#         #         label=f"Factor of {factor[0]} = {factor[1]}%")
#         ax.plot(line_vals, line_vals / factor_value, ls=':', color='black', linewidth=lw,
#                 label=f"{factor[1]}% estimates within [x/{factor[0]}, {factor[0]}x")
#         ax.plot(line_vals, line_vals * factor_value, ls=':', color='black', linewidth=lw)
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # AXIS SETTINGS
#     # ------------------------------------------------------------------------------------------------------------------
#     ax.set_xscale(xScale)
#     ax.set_yscale(yScale)
#     ax.set_xlim(limit)
#     ax.set_ylim(limit)
#
#     # Axis labels
#     xlabel_final = xlabel
#     ylabel_final = ylabel
#     if xunit:
#         xlabel_final += f" ({xunit})"
#     if yunit:
#         ylabel_final += f" ({yunit})"
#     ax.set_xlabel(xlabel_final, fontsize=fontsize)
#     ax.set_ylabel(ylabel_final, fontsize=fontsize)
#
#     # Grid
#     ax.grid(True, which='both', linestyle='--', alpha=0.4)
#     # ------------------------------------------------------------------------------------------------------------------
#     # LEGEND
#     # ------------------------------------------------------------------------------------------------------------------
#     fontP = FontProperties()
#     fontP.set_size(fontsize)
#     leg = ax.legend(loc=legendloc, prop=fontP)
#     leg.get_frame().set_edgecolor('black')
#     fig.tight_layout()
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # SAVE FIGURE
#     # ------------------------------------------------------------------------------------------------------------------
#     if fileName is not None:
#         save_dir = outputFilePath if outputFilePath else os.getcwd()
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, fileName)
#         print(f"Saving figure to:\n{save_path}")
#         fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
#
#     return fig, ax

import patsy
# def continuous_nonlinear_analysis(data, estimation_col, continuous_vars, spline_df=5, alpha=0.05):
#     """
#         Analyze continuous predictors of estimation performance without assuming linear relationships.
#         Dependent variable
#         ------------------
#         PerformanceError = abs(log(Estimation))
#         Interpretation:
#             PerformanceError = 0  -> ideal performance
#             Larger values         -> worse performance
#         Model
#         -----
#         PerformanceError =
#             intercept
#             + f1(X1)
#             + f2(X2)
#             + ...
#             + error
#
#         where each f(X) is represented by a cubic B-spline.
#         All predictors are included in the same model, so the significance
#         of each variable is evaluated while adjusting for all other variables.
#         Parameters
#         ----------
#         data : pandas.DataFrame
#         estimation_col : str
#             Name of the Estimation column. Values must be > 0.
#         continuous_vars : list of str
#             Continuous predictor variables.
#             Example:["ReleaseRate", "ReleaseDuration", "WindSpeed", "Temperature"]
#         spline_df : int, default=5
#             Controls the flexibility of each nonlinear relationship.
#         alpha : float, default=0.05
#             Statistical significance level.
#         Returns
#         -------
#         summary : pandas.DataFrame - One row per predictor with its joint p-value.
#         model : statsmodels fitted model
#         model_data : pandas.DataFrame
#             Data used in the fitted model.
#         predict_function : function
#             Function for predicting PerformanceError from new data.
#     """
#
#     # ============================================================
#     # 1. PREPARE DATA
#     # ============================================================
#     required_cols = [estimation_col] + continuous_vars
#     # Work on a copy so the original dataframe is not modified.
#     df = data[required_cols].copy()
#     # Remove observations containing missing values.
#     df = df.dropna()
#     # log(Estimation) is only defined for positive values.
#     df = df[df[estimation_col] > 0].copy()
#     if len(df) < 10:
#         raise ValueError("Too few valid observations remain after removing ")
#
#     # ============================================================
#     # 2. CREATE DEPENDENT VARIABLE
#     # ============================================================
#     # PerformanceError = |ln(Estimation)|
#     # Examples: Estimation = 1.0  -> Error = 0
#     # Estimation = 2.0  -> Error = 0.693
#     # Estimation = 0.5  -> Error = 0.693
#     # Therefore: lower error  = better performance; higher error = worse performance
#     df["_PerformanceError"] = np.abs(np.log(df[estimation_col]))
#
#     # ============================================================
#     # 3. BUILD NONLINEAR MODEL
#     # ============================================================
#     # Each continuous variable is represented using a cubic B-spline.
#     # This allows the relationship to:
#     # - increase
#     # - decrease
#     # - flatten
#     # - change direction
#     # without assuming a straight-line relationship.
#
#     spline_terms = []
#     for var in continuous_vars:
#         term = (f"bs(Q('{var}'), "
#                 f"df={spline_df}, "
#                 f"degree=3, "
#                 f"include_intercept=False)")
#         spline_terms.append(term)
#
#     formula = ("_PerformanceError ~ "+ " + ".join(spline_terms))
#
#     # ============================================================
#     # 4. CREATE MODEL MATRICES
#     # ============================================================
#     y, X = patsy.dmatrices(formula, data=df, return_type="dataframe")
#
#     # ============================================================
#     # 5. FIT MODEL
#     # ============================================================
#     # Fit OLS to the spline-expanded predictors.
#     # HC3 provides heteroscedasticity-robust standard errors.
#     model = sm.OLS(y, X).fit(cov_type="HC3")
#     # Save spline definitions so exactly the same transformations
#     # can be used for future predictions.
#     design_info = X.design_info
#
#     # ============================================================
#     # 6. PREDICTION FUNCTION
#     # ============================================================
#     def predict_function(new_data):
#         """
#         Predict PerformanceError for new predictor values.
#         """
#         new_X = patsy.build_design_matrices([design_info], new_data, return_type="dataframe")[0]
#         return np.asarray(model.predict(new_X), dtype=float).ravel()
#
#     # ============================================================
#     # 7. TEST EACH VARIABLE
#     # ============================================================
#     results = []
#     for var in continuous_vars:
#         # A spline produces several coefficients for one variable.
#         # Identify all model columns belonging to this predictor.
#
#         token = f"bs(Q('{var}')"
#         columns = [col for col in X.columns if token in col]
#
#         # --------------------------------------------------------
#         # Joint Wald test
#         # --------------------------------------------------------
#         # H0:
#         #     all spline coefficients for this variable = 0
#         # Therefore the resulting p-value asks:
#         # "Does this variable significantly affect PerformanceError
#         #  after accounting for all other predictors?"
#
#         R = np.zeros((len(columns), len(X.columns)))
#         for i, col in enumerate(columns):
#             R[i,X.columns.get_loc(col)] = 1
#
#         test = model.wald_test(R, scalar=True)
#         p_value = float(test.pvalue)
#         results.append({"Variable": var, "P_value": p_value, "Significant": p_value < alpha})
#
#
#     # ============================================================
#     # 8. CREATE SUMMARY TABLE
#     # ============================================================
#     summary = pd.DataFrame(results).sort_values("P_value").reset_index(drop=True)
#
#     for i, row in summary.iterrows():
#         print(f"The variable {row['Variable']} has a P value of {row['P_value']} and its statistical significance is {row['Significant']}")
#     return

def continuous_nonlinear_analysis(data, estimation_col, continuous_vars, spline_df=5, alpha=0.05):
    """
        Analyze continuous predictors of estimation performance without assuming linear relationships.
        Dependent variable
        ------------------
        PerformanceError = abs(log(Estimation))
        Interpretation:
            PerformanceError = 0  -> ideal performance
            Larger values         -> worse performance
        Model
        -----
        PerformanceError = intercept
            + f1(X1)
            + f2(X2)
            + ...
            + error

        where each f(X) is represented by a cubic B-spline.
        All predictors are included in the same model, so the significance of each variable is evaluated while adjusting for all other variables.

        General trend
        -------------
        For each predictor, the fitted model is evaluated across that predictor's observed range while all other
        predictors are held at their median values.

        The average derivative of the fitted curve is used to summarize the general trend:

            Average_Trend < 0:
                PerformanceError generally decreases as the variable increases.Therefore, estimation performance
                generally improves.

            Average_Trend > 0:
                PerformanceError generally increases as the variable increases. Therefore, estimation performance
                generally worsens.

            Average_Trend approximately 0:
                No clear overall increasing or decreasing trend.

        Important:
            Average_Trend is a summary of the overall nonlinear trend. It is NOT a constant regression coefficient.

        Parameters
        ----------
        data : pandas.DataFrame
        estimation_col : str
            Name of the Estimation column. Values must be > 0.
        continuous_vars : list of str
            Continuous predictor variables.
            Example: ["ReleaseRate", "ReleaseDuration", "WindSpeed", "Temperature"]

        spline_df : int, default=5
            Controls the flexibility of each nonlinear relationship.
        alpha : float, default=0.05
            Statistical significance level.

        Returns
        -------
        summary : pandas.DataFrame
            One row per predictor containing:
                - P_value
                - Significant
                - Average_Trend
                - Trend
        model : statsmodels fitted model
        model_data : pandas.DataFrame
            Data used in the fitted model.
        predict_function : function
            Function for predicting PerformanceError from new data.
    """

    # ==================================================================================================================
    # 1. PREPARE DATA
    # ==================================================================================================================
    required_cols = [estimation_col] + continuous_vars
    # Work on a copy so the original dataframe is not modified.
    df = data[required_cols].copy()
    # Remove observations containing missing values.
    df = df.dropna()
    # log(Estimation) is only defined for positive values.
    df = df[df[estimation_col] > 0].copy()
    if len(df) < 10:
        raise ValueError("Too few valid observations remain after removing ")

    # ==================================================================================================================
    # 2. CREATE DEPENDENT VARIABLE
    # ==================================================================================================================
    # PerformanceError = |ln(Estimation)|
    # Examples:
    # Estimation = 1.0  -> Error = 0
    # Estimation = 2.0  -> Error = 0.693
    # Estimation = 0.5  -> Error = 0.693
    # Therefore:
    # lower error  = better performance
    # higher error = worse performance
    df["_PerformanceError"] = np.abs(np.log(df[estimation_col]))

    # ==================================================================================================================
    # 3. BUILD NONLINEAR MODEL
    # ==================================================================================================================
    # Each continuous variable is represented using a cubic B-spline.
    # This allows the relationship to:
    # - increase
    # - decrease
    # - flatten
    # - change direction
    # without assuming a straight-line relationship.

    spline_terms = []
    for var in continuous_vars:
        term = (
            f"bs(Q('{var}'), "
            f"df={spline_df}, "
            f"degree=3, "
            f"include_intercept=False)")

        spline_terms.append(term)
    formula = ("_PerformanceError ~ " + " + ".join(spline_terms))

    # ==================================================================================================================
    # 4. CREATE MODEL MATRICES
    # ==================================================================================================================
    y, X = patsy.dmatrices(formula, data=df, return_type="dataframe")

    # ==================================================================================================================
    # 5. FIT MODEL
    # ==================================================================================================================
    # Fit OLS to the spline-expanded predictors. HC3 provides heteroscedasticity-robust standard errors.
    model = sm.OLS(y, X).fit(cov_type="HC3")

    # Save spline definitions so exactly the same transformations can be used for future predictions.
    design_info = X.design_info

    # ==================================================================================================================
    # 6. PREDICTION FUNCTION
    # ==================================================================================================================
    def predict_function(new_data):
        """
        Predict PerformanceError for new predictor values.
        """
        new_X = patsy.build_design_matrices([design_info], new_data, return_type="dataframe")[0]
        return np.asarray(model.predict(new_X), dtype=float).ravel()

    # ==================================================================================================================
    # 7. TEST EACH VARIABLE AND CALCULATE GENERAL TREND
    # ==================================================================================================================
    results = []
    for var in continuous_vars:
        # --------------------------------------------------------
        # A. JOINT WALD TEST
        # --------------------------------------------------------
        # A spline produces several coefficients for one variable.
        # Identify all model columns belonging to this predictor.
        token = f"bs(Q('{var}')"
        columns = [col for col in X.columns if token in col]
        # H0: all spline coefficients for this variable = 0
        # Therefore the resulting p-value asks: "Does this variable significantly affect PerformanceError
        # after accounting for all other predictors?"
        R = np.zeros((len(columns), len(X.columns)))
        for i, col in enumerate(columns):
            R[i, X.columns.get_loc(col)] = 1

        test = model.wald_test(R, scalar=True)
        p_value = float(test.pvalue)
        # --------------------------------------------------------
        # B. CALCULATE GENERAL NONLINEAR TREND
        # --------------------------------------------------------
        # Create values spanning the observed range of this variable.
        x_grid = np.linspace(df[var].min(), df[var].max(), 200)
        # Hold all predictors at their median values.
        # This allows us to isolate the adjusted relationship between the current variable and PerformanceError.
        reference = pd.DataFrame({v: [df[v].median()] for v in continuous_vars})

        # Repeat the reference observation for every value in the predictor grid.
        prediction_data = pd.concat([reference] * len(x_grid), ignore_index=True)

        # Only change the variable currently being evaluated.
        prediction_data[var] = x_grid

        # Predict PerformanceError across the variable's range.
        predicted_error = predict_function(prediction_data)

        # Calculate the derivative of the fitted nonlinear curve.
        # This measures: change in PerformanceError change in X Because the model is nonlinear, this slope may vary
        # across the predictor range.
        local_slopes = np.gradient(predicted_error, x_grid)

        # Average the local slopes to obtain a simple summary of the GENERAL direction of the relationship.
        average_trend = float(np.mean(local_slopes))

        # --------------------------------------------------------
        # C. INTERPRET GENERAL TREND
        # --------------------------------------------------------
        tolerance = 1e-8
        if average_trend < -tolerance:
            trend = ("Performance error generally decreases " "as the variable increases")
            performance_trend = ("Performance generally improves " "as the variable increases")
        elif average_trend > tolerance:
            trend = ("Performance error generally increases " "as the variable increases")
            performance_trend = ("Performance generally worsens " "as the variable increases")
        else:
            trend = ("No clear overall increasing or " "decreasing trend")

            performance_trend = ("No clear overall improvement or ""worsening in performance")


        # --------------------------------------------------------
        # D. STORE RESULT
        # --------------------------------------------------------
        results.append({"Variable": var, "P_value":p_value, "Significant": p_value < alpha, "Average_Trend": average_trend,
                        "Trend": trend, "Performance_Trend": performance_trend})

    # ==================================================================================================================
    # 8. CREATE SUMMARY TABLE
    # ==================================================================================================================
    summary = (pd.DataFrame(results).sort_values("P_value").reset_index(drop=True))

    # ==================================================================================================================
    # 9. PRINT RESULTS
    # ==================================================================================================================
    for _, row in summary.iterrows():
        print(f"\nVariable: {row['Variable']}")
        print(f"P-value: {row['P_value']:.5f}")
        print(f"Statistically significant: " f"{row['Significant']}")
        print(f"Average trend: "f"{row['Average_Trend']:.5f}")
        print(f"General trend: " f"{row['Trend']}")
        print(f"Interpretation: " f"{row['Performance_Trend']}")

    return

def scatter_plot(data, x_column, x_columnLower=None, x_columnUpper=None, y_column=None, y_columnLower=None, y_columnUpper=None,
                 xFactor=1, yFactor=1, edge_color="k", xunit="", yunit="", limit=None, sColor="blue", lColor="red", rColor="green",
                 lw=1.5, fig=None, figsize=None, axes=None, slabel="Data", factor=None, xlabel="", ylabel="", xScale="log",
                 yScale="log", fontsize=10, fileName=None, outputFilePath=None, legendloc="lower right", dpi=400, color_column=None,
                 cmap="viridis", colorbar_ticks=5, colorbar_label=""):
    """
    Create a scatter plot with:
        - optional x/y error bars
        - regression line and R-squared
        - 1:1 line
        - optional factor bounds
        - optional numerical scatter-marker coloring
        - horizontal colorbar above the plot

    Parameters
    ----------
    color_column : str, optional
        Numerical column used to determine scatter-marker colors.
        If None, all markers use sColor.

    cmap : str, default "viridis"
        Matplotlib colormap used when color_column is provided.

    colorbar_ticks : int, default 5
        Number of evenly spaced ticks on the colorbar.

    colorbar_label : str, optional
        Label displayed above the colorbar. If empty, color_column is used.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # SETUP FIGURE AND AXES
    # ------------------------------------------------------------------------------------------------------------------
    if axes is not None:
        ax = axes
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # ------------------------------------------------------------------------------------------------------------------
    # CLEAN AND SCALE DATA
    # ------------------------------------------------------------------------------------------------------------------
    required_cols = [x_column, y_column]
    if color_column is not None:
        if color_column not in data.columns:
            raise KeyError(f"color_column '{color_column}' is not present in data.")
        required_cols.append(color_column)
    data = data.dropna(subset=required_cols).copy()
    xVal = data[x_column] * xFactor
    yVal = data[y_column] * yFactor

    data["Trend"] = data[y_column]/data[x_column]
    # Optional multivariable spline analysis is available separately through
    # continuous_nonlinear_analysis; a basic scatterplot does not require it.
    # ------------------------------------------------------------------------------------------------------------------
    # Remove non-positive values for log scaling
    # ------------------------------------------------------------------------------------------------------------------
    if xScale == "log":
        mask = xVal > 0
        data = data.loc[mask]
        xVal = xVal.loc[mask]
        yVal = yVal.loc[mask]

    if yScale == "log":
        mask = yVal > 0
        data = data.loc[mask]
        xVal = xVal.loc[mask]
        yVal = yVal.loc[mask]

    if data.empty:
        raise ValueError("No valid observations remain after filtering.")

    # ------------------------------------------------------------------------------------------------------------------
    # ERROR BARS
    # ------------------------------------------------------------------------------------------------------------------
    xerror_bars = None
    yerror_bars = None

    # X error bars
    if (x_columnLower is not None and x_columnUpper is not None and x_columnLower in data.columns and x_columnUpper in data.columns):
        xVal_lower = data[x_columnLower] * xFactor
        xVal_upper = data[x_columnUpper] * xFactor
        xlower_errors = np.abs(xVal - xVal_lower)
        xupper_errors = np.abs(xVal_upper - xVal)
        xerror_bars = [xlower_errors, xupper_errors]

    # Y error bars
    if (y_columnLower is not None and y_columnUpper is not None and y_columnLower in data.columns and y_columnUpper in data.columns):
        yVal_lower = data[y_columnLower] * yFactor
        yVal_upper = data[y_columnUpper] * yFactor
        ylower_errors = np.abs(yVal - yVal_lower)
        yupper_errors = np.abs(yVal_upper - yVal)
        yerror_bars = [ylower_errors, yupper_errors]

    # ------------------------------------------------------------------------------------------------------------------
    # ERROR BAR LINES
    # Plot error bars separately from scatter markers so the markers can be colored individually from color_column.
    # ------------------------------------------------------------------------------------------------------------------
    if xerror_bars is not None or yerror_bars is not None:
        ax.errorbar(x=xVal, y=yVal, xerr=xerror_bars, yerr=yerror_bars, fmt="none", ecolor=edge_color, elinewidth=0.5,
                    capsize=3, capthick=0.5, alpha=0.8, zorder=1)

    # ------------------------------------------------------------------------------------------------------------------
    # SCATTER PLOT
    # ------------------------------------------------------------------------------------------------------------------
    if color_column is not None:
        #data[color_column] = data[color_column].round(1)
        color_values = pd.to_numeric(data[color_column], errors="coerce")
        if color_values.isna().any():
            raise ValueError(f"Column '{color_column}' contains non-numerical values.")
        color_min = color_values.min()
        color_max = color_values.max()
        scatter = ax.scatter(xVal, yVal, c=color_values, cmap=cmap, vmin=color_min, vmax=color_max, edgecolors=edge_color,
                             linewidths=0.5, label=slabel, alpha=0.8, zorder=2)

        # --------------------------------------------------------------------------------------------------------------
        # COLORBAR
        # --------------------------------------------------------------------------------------------------------------
        cbar = fig.colorbar(scatter, ax=ax, orientation="horizontal", location="top", pad=0.08)
        # Evenly spaced ticks over the complete variable range
        if color_min == color_max:
            ticks = np.array([color_min])
        else:
            ticks = np.linspace(color_min, color_max, colorbar_ticks)
        cbar.set_ticks(ticks)

        # Explicit tick labels
        cbar.set_ticklabels([f"{tick:.1f}" for tick in ticks])
        # Colorbar label
        cbar.set_label(colorbar_label if colorbar_label else color_column, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

    else:
        # Original single-color behavior
        scatter = ax.scatter(xVal, yVal, color=sColor, edgecolors=edge_color, linewidths=0.5, label=slabel, alpha=0.8,
                             zorder=2)

    # ------------------------------------------------------------------------------------------------------------------
    # REGRESSION + R-squared
    # ------------------------------------------------------------------------------------------------------------------
    X = np.asarray(xVal).reshape(-1, 1)
    y = np.asarray(yVal)
    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)
    sort_idx = np.argsort(np.asarray(xVal))
    predicted = model.predict(X)
    ax.plot(np.asarray(xVal)[sort_idx], predicted[sort_idx], ls="-.", color=rColor, linewidth=lw,
            label=fr"$R^2 = {r_squared:.2f}$")

    # ------------------------------------------------------------------------------------------------------------------
    # LIMITS
    # ------------------------------------------------------------------------------------------------------------------
    if limit is None:
        all_vals = np.concatenate([np.asarray(xVal), np.asarray(yVal)])
        # Log axes require positive limits
        if (xScale == "log" or yScale == "log"):
            all_vals = all_vals[all_vals > 0]
        min_lim = np.nanmin(all_vals)
        max_lim = np.nanmax(all_vals)
        limit = (min_lim, max_lim)

    # ------------------------------------------------------------------------------------------------------------------
    # 1:1 LINE
    # ------------------------------------------------------------------------------------------------------------------
    line_vals = np.asarray(limit)
    ax.plot(line_vals, line_vals, ls="--", color=lColor, linewidth=lw, label="1:1 Line")

    # ------------------------------------------------------------------------------------------------------------------
    # FACTOR BOUNDS
    # ------------------------------------------------------------------------------------------------------------------
    if factor is not None:
        factor_value = factor[0]
        ax.plot(line_vals, line_vals / factor_value, ls=":", color="black", linewidth=lw, label=(f"Factor of {factor[0]}: {factor[1]}%"))
        ax.plot(line_vals, line_vals * factor_value, ls=":", color="black", linewidth=lw)

    # ------------------------------------------------------------------------------------------------------------------
    # AXIS SETTINGS
    # ------------------------------------------------------------------------------------------------------------------
    ax.set_xscale(xScale)
    ax.set_yscale(yScale)
    ax.set_xlim(limit)
    ax.set_ylim(limit)

    # ------------------------------------------------------------------------------------------------------------------
    # AXIS LABELS
    # ------------------------------------------------------------------------------------------------------------------
    xlabel_final = xlabel
    ylabel_final = ylabel
    if xunit:
        xlabel_final += f" ({xunit})"
    if yunit:
        ylabel_final += f" ({yunit})"

    ax.set_xlabel(xlabel_final, fontsize=fontsize)
    ax.set_ylabel(ylabel_final, fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)

    # ------------------------------------------------------------------------------------------------------------------
    # GRID
    # ------------------------------------------------------------------------------------------------------------------
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # ------------------------------------------------------------------------------------------------------------------
    # LEGEND
    # ------------------------------------------------------------------------------------------------------------------
    fontP = FontProperties()
    fontP.set_size(fontsize)
    leg = ax.legend(loc=legendloc, prop=fontP)
    leg.get_frame().set_edgecolor("black")
    fig.tight_layout()

    # ------------------------------------------------------------------------------------------------------------------
    # SAVE FIGURE
    # ------------------------------------------------------------------------------------------------------------------
    if fileName is not None:
        save_dir = (outputFilePath if outputFilePath else os.getcwd())
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, fileName)
        print(f"Saving figure to:\n{save_path}")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax

def histogram(data, unit="", fig=None, axes=None, ci=(2.5, 97.5), figsize=(3.54331, 3.54331/1.5), bins=0.5,
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
        - bins (int, optional): Bin width (default: 0.5).
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

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
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
        mean_val = np.mean(data, dtype=np.float64) # Mean value
        ci_upper = np.percentile(data, ci[1])  # Upper bound of CI
        vline_positions = [ci_lower, mean_val, ci_upper]

        # Plot CI and mean lines
        linestyle = ["--", ":", "--"]
        # Labels for the vertical lines
        labels = [f"95% CI: [{round(ci_lower, 2)}, {round(ci_upper, 2)}]", f"Mean: {round(mean_val, 2)}", None]
        for position, ls, label in zip(vline_positions, linestyle, labels):
            ax.axvline(x=position, color='red', linewidth=lw, linestyle=ls, label=label)

        # Plot the histogram
        min_val, max_val = np.floor(np.min(data)), np.ceil(np.max(data))
        bin_edges = np.arange(min_val, max_val + bins, bins)
        ax.hist(data, bins=bin_edges, color=color, edgecolor=edgecolor, label=None)

        # np.savetxt(os.path.join(outputFilePath or os.getcwd(), "TTD.csv"), np.array(data), delimiter=",", fmt="%.18g")

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
                      xunit="Days", yunit=r"$\mathrm{kg~CH_4}$", color1="blue", color2="orange", xrotation=45):
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

    Returns:
    - fig, ax: The figure and axis objects for further customization.
    """
    # Setup figure and axes
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize) if figsize else plt.subplots()
    else:
        ax = axes if axes is not None else plt.gca()

    # Respect IncludeAll and avoid mutating data later used in other figures.
    CRDF, DRDF = CRDF.copy(), DRDF.copy()
    # if not aggQuant_all:
    #     exclude_pattern = r'Offline Record: Data Excluded|Maintenance Record: Data Excluded|Passive Observation: Data Excluded'
    #     CRDF = CRDF.loc[~CRDF['tc_Classification'].astype(str).str.contains(exclude_pattern, na=False)].copy()
    #     DRDF = DRDF.loc[~DRDF['dr_Classification'].astype(str).str.contains(exclude_pattern, na=False)].copy()

    exclude_pattern = r'Offline Record: Data Excluded|Maintenance Record: Data Excluded'
    CRDF = CRDF.loc[~CRDF['tc_Classification'].astype(str).str.contains(exclude_pattern, na=False)].copy()
    DRDF = DRDF.loc[~DRDF['dr_Classification'].astype(str).str.contains(exclude_pattern, na=False)].copy()

    # Preparing data
    CRDF.loc[:, time] = pd.to_datetime(CRDF[time], errors="coerce")
    DRDF.loc[:, time] = pd.to_datetime(DRDF[time], errors="coerce")
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
    tc_total = tc_emission_cummulative.iloc[-1] if len(tc_emission_cummulative) else 0.0
    dr_total = dr_emission_cummulative.iloc[-1] if len(dr_emission_cummulative) else 0.0

    # Calculate percent difference
    percent_difference = ((dr_total - tc_total) / tc_total) * 100 if tc_total else float('nan')

    # Plot cumulative emissions
    ax.plot(CRDF[time].tolist(), tc_emission_cummulative, label=f"Test center total: {tc_total:.1f} {yunit}",
            linewidth=lw, linestyle="--", color=color1)
    ax.plot(DRDF[time].tolist(), dr_emission_cummulative, label=f"Estimated total: {dr_total:.1f} {yunit}",
            linewidth=lw, linestyle="--", color=color2)

    # # Kolmogorov-Smirnov Test: For comparing the overall shape of the distributions, not just the means
    # stat, p_value = ks_2samp(tc_emission_cummulative, dr_emission_cummulative)
    # if p_value < 0.05:
    #     print(f'Kolmogorov-Smirnov Test shows that the p-value: {p_value} and is statistically significant')
    # else:
    #     print(f'Kolmogorov-Smirnov Test shows that the p-value: {p_value} and is not statistically significant')

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
    # Log-space evaluation avoids intermediate overflow. Infinity is a valid
    # indication that the fitted detection limit is numerically unbounded.
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        return np.exp((np.log(-np.log1p(-y)) - np.log(a)) / b)

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

    if not 0 < detectionlimit < 1 or Nbootstrap < 1 or step <= 0:
        raise ValueError('POD requires 0 < detectionlimit < 1, Nbootstrap >= 1, and step > 0.')
    if CRDF.empty or CRDF['Detection'].nunique() < 2:
        raise ValueError('POD curve fitting requires both TP and FN observations.')
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

    if param is None or not np.all(np.isfinite(param)):
        raise ValueError('No valid POD fit; check data and fitting bounds.')
    # Perform bootstrap analysis
    print("Performing bootstrap analysis...")

    # Fit each bootstrap draw once. Previously a second independent draw was
    # used to decide whether to keep the first, allowing failed fits through.
    bootstrap_results = []
    for _ in range(Nbootstrap):
        sample_rate, sample_class = resample(releaserate, classification)
        fitted, _ = optimizeCurve(a0s, b0s, bounds, sample_rate, sample_class)
        if fitted is not None and np.all(np.isfinite(fitted)):
            bootstrap_results.append(fitted)
    if not bootstrap_results:
        raise ValueError('No valid POD bootstrap fits; check data and fitting bounds.')

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
    # Preserve infinite bootstrap limits. An order-statistic percentile avoids
    # undefined interpolation such as inf - inf and does not discard uncertainty.
    DL5bound, DL95bound = np.quantile(DL90s, [0.05, 0.95], method='inverted_cdf')
    DL90 = float(InversemSigmoidModel(detectionlimit, param[0], param[1]))
    if np.isfinite(DL90):
        ax.axvline(DL90, color='k', linestyle=':', lw=0.8, zorder=2,
                   label=f"{percentage} POD: {DL90:.{digits}f} [{DL5bound:.{digits}f}, {DL95bound:.{digits}f}]")
        ax.axhline(detectionlimit, color='k', linestyle=':', lw=0.8)
    else:
        ax.axhline(detectionlimit, color='k', linestyle=':', lw=0.8,
                   label=f"{percentage} POD: unbounded fitted detection limit")

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


import os
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from matplotlib.font_manager import FontProperties

def scatter(df, xCategory="ControlledReleaseRate", yCategory="Capture fraction",
            classificationCategory="Classification", tp_only=True, xlabel="Controlled release rate", xunits=None,
            ylabel="Fraction of release duration detected", xScale="linear", xmin=None, xmax=None, figsize=(6, 4),
            fontsize=9, marker_size=20, alpha=0.6, ax=None, fileName="capture_fraction.png", filePath=None, dpi=400):
    """
        Scatterplot of controlled release rate versus capture fraction.
        Capture fraction is assumed to be stored as percent (0-100) and is converted to decimal (0-1).
        By default, only TP observations are plotted.

        Legend reports:
            - mean capture fraction
            - median capture fraction
            - 2.5th percentile
            - 97.5th percentile

        Parameters
        ----------
        df (pandas.DataFrame) : Input dataframe.
        xCategory (str) : Column containing controlled release rate.
        captureCategory (str) : Column containing capture fraction in percent.
        classificationCategory (str) : Column containing TP/FN/FP/TN classification.
        tp_only (bool) : If True, only rows classified as TP are included.
        xlabel, ylabel (str) : Axis labels.
        xunits (str or None) : Units appended to x-axis label.
        xScale (str) : "linear" or "log".
        xmin, xmax (float or None) : Optional x-axis limits.
        figsize (tuple) : Figure size.
        fontsize (float) : Base font size.
        marker_size (float) : Scatter marker size.
        alpha (float) : Scatter transparency.
        ax : matplotlib.axes.Axes or None
            Existing axis if desired.

        Returns:
            fig : matplotlib.figure.Figure
            ax : matplotlib.axes.Axes
    """

    data = df.copy()
    # ------------------------------------------------------------------------------------------------------------------
    # Keep TP observations only, if requested
    # ------------------------------------------------------------------------------------------------------------------
    if tp_only:
        data = data.loc[data[classificationCategory].eq("TP")].copy()

    # ------------------------------------------------------------------------------------------------------------------
    # Keep required columns
    # ------------------------------------------------------------------------------------------------------------------
    data = data[[xCategory, yCategory]].copy()
    # Convert to numeric
    data[xCategory] = pd.to_numeric(data[xCategory], errors="coerce")
    data[yCategory] = pd.to_numeric(data[yCategory], errors="coerce")
    d90 = data.loc[data[yCategory] >= 90]
    print(f"Fraction of TP detections with capture fraction of at least 90%: {round(len(d90) * 100 / len(data), 2)}")
    # Remove missing values
    data.dropna(subset=[xCategory, yCategory], inplace=True)
    # Log axis requires positive x values
    if xScale == "log":
        data = data.loc[data[xCategory] > 0].copy()

    # ------------------------------------------------------------------------------------------------------------------
    # Convert percent -> decimal
    # ------------------------------------------------------------------------------------------------------------------
    data["CaptureFractionDecimal"] = (data[yCategory] / 100.0)

    # ------------------------------------------------------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------------------------------------------------------
    capture = data["CaptureFractionDecimal"].to_numpy()
    mean_capture = np.mean(capture)
    median_capture = np.median(capture)
    p2_5 = np.percentile(capture, 2.5)
    p97_5 = np.percentile(capture, 97.5)

    # ------------------------------------------------------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ------------------------------------------------------------------------------------------------------------------
    # Scatterplot
    # ------------------------------------------------------------------------------------------------------------------
    ax.scatter(data[xCategory], data["CaptureFractionDecimal"], s=marker_size, alpha=alpha, facecolors="blue",
               edgecolors="k", linewidths=0.8)

    # ------------------------------------------------------------------------------------------------------------------
    # Add invisible handles so statistics appear in legend
    # ------------------------------------------------------------------------------------------------------------------
    ax.plot([], [], linestyle="none", label=f"Mean = {mean_capture:.2f}")
    ax.plot([], [], linestyle="none", label=f"Median = {median_capture:.2f}")
    ax.plot([], [], linestyle="none", label=f"2.5th percentile = {p2_5:.2f}")
    ax.plot([], [], linestyle="none", label=f"97.5th percentile = {p97_5:.2f}")

    # ------------------------------------------------------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------------------------------------------------------
    ax.set_xscale(xScale)
    ax.set_ylim(-0.02, 1.02)
    if xmin is not None:
        ax.set_xlim(left=xmin)
    if xmax is not None:
        ax.set_xlim(right=xmax)
    if xunits:
        x_label = f"{xlabel} ({xunits})"
    else:
        x_label = xlabel

    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid(alpha=0.1)

    # ------------------------------------------------------------------------------------------------------------------
    # 16. Legend
    # ------------------------------------------------------------------------------------------------------------------
    fontP = FontProperties()
    fontP.set_size(fontsize)
    leg = ax.legend(loc="best", prop=fontP, frameon=True)
    leg.get_frame().set_linewidth(0.5)

    # ------------------------------------------------------------------------------------------------------------------
    # 17. Save plot
    # ------------------------------------------------------------------------------------------------------------------
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)

        print(f"saving: {path}")
        fig.savefig(path, dpi=dpi, bbox_inches='tight')

    return fig, ax


def LOGISTICREGRESSION(Df, filePath=None, fileName=None, desiredLDLFraction=0.90, Nbootstrap=1000, fontsize=7,
                       xCategory='tc_EPBFE', yCategory='Classification', xlabel='Whole gas emission rate',
                       ThresholdRateAboveBaseline=None, xunits='slpm', figsize=None, fig=None, axes=None,
                       ScaleFactor=1, ylabel="Probability of Detection (POD)", borderpad=0.4, eps=1e-6,
                       xScale="linear", yScale="linear", xmax=None, digits=1, xmin=None, mColors='b',
                       bColors='#B0C4DE', loc='upper left', baseline=0, dpi=400, ci=0.95, n_curve_points=1000,
                       bootstrap_alpha=0.10, random_state=None):
    """
        Perform logistic regression with bootstrap uncertainty estimation
        for a Probability of Detection (POD) curve.

        Parameters
        ----------
        Df (pandas.DataFrame) : Input dataframe containing the predictor and classification columns.
        filePath (str, optional) : Directory in which to save the plot.
        fileName (str, optional) : Filename for the saved plot.
        desiredLDLFraction (float, default=0.90) : Desired POD level for determination of the detection limit.
        Nbootstrap (int, default=1000) : Number of bootstrap resamples.
        fontsize (int, default=7) : Plot font size.
        xCategory (str) : Name of the independent-variable column.
        yCategory (str) : Name of the classification column.
        xlabel (str) : X-axis label.
        xunits (str) : X-axis units.
        figsize (tuple, optional) : Figure size.
        fig, axes (matplotlib objects, optional) : Existing figure and axes.
        ScaleFactor (float, default=1) : Scaling factor applied to xCategory.
        ylabel (str) : Y-axis label.
        borderpad (float) : Legend border padding.
        eps (float) : Small positive value used to avoid log(0).
        xScale (str) : X-axis scale, e.g. "linear" or "log".
        yScale (str) : Y-axis scale.
        xmax, xmin (float, optional) : X-axis limits.
        digits (int) : Number of decimal places used when displaying DL values.
        mColors: Color for the main logistic regression curve.
        bColors: Color for bootstrap curves.
        loc (str) : Legend location.
        baseline (float) : Baseline value subtracted from xCategory.
        dpi (int) : Saved figure resolution.
        ci (float, default=0.95) : Bootstrap confidence level.
        n_curve_points (int, default=1000) : Number of points used to evaluate each fitted POD curve.
        bootstrap_alpha (float, default=0.10) : Transparency of individual bootstrap curves.
        random_state (int, optional) : Random seed for reproducible bootstrap sampling.

        Returns
            fig : matplotlib.figure.Figure
                  Generated figure.
            ax : matplotlib.axes.Axes
                 Generated axes.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # 1. Figure / axes
    # ------------------------------------------------------------------------------------------------------------------

    plt.rcParams.update({'font.size': fontsize})

    if fig is None:
        if figsize:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()
    else:
        ax = axes if axes is not None else plt.gca()

    # ------------------------------------------------------------------------------------------------------------------
    # 2. Prepare data
    # ------------------------------------------------------------------------------------------------------------------

    # Only TP and FN observations are used for POD modeling
    CRDF = Df.loc[Df[yCategory].isin(['TP', 'FN'])].copy()
    CRDF.reset_index(drop=True, inplace=True)

    # Prevent duplicated column names from causing problems
    CRDF = CRDF.loc[:, ~CRDF.columns.duplicated()]

    # Keep only required columns
    logisticDf = CRDF[[xCategory, yCategory]].copy()

    # TP = detected = 1
    # FN = not detected = 0
    logisticDf['Detection'] = (logisticDf[yCategory].eq('TP').astype(int))

    # Make sure predictor is numeric
    logisticDf[xCategory] = pd.to_numeric(logisticDf[xCategory], errors='coerce')

    # Remove rows with invalid predictor values
    logisticDf.dropna(subset=[xCategory, 'Detection'], inplace=True)

    # Apply baseline correction and scaling
    logisticDf[xCategory] = (logisticDf[xCategory] - baseline) * ScaleFactor

    # Logistic model uses ln(x), so x must be positive.
    logisticDf = logisticDf.loc[logisticDf[xCategory] > 0].copy()
    logisticDf.reset_index(drop=True, inplace=True)

    # Need both TP and FN for logistic regression
    if logisticDf['Detection'].nunique() < 2:
        raise ValueError("Logistic regression requires at least one TP and at least one FN observation.")

    if len(logisticDf) < 2:
        raise ValueError("Insufficient observations for logistic regression.")

    # ------------------------------------------------------------------------------------------------------------------
    # 3. Main logistic regression model
    # ------------------------------------------------------------------------------------------------------------------
    xDatafield = logisticDf[xCategory].to_numpy(dtype=float)
    Y = logisticDf['Detection'].to_numpy(dtype=int)
    minFlowRate = np.min(xDatafield)
    maxFlowRate = np.max(xDatafield)

    # Model predictor = ln(x)
    X = np.log(np.maximum(xDatafield, eps)).reshape(-1, 1)
    logReg = LogisticRegression(solver='lbfgs')
    logReg.fit(X, Y)
    modelCoef = logReg.coef_[0, 0]
    modelIntercept = logReg.intercept_[0]
    param = [modelIntercept, modelCoef]

    # Main DL estimate
    ldl = invlogisticRegression(p=desiredLDLFraction, B=param)

    # ------------------------------------------------------------------------------------------------------------------
    # 4. Prediction grid
    # ------------------------------------------------------------------------------------------------------------------
    # Use a fixed number of plotting points.
    # Do NOT use Nbootstrap as the number of x-values.
    plot_min = minFlowRate
    plot_max = maxFlowRate

    # If user supplied valid limits, use them
    if xmin is not None and xmin > 0:
        plot_min = xmin
    if xmax is not None and xmax > plot_min:
        plot_max = xmax

    if xScale == "log":
        xMain = np.logspace(np.log10(max(plot_min, eps))-2, np.log10(plot_max)+2, n_curve_points)

    else:
        xMain = np.linspace(plot_min, plot_max, n_curve_points)

    xMain_log = np.log(np.maximum(xMain, eps)).reshape(-1, 1)
    yMain = logReg.predict_proba(xMain_log)[:, 1]

    # ------------------------------------------------------------------------------------------------------------------
    # 5. Bootstrap containers
    # ------------------------------------------------------------------------------------------------------------------
    rng = np.random.default_rng(random_state)
    n_obs = len(logisticDf)

    # Each COLUMN corresponds to one bootstrap model
    Ymc = np.full((len(xMain), Nbootstrap), np.nan, dtype=float)

    # Bootstrap DL values
    LDLmc = np.full(Nbootstrap, np.nan, dtype=float)

    # Row 0 = intercept
    # Row 1 = coefficient
    Bmc = np.full((2, Nbootstrap), np.nan, dtype=float)

    # ------------------------------------------------------------------------------------------------------------------
    # 6. Bootstrap logistic regressions
    # ------------------------------------------------------------------------------------------------------------------
    successful_bootstraps = 0
    for i in range(Nbootstrap):
        # Sample WITH replacement from indices 0 ... n_obs-1
        bootstrap_idx = rng.integers(low=0, high=n_obs, size=n_obs)
        bootstrapDf = logisticDf.iloc[bootstrap_idx]
        ybot = bootstrapDf['Detection'].to_numpy(dtype=int)

        # Skip samples that contain only TP or only FN
        if np.unique(ybot).size < 2:
            continue

        xbot_values = bootstrapDf[xCategory].to_numpy(dtype=float)
        xbot = np.log(np.maximum(xbot_values, eps)).reshape(-1, 1)

        try:
            botmodel = LogisticRegression(solver='lbfgs')
            botmodel.fit(xbot, ybot)
            botCoef = botmodel.coef_[0, 0]
            botIntercept = botmodel.intercept_[0]
            botB = [botIntercept, botCoef]
            # Bootstrap DL
            boot_ldl = invlogisticRegression(p=desiredLDLFraction, B=botB)
            # Keep only finite estimates
            if np.isfinite(boot_ldl):
                LDLmc[i] = boot_ldl
            Bmc[0, i] = botIntercept
            Bmc[1, i] = botCoef

            # IMPORTANT:
            # one entire logistic curve is stored in COLUMN i
            Ymc[:, i] = botmodel.predict_proba(xMain_log)[:, 1]
            successful_bootstraps += 1

        except Exception:
            continue

    # ------------------------------------------------------------------------------------------------------------------
    # 7. Confidence interval for DL
    # ------------------------------------------------------------------------------------------------------------------
    valid_ldl = LDLmc[np.isfinite(LDLmc)]
    alpha = 1 - ci
    if len(valid_ldl) > 0:
        LowerCL = np.nanpercentile(valid_ldl, 100 * alpha / 2)
        UpperCL = np.nanpercentile(valid_ldl, 100 * (1 - alpha / 2))
    else:
        LowerCL = np.nan
        UpperCL = np.nan

    # ------------------------------------------------------------------------------------------------------------------
    # 8. Equation / legend label
    # ------------------------------------------------------------------------------------------------------------------
    denominator = ('1 + e^{(%s%s*\\ln x)}'% (signed_fmt(-param[0]), signed_fmt(-param[1])))
    equation = r'$\frac{1}{%s}$' % denominator
    if (np.isfinite(ldl) and ldl >= minFlowRate and ldl <= maxFlowRate):
        dl = latex_float(round(ldl, digits))
        if (np.isfinite(LowerCL) and np.isfinite(UpperCL)):
            lowLDL = latex_float(round(LowerCL, digits))
            upLDL = latex_float(round(UpperCL, digits))
            lab = (f'{dl} 'f'[{lowLDL}, {upLDL}]')
        else:
            lab = f'{dl}'
        label = (f'DL{int(desiredLDLFraction * 100)}: 'f'{lab}')
    else:
        label = (f'DL{int(desiredLDLFraction * 100)} ''beyond range')

    # ------------------------------------------------------------------------------------------------------------------
    # 9. Plot observations
    # ------------------------------------------------------------------------------------------------------------------
    ax.scatter(logisticDf[xCategory], logisticDf['Detection'], label='TP & FN', s=5, facecolors='none', edgecolors='k',
               zorder=3)

    # ------------------------------------------------------------------------------------------------------------------
    # 10. Plot bootstrap curves
    # ------------------------------------------------------------------------------------------------------------------
    for i in range(Nbootstrap):
        # Ignore unsuccessful fits
        if np.isnan(Ymc[:, i]).all():
            continue

        # plot COLUMN i, not ROW i
        ax.plot(xMain, Ymc[:, i], ls='solid', lw=0.5, c=bColors, marker='None', alpha=bootstrap_alpha, zorder=1)

    # ------------------------------------------------------------------------------------------------------------------
    # 11. Plot main logistic regression curve
    # ------------------------------------------------------------------------------------------------------------------
    ax.plot(xMain, yMain, ls='solid', lw=1, c=mColors, marker='None', label=equation, zorder=4)

    # ------------------------------------------------------------------------------------------------------------------
    # 12. Plot desired detection threshold
    # ------------------------------------------------------------------------------------------------------------------
    if np.isfinite(ldl):
        # Horizontal DL probability line
        ax.plot([plot_min, ldl], [desiredLDLFraction, desiredLDLFraction], ls=':', lw=1.0, c='k', marker='None',
                zorder=4)
        # Vertical DL line
        ax.plot([ldl, ldl], [0, desiredLDLFraction], ls=':', lw=1.0, c='k', marker='None', zorder=4, label=label)

    # if not pd.isna(ThresholdRateAboveBaseline):
    #     ax.axvline(x=ThresholdRateAboveBaseline * ScaleFactor, color='red', linestyle='--', linewidth=1.0, label="Detection Threshold")
    # ------------------------------------------------------------------------------------------------------------------
    # 13. Axes
    # ------------------------------------------------------------------------------------------------------------------
    ax.set_xscale(xScale)
    ax.set_yscale(yScale)
    ax.set_ylim([-0.05, 1.05])

    if xmin is not None and xmax is not None:
        ax.set_xlim([xmin, xmax])

    elif xmin is not None:
        ax.set_xlim(left=xmin)

    elif xmax is not None:
        ax.set_xlim(right=xmax)

    else:
        ax.set_xlim([plot_min, plot_max])

    # ------------------------------------------------------------------------------------------------------------------
    # 14. Grid
    # ------------------------------------------------------------------------------------------------------------------
    ax.grid(axis='both', alpha=0.05)

    # Optional logarithmic decade lines
    if xScale == "log":
        current_xlim = ax.get_xlim()
        positive_min = max(current_xlim[0], eps)
        positive_max = current_xlim[1]
        if positive_max > positive_min:
            a_min = int(np.ceil(np.log10(positive_min)))
            a_max = int(np.floor(np.log10(positive_max)))
            for exponent in range(a_min, a_max + 1):
                ax.axvline(x=10 ** exponent, c='k', linestyle='--', linewidth=0.03, zorder=0)


    # ------------------------------------------------------------------------------------------------------------------
    # 15. Labels
    # ------------------------------------------------------------------------------------------------------------------
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if xlabel:
        ax.set_xlabel(f"{xlabel} ({xunits})", fontsize=fontsize)

    # ------------------------------------------------------------------------------------------------------------------
    # 16. Legend
    # ------------------------------------------------------------------------------------------------------------------
    fontP = FontProperties()
    fontP.set_size(fontsize)
    leg = ax.legend(loc=loc, prop=fontP, borderpad=borderpad)
    leg.get_frame().set_linewidth(0.5)

    # ------------------------------------------------------------------------------------------------------------------
    # 17. Save plot
    # ------------------------------------------------------------------------------------------------------------------
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)

        print(f"saving: {path}")
        fig.savefig(path, dpi=dpi, bbox_inches='tight')

    return fig, ax


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
