import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, auc

from scripts_thesis.utils import *
from scripts_thesis.charter import *
from scripts_thesis.roc_display import RocCurveDisplayPlotly

import numpy as np
import pandas as pd

custom_params = {"axes.spines.right": False, "axes.spines.top": False, "figure.figsize":(8, 8)}
sns.set_theme(context='notebook', style='darkgrid', palette='deep', rc= custom_params)

hex_colors = [mcolors.to_hex(color) for color in sns.diverging_palette(145, 300, s=60, n=5)]
hex_colors.reverse()


# color constants and palettes
CORAL = ["#D94535", "#ed8a84", "#f3b1ad", "#f9d8d6"]
DARK_GREY = ["#323C46", "#77949d", "#a4b7be", "#d2dbde"]
STONE = ["#7d8791", "#a5aaaa", "#d2d2d2", "#ebebeb"]
BLUE_GREY = ["#94b7bb", "#bfd4d6", "#d4e2e4", "#eaf1f1"]
BROWN = ["#7a6855", "#b4a594", "#cdc3b8", "#e6e1db"]
PURPLE = ["#8d89a5", "#bbb8c9", "#d1d0db", "#e8e7ed"]
PALETTE = CORAL + BLUE_GREY[::-1]


# matplotlib and seaborn :


def charter_graph(width=12, length=8):
    fig, ax = plt.subplots()
    fig.set_size_inches(width, length)
    plt.rcParams.update({"font.size": 12})
    plt.rcParams["font.family"] = "FuturaTOT"
    ax.patch.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(DARK_GREY[1])
    ax.spines["left"].set_color(DARK_GREY[1])
    ax.tick_params(colors=DARK_GREY[1])
    # ax.tick_params(axis='x', labelrotation=90)
    ax.spines["bottom"].set_color(DARK_GREY[2])
    ax.spines["left"].set_color(DARK_GREY[2])
    return fig, ax


def charter_multiple(width=12, length=8, rows=1, cols=1):
    fig, axes = plt.subplots(rows, cols)
    fig.set_size_inches(width, length)
    plt.rcParams.update({"font.size": 14})
    plt.rcParams["font.family"] = "FuturaTOT"
    for ax in axes:
        ax.patch.set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(DARK_GREY[0])
        ax.spines["left"].set_color(DARK_GREY[0])
        ax.tick_params(colors=DARK_GREY[0])
        ax.spines["bottom"].set_color(DARK_GREY[2])
        ax.spines["left"].set_color(DARK_GREY[2])
    return fig, axes


# plotly :
def charter_plotly(fig, width=1000, height=600, title_place=0.3) -> go.Figure:
    fig.update_xaxes(
        title_font=dict(size=24, family="FuturaTOT", color=DARK_GREY[1]),
        title_font_family="FuturaTOT",
        color=DARK_GREY[2],
        linecolor=DARK_GREY[2],
        tickfont=dict(size=12),
        showgrid=False,
    )
    fig.update_yaxes(
        title_font=dict(size=24, family="FuturaTOT", color=DARK_GREY[1]),
        title_font_family="FuturaTOT",
        color=DARK_GREY[2],
        linecolor=DARK_GREY[2],
        tickfont=dict(size=12),
        showgrid=False,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        width=width,
        height=height,
        title=dict(
            x=title_place, font=dict(family="FuturaTOT", size=28, color=DARK_GREY[1]),
        ),
        legend=dict(font=dict(family="FuturaTOT", size=24, color=DARK_GREY[1],)),
    )
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, width: int= 600, height: int= 600) -> go.Figure:
    """
    Convenience function to print a confusion matrix with the predicted results y_pred

    Parameters
    ----------
    y_true : np.array
        Array of real data
    y_pred : np.array
        Array of predicted data
    width : int, optional
        dimension of the image, by default 400
    height : int, optional
        dimension of the image, by default 400

    Returns
    -------
    fig : go.Figure
        A confusion matrix of the model's result
    """

    labels = sorted(list(set(y_true)))
    df_lambda = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index = labels,
        columns = labels
    )

    df_lambda = df_lambda/len(y_true) #Get results as proportion of total results
    df_lambda = df_lambda.apply(lambda value : np.round(value, 2))

    acc = accuracy_score(y_true, y_pred)
    f1s = f1_score(y_true, y_pred, average = 'weighted')
    precision = precision_score(y_true, y_pred, average = 'weighted')

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
                   z=df_lambda.values,
                   x=[str(col) for col in df_lambda.columns],
                   y=[str(col) for col in df_lambda.columns],
                   colorscale='RdBu_r',
                   text=df_lambda.values,
                   texttemplate="%{text}%",
                   textfont={"size":16},
                   colorbar=dict(title='Proportion'),
                   hoverongaps = False)
                   )

    fig.update_layout(
        title=f'Confusion Matrix : Overall results <br><sup>Accuracy: {acc:.2f}, F1: {f1s:.2f}, Precision: {precision:.2f}</sup>', #<br> is a line break, and <sup> is superscript
        xaxis=dict(title='Predicted'),
        yaxis=dict(title='Actual'),
        width=width,
        height=height,
        title_font=dict(
            family='Arial',
            color='black',
            size=20
        ),
        font=dict(family='Arial', color='black')
        )

    return fig


def model_explainer(df: pd.DataFrame, x: str= "coef", y: str= "feature")-> go.Figure:
    """
    A function that takes the dataframe outputed by get_top_features() and creates a barplot of most important features using Plotly

    Parameters
    ----------
    df : pd.DataFrame
        a dataframe with a categorical column (feature) and a numerical column (coef)
    x : str, optional
        column for the x-axis of the graph, by default "coef"
    y : str, optional
        column for the y-axis of the graph, by default "feature"

    Returns
    -------
    fig : go.Figure
        A plotly bar graph showing the most important coeficients and their values
    """

    colors = px.colors.qualitative.Dark24[:20]
    template = 'CLF: %{customdata}<br>Feature: %{y}<br>Coefficient: %{x:.2f}'

    fig = px.bar(
        data_frame = df,
        x = x,
        y = y,
        facet_col_wrap = 3,
        facet_col_spacing = .15,
        height = 1200,
        labels = {
            'coef': 'Coefficient',
            'feature': ''
        },
        title = f'Top {len(df)} Strongest Predictors by SDG'
    )

    fig.for_each_trace(lambda x: x.update(hovertemplate = template))
    fig.for_each_trace(lambda x: x.update(marker_color = colors.pop(0)))
    fig.update_yaxes(matches = None, showticklabels = True)
    fig.show()

    return fig


def model_dl_examiner(train: np.ndarray, val: np.ndarray) -> go.Figure:
    """
    A convenience function to produce the train and val loss for a trained TF model

    Parameters
    ----------
    train : np.ndarray
        The history of train losses from a trained model
    val : np.ndarray
        The history of val losses from a trained model

    Returns
    -------
    fig : go.Figure
        A plotly line chart
    """

    assert len(train) == len(val)

    x = np.arange(len(train))

    fig = go.Figure()

    for color, name, line in zip([hex_colors[0], hex_colors[-1]], ["Train results", "Val results"], [train, val]):
        fig.add_trace(
            go.Scatter(x=x, y=line, mode='lines', name=name, marker_color = color)
            )

    fig.update_layout(
        xaxis=dict(title='Epochs'),
        yaxis=dict(title='Loss: Binary cross-entropy'),
        title="Train and val losses across epochs",
        legend=dict(
        orientation="v",
        yanchor="bottom",
        y=0.8,
        xanchor="right",
        bgcolor="LightSteelBlue",
        x=0.98)
        )

    fig.show()


def _auc_curve(test_array: np.ndarray, target_array: np.ndarray, n_splits: int, **kwargs) -> go.Figure:
    """
    A function used to calculate cross_validated auc.

    Used as part of auc_cross_val

    Parameters
    ----------
    test_array : np.ndarray
        The array of test values, needs to be of dimensions (n, m) with n the number of folds and m the length of the test data
    target_array : np.ndarray
        The array of predicted values, can be binary or probabilities
        Needs to be of dimensions (n, m) with n the number of folds and m the length of the test data
    n_splits : int
        Number of splits in the cross validation

    Returns
    -------
    fig : go.Figure
        The cross validated AUC curves in 1 plotly figure
    """

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig = go.Figure()
    mask = len(test_array) - 1

    for fold, (y_test, y_pred) in enumerate(zip(test_array, target_array)):

        viz = RocCurveDisplayPlotly.from_predictions(y_test, y_pred,
                                                     pos_label=1, fold=fold,
                                                     n_splits=n_splits,
                                                     plot_chance_level=(mask == fold),
                                                     fig=fig,
                                                     show_fig=False)

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr) # Interpolates additional points for the curve
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    #std_auc = np.std(aucs)

    fig.add_trace(go.Scatter(x=mean_fpr,
                                y=mean_tpr,
                                mode='lines',
                                line = dict(color=hex_colors[-1], width=2, dash='dot'),
                                name=f"Mean ROC (AUC = {np.round(mean_auc, 2)})"
                                )
                )

    #Adding a confidence interval
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    fig.add_trace(go.Scatter(x=mean_fpr,
                                y=tprs_upper,
                                mode='lines',
                                line_color='grey',
                                name="Lower std bound",
                                showlegend=False
                                )
        )
    fig.add_trace(go.Scatter(x=mean_fpr,
                                y=tprs_lower,
                                mode='lines',
                                fill='tonexty',
                                line_color='grey',
                                name="CI interval"
                                )
        )

    fig.update_layout(
        xaxis=dict(range=[-0.05, 1.05], title="False Positive Rate"),
        yaxis=dict(range=[-0.05, 1.05], title="True Positive Rate"),
        title="ROC curves with variability\n (CV)",
        legend=dict(title_text='ROC results',
                    yanchor="bottom",
                    y=0.0,
                    xanchor="left",
                    x= 1.05,
                    )
        )

    width = kwargs.get("width", 950)
    height = kwargs.get("height", 600)

    fig = charter_plotly(fig, width=width, height=height)

    return fig

def auc_cross_val(auc_metrics: tuple, n_splits: int= 5):
    """
    A function to produce a figure with several ROC curves when cross-validating a model

    Parameters
    ----------
    auc_metrics : tuple
        A tuple containing lists of lists for respectively y_test, y_pred and y_pred proba by cv fold
    n_splits : int, optional
        Number of folds by cv, by default 5

    Returns
    -------
    tuple
        A tuple containing one or two plotly graphs depending on the type of classifiers used
    """

    if len(auc_metrics) == 2:
        test_array, pred_array = auc_metrics
        return (_auc_curve(test_array, pred_array, n_splits=n_splits), None) #A half empty tuple

    elif len(auc_metrics) == 3:
        test_array, pred_array, proba_array = auc_metrics
        proba_array = [arr[:, 1] for arr in proba_array]
        return (_auc_curve(test_array, proba_array, n_splits=n_splits), _auc_curve(test_array, pred_array, n_splits=n_splits))
