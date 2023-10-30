import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, auc, fbeta_score

from scripts_thesis.utils import *
from scripts_thesis.charter import *
from scripts_thesis.ml_display import RocCurveDisplayPlotly, PrecisionRecallDisplayPlotly

import numpy as np
import pandas as pd

from colorama import Fore, Style

custom_params = {"axes.spines.right": False, "axes.spines.top": False, "figure.figsize":(8, 8)}
sns.set_theme(context='notebook', style='darkgrid', palette='deep', rc= custom_params)

hex_colors = [mcolors.to_hex(color) for color in sns.diverging_palette(145, 300, s=60, n=5)]
hex_colors.reverse()


#TODO Consider whether all graph function could be moved with a larger Graph class
#TODO OR create Graph class and combine with Model class to store outputs and be able to call instance method
#TODO that compute graph from it

#Look into multiple inheritance in this case

def plot_confusion_matrix(test_array: np.ndarray, target_array: np.ndarray, threshold: float=0.5, width: int= 600, height: int= 600) -> go.Figure:
    """
    Convenience function to print a confusion matrix with the predicted results y_pred and actual data y_true

    Parameters
    ----------
    test_array : np.ndarray
        Array of real data. Can be 1-D (one set of data) or 2-D (cross-validated data)
    target_array : np.ndarray
        Array of predicted data. Can be 1-D (one set of predictions) or 2-D (cross-validated predictions)
    width : int, optional
        dimension of the image, by default 400
    height : int, optional
        dimension of the image, by default 400

    Returns
    -------
    fig : go.Figure
        A confusion matrix of the model's result
    """

    if len(test_array) != len(target_array):
        raise ValueError("test_array and target_array must have the same shape")


    if is_array_like(test_array[0]):
        test_array = np.concatenate(test_array)
        target_array = np.concatenate(target_array)

    if set(target_array) != set([0, 1]):
        target_array = np.where(target_array>=threshold, 1, 0)

    labels = sorted(list(set(test_array)))

    df_lambda = pd.DataFrame(
        confusion_matrix(test_array, target_array),
        index = labels,
        columns = labels
    )

    #Get results as proportion of total results
    df_lambda = df_lambda.div(np.sum(df_lambda, axis=1), axis=0) *100
    df_lambda = df_lambda.apply(lambda value : np.round(value, 2))

    acc = accuracy_score(test_array, target_array)
    fbetas = fbeta_score(test_array, target_array, beta=0.5)
    precision = precision_score(test_array, target_array)
    recall = recall_score(test_array, target_array)

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

    title = f'Confusion Matrix : Overall results <br><sup>Accuracy: {acc:.2f}, FBeta: {fbetas:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}</sup>'
    fig.update_layout(#<br> is a line break, and <sup> is superscript
        title=title,
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

    fig.show()

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

    Used as part of graph_cross_val

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
        The cross validated AUC curves in 1 plotly go.Figure
    """

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig = go.Figure()

    mask = len(test_array) - 1

    for fold, (y_test, y_pred) in enumerate(zip(test_array, target_array)):

        viz = RocCurveDisplayPlotly.from_predictions(y_test,
                                                     y_pred,
                                                     pos_label=1,
                                                     fold=fold,
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
                                name=f"Mean ROC (AUC = {np.round(mean_auc, 3)})"
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
    fig.show()

    return fig


def _prc_curve(test_array: np.ndarray, target_array: np.ndarray, n_splits: int, **kwargs) -> go.Figure:
    """
    A function used to calculate cross_validated precision recall.

    Used as part of graph_cross_val

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
        The cross validated PRC curves in 1 plotly figure
    """

    mask = len(test_array) - 1

    fig = go.Figure()

    for fold, (y_test, y_pred) in enumerate(zip(test_array, target_array)):

        viz = PrecisionRecallDisplayPlotly.from_predictions(y_test,
                                                            y_pred,
                                                            pos_label=1,
                                                            fold=fold,
                                                            n_splits=n_splits,
                                                            plot_chance_level=(mask == fold),
                                                            fig=fig,
                                                            show_fig=False)

    fig.update_layout(
        xaxis=dict(range=[viz.prevalence_pos_label - viz.prevalence_pos_label/4, 1.05]),
        yaxis=dict(range=[viz.prevalence_pos_label - viz.prevalence_pos_label/4, 1.05]),
        title="PRC curves with variability\n (CV)",
        legend=dict(title_text='PRC results',
                    yanchor="bottom",
                    y=0.0,
                    xanchor="left",
                    x= 1.05,
                    )
        )

    width = kwargs.get("width", 950)
    height = kwargs.get("height", 600)

    fig = charter_plotly(fig, width=width, height=height)
    fig.show()

    return fig


def metrics_on_one_plot(test_array: list, target_array: list) -> go.Figure:
    """
    Creates a threshold curve with accuracy, precision, recall and f1 score

    Parameters
    ----------
    test_array : list
        A list of arrays of y_test values
    target_array : list
        A list of arrays of y_pred values

    Returns
    -------
    go.Figure
        A figure with 5 line traces on it

    Raises
    ------
    ValueError
        In case someone passes predictions rather than probabilities
    """

    if set(target_array[0]) == set([0, 1]):
        raise ValueError("Probabilities must be passed in 'target_array' to examine thresholds")

    if len(test_array) !=  len(target_array):
        raise ValueError("Predicted and actual values must have the same length")

    thresholds = np.linspace(0, 1, 500)

    metrics = ["accuracy", "precision", "recall", "fbeta", "queue_rate"]
    names = dict(zip(np.arange(len(metrics)), metrics))

    #3D array with axis 0 being thresholds, axis 1 being metrics and axis 2 being folds
    results = np.zeros((len(thresholds), len(metrics), len(test_array)))

    for fold, (y_test, y_proba) in enumerate(zip(test_array, target_array)):

        for i, threshold in enumerate(thresholds):

            y_pred = np.where(y_proba>=threshold, 1, 0)
            results[i,:,fold] = [accuracy_score(y_test, y_pred),
                                precision_score(y_test, y_pred, zero_division=1),
                                recall_score(y_test, y_pred, zero_division=0),
                                fbeta_score(y_test, y_pred, beta=0.5),
                                queue_rate(y_pred, threshold)
                                ]

    means = np.mean(results, axis=2, keepdims=False)
    stds = np.std(results, axis=2, keepdims=False)

    means_plus = means + 1.96 * stds/np.sqrt(results.shape[2])
    means_minus = means - 1.96 * stds/np.sqrt(results.shape[2])


    #Add metrics lines
    graphs = [dict(type="scatter",
                x=thresholds,
                y=means[:,i],
                name=names.get(i),
                mode="lines",
                marker=dict(color=PALETTE[i % len(PALETTE), 0])#, width=2),
                )
            for i in range(means.shape[1])
            ]

    #Add confidence intervals
    graphs_ci = [dict(type="scatter",
                        x=np.array(list(thresholds) + list(thresholds)[::-1]),
                        y=np.array(list(means_plus[:,i]) + list(means_minus[:,i])[::-1]),
                        fill='toself',
                        showlegend=False,
                        mode='lines',
                        hoverinfo="skip",
                        opacity=0.3,
                        line=dict(color=DARK_GREY[1])#, width=1),
                        )
                   for i in range(means.shape[1])
                   ]

    max_indices = np.argmax(means, axis=0, keepdims=False)
    best_queue = find_nearest(means[:, 4], 0.20)
    graphs_vertical = [dict(type="scatter",
                            x=[thresholds[max_indices[3]], thresholds[max_indices[3]]],
                            y=[0,1],
                            name="Best FBeta score",
                            mode='lines',
                            line=dict(color=BLUE_GREY[1], width=3, dash='dash')
                            ),
                       dict(type="scatter",
                            x=[thresholds[best_queue], thresholds[best_queue]],
                            y=[0,1],
                            name="Optimal queue rate",
                            mode='lines',
                            line=dict(color=BROWN[1], width=3, dash='dash')
                            ),
                       ]

    graphs = graphs + graphs_ci + graphs_vertical

    layout = dict(title={"text": "Threshold plot", 'font': {'size': 20}},
                  xaxis={"title": "Thresholds"},
                  yaxis={"title": "Score", "range": [-0.05, 1]})
    fig = go.Figure(data=graphs, layout=layout)

    fig.show()

    return thresholds[max_indices[3]]

def feature_importance_plotting(features: np.ndarray) -> go.Scatter:
    """
    _summary_

    Parameters
    ----------
    features : np.ndarray
        _description_

    Returns
    -------
    go.Scatter
        _description_
    """

    index = np.ones(len(features))
    fig =  px.scatter(y=index, x=features, color_discrete_sequence=hex_colors)
    fig.update_layout(title="Feature importance Plot",
                      xaxis_title="Feature Importance",
                      yaxis_title="")
    return fig


def probability_distribution(test_array: list, target_array: list) -> go.Histogram:
    """
    Computes a histogram with the probability distribution of both classes based
    on the output of a model.

    Parameters
    ----------
    test_array : list
        A list of arrays of y_test values
    target_array : list
        A list of arrays of y_pred values

    Returns
    -------
    go.Histogram
        A plotly histogram with the distribution of both classes
    """
    fig = px.histogram(x = target_array, color = test_array,
                       color_discrete_sequence=[hex_colors[0], hex_colors[-1]], marginal="violin",
                       nbins=40)

    fig.update_layout(title="Probability distribution by class",
                      xaxis_title="Output probabilities",
                      yaxis_title="Count")
    fig.show()

    return fig

def graphs_cross_val(auc_metrics: dict, n_splits: int= 5):
    """
    Centralises the calls for the various graph functions in one place

    Parameters
    ----------
    auc_metrics : dict
        A dictionnary containing lists of lists for respectively y_test, y_pred or y_proba by cv fold
    n_splits : int, optional
        Number of folds by cv, by default 5

    Returns
    -------
    tuple
        A tuple containing two plotly graphs depending on the type of classifiers used
    """

    print(Fore.BLUE + "Producing graph" + Style.RESET_ALL)

    sample_length = len(auc_metrics.get("test_array"))
    sample = np.random.choice(sample_length, int(sample_length/2), replace=False)

    plot_confusion_matrix(**auc_metrics)
    #p.array(v)[[sample], :].reshape(len(sample), len(v[0])
    auc_metrics = {k : [list(val) for i, val in enumerate(v) if i in sample] for k, v in auc_metrics.items()}

    best_threshold = metrics_on_one_plot(**auc_metrics)

    return (_auc_curve(**auc_metrics, n_splits=n_splits), _prc_curve(**auc_metrics, n_splits=n_splits), best_threshold)
