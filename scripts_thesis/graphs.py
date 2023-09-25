import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, RocCurveDisplay, auc

from scripts_thesis.utils import *

import numpy as np
import pandas as pd

custom_params = {"axes.spines.right": False, "axes.spines.top": False, "figure.figsize":(8, 8)}
sns.set_theme(context='notebook', style='darkgrid', palette='deep', rc= custom_params)

hex_colors = [mcolors.to_hex(color) for color in sns.diverging_palette(145, 300, s=60, n=5)]
hex_colors.reverse()

def plot_confusion_matrix(y_true: np.array, y_pred: np.array, width= 600, height= 600) -> go.Figure:
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

def auc_cross_val(test_array: list, pred_array: list, n_splits: int= 5):
    """
    A function to produce a figure with several ROC curves when cross-validating a model

    Parameters
    ----------
    test_array : list
        A list of lists with the actual values for each fold
    pred_array : list
        A list of lists with the predicted values for each fold
    n_splits : int, optional
        Number of folds by cv, by default 5

    Returns
    -------
    plt.Figure
        A matplotlib figure

    """

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)


    fig, ax = plt.subplots(figsize=(6, 6))

    for fold, (y_test, y_pred) in enumerate(zip(test_array, pred_array)):

        viz = RocCurveDisplay.from_predictions(
            y_test,
            y_pred,
            name=f"ROC fold {1 + fold // n_splits} - {fold % n_splits + 1}",
            alpha=0.3,
            lw=1,
            ax=ax,
            #plot_chance_level=(fold == n_splits * 2 - 1)
            )


        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr) # Interpolates additional points for the curve
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)


    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC curve with variability\n (Positive label)",
    )

    ax.axis("square")
    ax.legend(bbox_to_anchor=(1.65, 0.25), loc="lower right")
    plt.show()

    return fig