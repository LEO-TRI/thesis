import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score

import plotly.express as px
import plotly.graph_objects as go

def date_range(min_date, date, max_date):
    return (min_date <= date <= max_date)
date_range_vec = np.vectorize(date_range)

def update_prop(handle, orig):
    handle.update_from(orig)
    x,y = handle.get_data()
    handle.set_data([np.mean(x)]*2, [0, 2*y[0]])

def table_color(data:pd.Series, palette_min=145, palette_up=300, n=5):

    hex_colors = [mcolors.to_hex(color) for color in sns.diverging_palette(palette_min, palette_up, s=60, n=n)]
    hex_colors.reverse()

    bins = np.quantile(data.values, np.arange(0, 1.1, 0.2))
    vals = data.values

    cell_color = [hex_colors[0] if val <= bins[0] else hex_colors[1] if bins[0] < val < bins[1] else hex_colors[2] if bins[1] < val < bins[2] \
        else hex_colors[3] if bins[2] <= val <= bins[3] else hex_colors[4] for val in vals]

    return cell_color

def line_adder(h_coord=0.5, color="black", linestyle="-", *args):
    line = plt.Line2D([0.15,0.85], [h_coord, h_coord], transform=args.transFigure, color="black", linestyle="-")
    return args.add_artist(line)

def custom_combiner(feature, category):
    return str(category)

def get_top_features(vectoriser, clf, selector = None, top_n: int = 25, how: str = 'long'):
    """
    Convenience function to extract top_n predictor per class from a model.
    """

    assert hasattr(vectoriser, 'get_feature_names_out')
    assert hasattr(clf, 'coef_')
    assert hasattr(selector, 'get_support')
    assert how in {'long', 'wide'}, f'how must be either long or wide not {how}'

    features = vectoriser.get_feature_names_out()
    if selector is not None:
        features = features[selector.get_support()]
    axis_names = [f'feature_{x + 1}' for x in range(top_n)]

    if len(clf.classes_) > 2:
        results = list()
        for c, coefs in zip(clf.classes_, clf.coef_):
            idx = coefs.argsort()[::-1][:top_n]
            results.extend(tuple(zip([c] * top_n, features[idx], coefs[idx])))
    else:
        coefs = clf.coef_.flatten()
        idx = coefs.argsort()[::-1][:top_n]
        results = tuple(zip([clf.classes_[1]] * top_n, features[idx], coefs[idx]))

    df_lambda = pd.DataFrame(results, columns =  ['SDG', 'feature', 'coef'])

    if how == 'wide':
        df_lambda = pd.DataFrame(
            np.array_split(df_lambda['feature'].values, len(df_lambda) / top_n),
            index = clf.classes_ if len(clf.classes_) > 2 else [clf.classes_[1]],
            columns = axis_names
        )

    #df_lambda["SDG"] = df_lambda["SDG"].astype(str)
    #df_lambda["SDG"] = df_lambda["SDG"].map(DataProcess().sdg)

    return df_lambda

def sdg_explainer(df: pd.DataFrame)-> px.bar:
    colors = px.colors.qualitative.Dark24[:20]
    template = 'SDG: %{customdata}<br>Feature: %{y}<br>Coefficient: %{x:.2f}'

    fig = px.bar(
        data_frame = df,
        x = 'coef',
        y = 'feature',
        custom_data = ['SDG'],
        facet_col = 'SDG',
        facet_col_wrap = 3,
        facet_col_spacing = .15,
        height = 1200,
        labels = {
            'coef': 'Coefficient',
            'feature': ''
        },
        title = 'Top 15 Strongest Predictors by SDG'
    )

    fig.for_each_trace(lambda x: x.update(hovertemplate = template))
    fig.for_each_trace(lambda x: x.update(marker_color = colors.pop(0)))
    #fig.for_each_annotation(lambda x: x.update(text = fix_sdg_name(x.text.split("=")[-1])))
    fig.update_yaxes(matches = None, showticklabels = True)
    fig.show()

    return fig

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, width:int=400, height:int=400) -> go.Figure:
    """
    Convenience function to display a confusion matrix in a graph.
    """
    labels = sorted(list(set(y_true)))
    df_lambda = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index = labels,
        columns = labels
    )
    total = np.sum(df_lambda, axis=1)
    df_lambda = df_lambda/total
    df_lambda = df_lambda.apply(lambda value : np.round(value, 2))

    acc = accuracy_score(y_true, y_pred)
    f1s = f1_score(y_true, y_pred, average = 'weighted')
    precision = precision_score(y_true, y_pred, average = 'weighted')

    fig = px.imshow(df_lambda, text_auto=True,
                    color_continuous_scale='RdBu_r',
                    labels=dict(x="Predicted", y="Actual", color="Proportion"),
                    x=df_lambda.columns,
                    y=df_lambda.index,
                    title=f'Accuracy: {acc:.2f}, F1: {f1s:.2f}, Precision: {precision:.2f}',
                    width=width, height=height)

    fig.update_layout(
        title={
            'y':0.88,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            "font_family": "Arial",
            "font_color": "black",
            "font_size":14},
        font_family="Arial",
        font_color="black"
        )

    fig.show()

    import plotly.graph_objects as go

    #fig = go.Figure(data=go.Heatmap(
    #                z=[row for ind, row in df_lambda.iterrows()],
    #                x=df_lambda.columns,
    #                y=df_lambda.index,
    #                hovertemplate = 'Price: %{y:$.2f}<extra></extra>',
    #                hoverongaps = False))
    #fig.show()

    return fig
