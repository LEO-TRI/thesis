import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
import plotly.express as px


def date_range(min_date, date, max_date) -> [bool]:
    return (min_date <= date <= max_date)


date_range_vec = np.vectorize(date_range)


def update_prop(handle, orig):
    handle.update_from(orig)
    x,y = handle.get_data()
    handle.set_data([np.mean(x)]*2, [0, 2*y[0]])


def table_color(data:pd.Series, palette_min: int=145, palette_up: int=300, n: int=5) -> list:
    """Convenience function to convert a numerical sequence into a color sequence based on its quantiles

    Parameters
    ----------
    data : pd.Series
        numerical data on which to build the color scale
    palette_min : int, optional
        minimum for the palette spectrum, by default 145
    palette_up : int, optional
        maxmimum for the palette spectrum, by default 300
    n : int, optional
        Number of discrete colors to have in the palette, by default 5

    Returns
    -------
    list
        Returns a list of the size of data and with each element being a color hexacode 
    """
    hex_colors = [mcolors.to_hex(color) for color in sns.diverging_palette(palette_min, palette_up, s=60, n=n)]
    hex_colors.reverse()

    bins = np.quantile(data.values, np.linspace(0, 1, num=n))
    vals = data.values

    cell_color = [hex_colors[0] if val <= bins[0] else hex_colors[1] if bins[0] < val < bins[1] else hex_colors[2] if bins[1] < val < bins[2] \
        else hex_colors[3] if bins[2] <= val <= bins[3] else hex_colors[4] for val in vals]

    return cell_color


def line_adder(h_coord=0.5, color="black", linestyle="-", *args):
    line = plt.Line2D([0.15,0.85], [h_coord, h_coord], transform=args.transFigure, color=color, linestyle=linestyle)
    return args.add_artist(line)


def custom_combiner(feature, category):
    return str(category)


def get_top_features(model, has_selector: bool = True, top_n: int = 25, how: str = 'long') -> pd.DataFrame:
    """Convenience function to extract top_n predictor per class from a model.

    Parameters
    ----------
    vectoriser : _type_
        The sklearn vectoriser used to transform strings into words
    clf : _type_
        The sklearn model used to classify the data 
    selector : _type_, optional
        The selector used to reduce the number of features, by default None
    top_n : int, optional
        Number of top features to return, by default 25
    how : str, optional
        Shape of the output, by default 'long'

    Returns
    -------
    pd.DataFrame
        Output dataframe with rows being each one of the top_n most important coef with its respective value 
    """

    vectoriser = model["preprocessor"].get_params().get("transformers")[1][1]
    ohe = model["preprocessor"].get_params().get("transformers")[2][1]
    clf = model["clf"]

    assert hasattr(vectoriser, 'get_feature_names_out')
    assert hasattr(ohe, 'get_feature_names_out')
    assert hasattr(clf, 'coef_')
    assert how in {'long', 'wide'}, f'how must be either long or wide not {how}'

    if has_selector:
        selector = model["selector"]
        assert hasattr(selector, 'get_support')

    num_cols = model["preprocessor"].get_params().get("transformers")[0][2]
    text_cols = vectoriser.get_feature_names_out()
    ohe_cols = ohe.get_feature_names_out()
    features = np.concatenate([num_cols, text_cols, ohe_cols])

    if has_selector:
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

    df_lambda = pd.DataFrame(results, columns =  ['Class', 'feature', 'coef'])

    if how == 'wide':
        df_lambda = pd.DataFrame(
            np.array_split(df_lambda['feature'].values, len(df_lambda) / top_n),
            index = clf.classes_ if len(clf.classes_) > 2 else [clf.classes_[1]],
            columns = axis_names
        )

    return df_lambda


def plot_confusion_matrix(y_true: np.array, y_pred: np.array, width=400, height=400) -> px.imshow:
    """_summary_

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
    px.imshow
        A confusion matrix of the model's result 
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

    return fig


def model_explainer(df: pd.DataFrame, x: str= "coef", y: str= "feature")-> px.bar:
    """A function that takes the dataframe outputed by get_top_features() and creates a barplot of most important features using Plotly

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
    px.bar
        A bar graph showing the most important coeficients and their values
    """

    colors = px.colors.qualitative.Dark24[:20]
    template = 'SDG: %{customdata}<br>Feature: %{y}<br>Coefficient: %{x:.2f}'

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