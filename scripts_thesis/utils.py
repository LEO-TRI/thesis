import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from scipy import stats, sparse

from sklearn.feature_selection import chi2, SelectFpr, SelectFwe

hex_colors = [mcolors.to_hex(color) for color in sns.diverging_palette(145, 300, s=60, n=5)]
hex_colors.reverse()

custom_params = {"axes.spines.right": False, "axes.spines.top": False, "figure.figsize":(8, 8)}
sns.set_theme(context='notebook', style='darkgrid', palette='deep', rc= custom_params)

def date_range(min_date, date, max_date) -> [bool]:
    return (min_date <= date <= max_date)

date_range_vec = np.vectorize(date_range)

def update_prop(handle, orig):
    handle.update_from(orig)
    x,y = handle.get_data()
    handle.set_data([np.mean(x)]*2, [0, 2*y[0]])


def table_color(data:pd.Series, palette_min: int=145, palette_up: int=300, n: int=5) -> list:
    """
    Convenience function to convert a numerical sequence into a color sequence based on its quantiles

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
    cell_color : list
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
    """
    A convenience function that can be used in sklearn's ohe to format column names.

    Used only in build pipelines in model_ML.py

    Requires sklearn version >= 1.3.0
    """
    return str(category)

def get_top_features(model, has_selector: bool= True, top_n: int= 25, how: str= 'long') -> pd.DataFrame:
    """
    Convenience function to extract top_n predictor per class from a model.

    Parameters
    ----------
    model : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline
        The model must have 4 elements:

            vectoriser : sklearn.feature_extraction.text.TfidfVectorizer
                The sklearn vectoriser used to transform strings into words

            clf : sklearn.linear_model.LogisticRegression
                The sklearn predictor used to classify the data

            ohe: sklearn.preprocessing.OneHotEncoder
                The sklearn preprocessor for categorical data

            selector : sklearn.feature_selection.SelectKBest, optional
                The sklearn selector used to reduce the number of features, by default None

    top_n : int, optional
        Number of top features to return, by default 25
    how : str, optional
        Shape of the output, by default 'long'

    Returns
    -------
    df_lambda : pd.DataFrame
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

def params_combiner(classifier: str="logistic", params_clf: dict= None) -> dict:
    """
    A convenience function used to create the parameter dictionnary for hyperparameter tuning

    Functions only with RandomizedSearchCV because of the use of scipy.stats with a continuous distribution.

    Parameters
    ----------
    classifier : str, optional
        The classifier used in the pipeline, by default "logistic"
    params_clf : dict, optional
        The classifier's dict of params, by default None

    Returns
    -------
    pipe_params : dict
        A full dictionnary of hyperparameters and potential values
    """

    pipe_params = dict(preprocessing__text__selectkbest__k=np.arange(100, 2000 + 1, 100),
                       preprocessing__text__text_preprocessing__text1__ngram_range=[(1, 1), (1, 2), (1, 3)],
                       preprocessing__text__text_preprocessing__text2__ngram_range=[(1, 1), (1, 2), (1, 3)],
                       preprocessing__text__text_preprocessing__text3__ngram_range=[(1, 1), (1, 2), (1, 3)],
                       preprocessing__text__text_preprocessing__text1__norm=["l1", "l2"],
                       preprocessing__text__text_preprocessing__text2__norm=["l1", "l2"],
                       preprocessing__text__text_preprocessing__text3__norm=["l1", "l2"]
                       )

    if classifier == "logistic":
        params_clf = dict(classifier__C=stats.uniform(loc=0, scale=5),
                          classifier__penalty=["l1", "l2"]
                          )

    elif classifier == "gbt":
        params_clf = dict(classifier__learning_rate=stats.uniform(loc=0, scale=1),
                          classifier__max_depth=np.arange(1, 5),
                          classifier__max_leaf_nodes=np.arange(5, 60),
                          classifier__l2_regularization=stats.uniform(loc=0, scale=1),
                          classifier__max_bins=np.arange(50, 256)
                          )

    elif classifier == "sgd":
        params_clf = dict(classifier__penalty=["l1", "l2"],
                          classifier__alpha=stats.uniform(loc=0, scale=1),
                          classifier__learning_rate=["constant", "optimal"],
                          classifier__loss=["hinge", "squared_hinge", "perceptron"]
        )

    elif classifier == "random_forest":
        params_clf = dict(classifier__n_estimators=np.arange(50, 301, 10),
                          classifier__max_depth=np.arange(1, 10),
                          classifier__max_leaf_nodes=np.arange(20, 101),
                          classifier__min_samples_split =np.arange(2, 50),
                          classifier__min_samples_leaf=np.arange(1, 50),
                          classifier__max_features=["log2", "sqrt"],
                          )

    elif classifier == "xgb":
        params_clf = dict(classifier__n_estimators=np.arange(10, 101, 5),
                          classifier__max_depth=np.arange(1, 5),
                          #classifier__grow_policy=['depthwise', 'lossguide'],
                          classifier__learning_rate=stats.uniform(loc=0, scale=1),
                          classifier__booster =["gbtree", "gblinear", "dart"],
                          classifier__reg_alpha=stats.uniform(loc=0, scale=1)
                          )

    elif classifier == "stacked":
        params_clf = dict(classifier__final_estimator__C=stats.uniform(loc=0, scale=5),
                          classifier__final_estimator__penalty=["l1", "l2"],
                          classifier__rf__n_estimators=np.arange(50, 301, 10),
                          classifier__rf__max_depth=np.arange(1, 5),
                          classifier__rf__max_leaf_nodes=np.arange(20, 101),
                          classifier__rf__min_samples_split =np.arange(2, 50),
                          classifier__rf__min_samples_leaf=np.arange(1, 50),
                          classifier__rf__max_features=["log2", "sqrt"],
                          classifier__gbt__learning_rate=stats.uniform(loc=0, scale=1),
                          classifier__gbt__max_depth=np.arange(1, 5),
                          classifier__gbt__max_leaf_nodes=np.arange(5, 60),
                          classifier__gbt__l2_regularization=stats.uniform(loc=0, scale=1),
                          classifier__gbt__max_bins=np.arange(50, 256)
                        )

    if params_clf is not None:
        pipe_params.update(params_clf)

    return pipe_params

def params_extracter(model: object) -> dict:
    """
    A convenience function to extract the parameters from a randomized search

    Parameters
    ----------
    model : sklearn.pipeline
        A sklearn model or pipeline that has been cross-searched

    Returns
    -------
    dict
        A dict of the best scores for the cross validation in the order : AUC, accuracy, precision
    """

    ind = model.cv_results_.get("mean_test_precision").argmax() #Get the best results according to precision
    return {key[10:]: model.cv_results_.get(key)[ind] for key in model.cv_results_.keys() if "mean_test_" in key} #[10:] is used to remove mean_test_,

def sparse_to_dense(X: sparse.csr_matrix) -> np.ndarray:
    """
    Convert a sparse matrix to a dense matrix.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        The input data, which can be either a dense or sparse matrix.

    Returns
    -------
    {array-like, numpy.ndarray}
        A dense matrix if the input is sparse; otherwise, the input is returned unchanged.
    """

    if isinstance(X, sparse.csr_matrix):
        return X.toarray()
    else:
        return X
