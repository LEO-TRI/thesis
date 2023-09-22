########################### ML TEMPLATE ##############################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report, RocCurveDisplay, auc
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer

import pandas as pd
import numpy as np

from scripts_thesis.utils import custom_combiner
from scripts_thesis.params import *

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from colorama import Fore, Style

import os
import pickle

scoring = ['accuracy', 'precision', 'recall', 'f1']

def print_results(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Convenience function used to quickly compute and display the evaluation metrics of a model.
    Can be used after getting y_pred from a trained model.
    Retuurns a dictionnary with 4 metrics and their corresponding values.

    Parameters
    ----------
    y_test : np.ndarray
        Array of the real values
    y_pred : np.ndarray
        Array of the predicted values

    Returns
    -------
    dict
        Dictionnary of evaluation metrics
    """
    metrics = [np.round(accuracy_score(y_test, y_pred), 2),
               np.round(precision_score(y_test, y_pred, zero_division= 0), 2),
               np.round(recall_score(y_test, y_pred, zero_division= 0), 2),
               np.round(f1_score(y_test, y_pred, zero_division= 0), 2)
               ]

    metrics_name = ["accuracy", "precision", "recall", "f1"]
    metrics = dict(zip(metrics_name, metrics))

    print(f"Accuracy: {metrics.get('accuracy')}",
          f"Precision: {metrics.get('precision')}",
          f"Recall: {metrics.get('recall')}",
          f"F1 Score: {metrics.get('f1')}")

    return metrics


def baseline_model(y: np.ndarray, test_split: float=0.3) -> np.ndarray:
    """
    Function computing the baseline to beat by the new model.
    Produces two baseline, one coming from a random guess and the other from predicting only the majority class

    Parameters
    ----------
    y : np.ndarray/array_like
        The target feature
    test_split : float, optional
        The split ratio between train and test, by default 0.3

    Returns
    -------
    np.ndarray
        Array of predicted results.
    """

    print(Fore.MAGENTA + "\n ⭐️ Results to beat" + Style.RESET_ALL)
    y_train, y_test = train_test_split(y, test_size=test_split, random_state=42, stratify=y)

    print(Fore.BLUE + "\n Result for random baseline" + Style.RESET_ALL)
    y_random = np.random.randint(0, 2, size=len(y_test))
    print_results(y_test, y_random)

    print(Fore.BLUE + "\n Result for majority baseline" + Style.RESET_ALL)
    y_majority = np.zeros(len(y_test))
    print_results(y_test, y_majority)

    return y_random, y_majority


def build_pipeline(numeric_cols:list[str], text_cols:list[str], other_cols:list[str],
                   description:str='description', amenities:str='amenities', host:str="host_about",
                   max_features: int=1000, max_features_tfidf: int=10000, max_kbest: int=1000) -> Pipeline:
    """
    A convenience function created to quickly build a pipeline. Requires the columns' names for the column transformer.
    Pipeline takes a cleaned dataset.
    Pipeline does the preprocessing, the balancing of the classes and instantiate a sklearn's model.
    Returns the pipeline.

    Parameters
    ----------
    numeric_cols : list(str)
        The numerical columns of the dataset
    text_cols : list(str)
        The text columns of the dataset
    other_cols : list(str)
        The remaining columns of the dataset
    max_features : int, optional
        How many columns to keep from the tfidf vectorization, by default 1000

    Returns
    -------
    Pipeline
        A sklearn pipeline, not fitted
    """

    numeric_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(random_state=1830)),
        ('scaler', RobustScaler())
    ])

    num_transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
        ],
        remainder='drop'  # Pass through any other columns not specified
    )

    text_transformers = ColumnTransformer(
        transformers=[
            ('text1', TfidfVectorizer(max_features=max_features_tfidf, ngram_range = (1, 3), max_df=0.8, norm="l1"), description),
            ('text2', TfidfVectorizer(max_features=max_features_tfidf, ngram_range = (1, 3), max_df=0.8, norm="l1"), amenities),
            ('text3', TfidfVectorizer(max_features=max_features_tfidf, ngram_range = (1, 3), max_df=0.8, norm="l1"), host)
            ],
        remainder='drop'  # Pass through any other columns not specified
        )

    text_pipe = Pipeline([
        ('text_preprocessing', text_transformers),
        ('selectkbest', SelectKBest(chi2, k=max_kbest))
        ]
                         )

    cat_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), other_cols)
        ],
        remainder='drop'  # Pass through any other columns not specified
    )

    column_transformer = FeatureUnion([("text", text_pipe),
                                       ("num", num_transformer),
                                       ("cat", cat_transformer)
                                       ])

    # Create the final pipeline
    pipeline = Pipeline([
        ("balancing", RandomUnderSampler(random_state=1830)),
        ('preprocessing', column_transformer),
       #('smote', SMOTE(random_state=42, k_neighbors=20)),
       #('selector', SelectKBest(chi2, k = 2000)),
        ('classifier', LogisticRegression(penalty = 'l2', C = .9,
                                multi_class = 'auto', class_weight = 'balanced',
                                random_state = 1830, solver = 'newton-cg', max_iter = 100))
        ]
                        )

    return pipeline


def train_model(X: pd.DataFrame, y: np.ndarray, test_split: float=0.3, max_features: int=1000, n_splits: int = 5) -> Pipeline:
    """
    Fit the passed model with the passed data and return a tuple (fitted_model, history)

    Parameters
    ----------
    X : pd.DataFrame
        The dataframe of features
    y : np.ndarray
        The target variable
    test_split : float, optional
        The split between train and test samples, by default 0.3
    max_features : int, optional
        The max number of features for the tfidf vectorizer, by default 1000
    n_splits : int, optional
        The number of folds for the cross-val

    Returns
    -------
    Pipeline : imblearn.pipeline.Pipeline/sklearn.pipeline.Pipeline
        A fitted pipeline object
    res : pd.DataFrame
        A dataframe with the mean cross-validated metrics (4 in total)
    """

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    text_cols = ["amenities", "description", "host_about"]
    other_cols = list(set(X.columns) - set(numeric_cols) - set(text_cols))

    pipe_model = build_pipeline(numeric_cols, other_cols, max_features_tfidf = max_features)

    print(Fore.BLUE + "\nLaunching CV" + Style.RESET_ALL)

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=2, random_state=42)

    res = []
    pred_list = []
    test_list = []

    for fold, (train, test) in enumerate(cv.split(X, y)):
        pipe_model.fit(X[train], y[train])
        y_pred = pipe_model.predict(X[test])

        res.append(print_results(y[test], y_pred))
        pred_list.append(y_pred)
        test_list.append(y[test])








    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=1830, stratify=y)
    res = cross_validate(pipe_model, X_train, y_train, verbose=2, cv=cv, scoring=scoring)
    res = pd.DataFrame(res)

    #size_data = y_train.value_counts().sort_values(ascending=False).iloc[0] * 16
    print(f"✅ Model trained on \n {len(X_train)} original rows")
    print(f"Mean cross_validated accuracy: {round(np.mean(res.get('test_accuracy')), 2)}")
    print(f"Mean cross_validated precision: {round(np.mean(res.get('test_precision')), 2)}")

    pipe_model.fit(X_train, y_train)

    return pipe_model, res


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, test_split:float=0.3) -> pd.DataFrame:
    """
    Evaluate trained model performance on the dataset

    Parameters
    ----------
    model : object
        A sklearn model, instantiated from a pickle file or trained before.
    X : pd.DataFrame
        The dataframe of features
    y : pd.Series/np.ndarray
        The target variable
    test_split : float, optional
        The split ratio between train and test, by default 0.3

    Returns
    -------
    pd.DataFrame
        A dataframe with the evaluated metrics (4 metrics)
    y_pred: np.ndarray
        A np.array of the model's prediction
    y_test: np.ndarray
        A np.array of the real data
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42, stratify=y)

    print(Fore.BLUE + f"\nEvaluating model on {len(X_test)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    y_pred = model.predict(X_test)

    metrics = [accuracy_score(y_test, y_pred) , precision_score(y_test, y_pred, average="macro"),
               f1_score(y_test, y_pred, average="macro"), recall_score(y_test, y_pred, average="macro")]
    metrics_name = ["res_accuracy", "res_precision", "res_f1", "res_recall"]

    print(f"✅ Model evaluated")
    print_results(y_test, y_pred)


    print(f"✅ Model evaluated, accuracy: {np.round(metrics[0], 2)}, precision: {np.round(metrics[1], 2)}, recall: {np.round(metrics[2], 2)}")

    print(f"✅ Full Classification Report")
    print(classification_report(y_test, y_pred, zero_division = 0))

    results = dict(zip(metrics_name, metrics))
    results = {key: [value] for key, value in results.items()}
    return pd.DataFrame(results, index=[0]), y_pred, y_test


def predict_model(model, X: pd.DataFrame) -> np.ndarray:
    """Function to predict using a trained model

    Parameters
    ----------
    model : object
        A trained sklearn model
    X : pd.DataFrame
        Data passed in the model to obtained predictions

    Returns
    -------
    np.ndarray
        An array of 0 and 1, depending on predictions
    """
    return model.predict(X), model.predict_proba(X)


def load_model(model_name: str = None) -> None:
    """
    Convenience function to load a pickled model

    Parameters
    ----------
    model_name : str, optional
        the name of a trained model, if None, returns the latest saved model, by default None

    Returns
    -------
    model : object
        The loaded sklearn model
    """
    if model_name==None:
        full_file_path = os.path.join(LOCAL_MODEL_PATH, "None")
    else:
        full_file_path = os.path.join(LOCAL_MODEL_PATH, model_name)

    if not os.path.exists(full_file_path):
        files = [os.path.join(LOCAL_MODEL_PATH, file) for file in os.listdir(LOCAL_MODEL_PATH) if file.endswith(".pkl")]

        if len(files)==0:
            print("No model trained, please train a model")
            return None

        print("No specific model passed, returning latest saved model")
        full_file_path = max(files, key=os.path.getctime)

    model = pickle.load(open(full_file_path, 'rb'))
    return model
