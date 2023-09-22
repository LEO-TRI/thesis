########################### ML TEMPLATE ##############################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
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


def print_results(y_test: np.array, y_pred: np.array) -> dict:
    """
    Convenience function used to quickly compute and display the evaluation metrics of a model. 
    Can be used after getting y_pred from a trained model. 
    Retuurns a dictionnary with 4 metrics and their corresponding values.

    Parameters
    ----------
    y_test : np.array
        Array of the real values 
    y_pred : np.array
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


def baseline_model(y: np.array,
        test_split: float=0.3
    ) -> np.array:
    """
    Function computing the baseline to beat by the new model. 
    Produces two baseline, one coming from a random guess and the other from predicting only the majority class

    Parameters
    ----------
    y : np.array/array_like
        The target feature
    test_split : float, optional
        The split ratio between train and test, by default 0.3

    Returns
    -------
    np.array
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


def build_pipeline(numeric_cols:[str], text_cols:[str], other_cols:[str], max_features:int=1000) -> Pipeline:
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
        ('imputer', IterativeImputer(random_state=42)),
        ('scaler', RobustScaler())
    ])

    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range = (1, 3), max_df=0.8, norm="l2"))
    ])

    other_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))#, feature_name_combiner=custom_combiner))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('text', text_transformer, text_cols),
            ('other', other_transformer, other_cols)
        ])

    pipeline = Pipeline(steps=[("balancing", RandomUnderSampler(random_state=1830)),
                               ('preprocessor', preprocessor),
                               #('smote', SMOTE(random_state=42, k_neighbors=20)),
                               ('selector', SelectKBest(chi2, k = 2000)),
                               ('clf', LogisticRegression(penalty = 'l2', C = .9,
                                multi_class = 'multinomial', class_weight = 'balanced',
                                random_state = 42, solver = 'newton-cg', max_iter = 100))
                               ])

    return pipeline


def train_model(
        X: pd.DataFrame,
        y: np.array,
        test_split: float=0.3,
        max_features: int=1000
    ) -> Pipeline:
    """ 
    Fit the passed model with the passed data and return a tuple (fitted_model, history)

    Parameters
    ----------
    X : pd.DataFrame
        The dataframe of features
    y : pd.Series
        The target variable
    test_split : float, optional
        _description_, by default 0.3
    max_features : int, optional
        _description_, by default 1000

    Returns
    -------
    Pipeline
        A fitted pipeline object
    res : pd.DataFrame
        A dataframe with the mean cross-validated metrics (4 in total) 
    """

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    text_cols = ["description", "amenities"]
    other_cols = list(set(X.columns) - set(numeric_cols) - set(text_cols))

    pipe_model = build_pipeline(numeric_cols, text_cols, other_cols, max_features = max_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42, stratify=y)

    print(Fore.BLUE + "\nLaunching CV" + Style.RESET_ALL)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    res = cross_validate(pipe_model, X_train, y_train, verbose=2, cv=cv, scoring=scoring)
    res = pd.DataFrame(res)

    #size_data = y_train.value_counts().sort_values(ascending=False).iloc[0] * 16
    print(f"✅ Model trained on \n {len(X_train)} original rows")
    print(f"Mean cross_validated accuracy: {round(np.mean(res.get('test_accuracy')), 2)}")
    print(f"Mean cross_validated precision: {round(np.mean(res.get('test_precision')), 2)}")

    pipe_model.fit(X_train, y_train)

    return pipe_model, res


def evaluate_model(
        model,
        X: pd.DataFrame,
        y: pd.Series,
        test_split:float=0.3
    ) -> pd.DataFrame:
    """
    Evaluate trained model performance on the dataset

    Parameters
    ----------
    model : _type_
        A sklearn model, instantiated from a pickle file or trained before. 
    X : pd.DataFrame
        The dataframe of features
    y : pd.Series
        The target variable
    test_split : float, optional
        The split ratio between train and test, by default 0.3

    Returns
    -------
    pd.DataFrame
        A dataframe with the evaluated metrics (4 metrics)
    y_pred: np.array
        A np.array of the model's prediction
    y_test: np.array
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


def predict_model(model, X: str) -> np.array:
    """_summary_

    Parameters
    ----------
    model : _type_
        _description_
    X : str
        _description_

    Returns
    -------
    np.array
        _description_
    """
    return model.predict(X), model.predict_proba(X)


def load_model(model_name: str = None) -> None:
    """_summary_

    Parameters
    ----------
    model_name : str, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
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
