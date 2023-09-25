########################### ML TEMPLATE ##############################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report, make_scorer
from sklearn.ensemble import HistGradientBoostingClassifier , RandomForestClassifier, StackingClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import pandas as pd
import numpy as np
import time
import os
import pickle

from scripts_thesis.utils import custom_combiner, params_combiner, sparse_to_dense
from scripts_thesis.params import *

from colorama import Fore, Style

def print_results(y_test: np.ndarray, y_pred: np.ndarray, verbose: bool= True, fold: int=None) -> dict:
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
    verbose: bool
        Whether to print the results, by default True
    fold: int
        The fold on which the data was train, by default None

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

    if fold is not None:
        metrics['fold'] = fold + 1

    if verbose:
        print(Fore.BLUE + f"Accuracy: {metrics.get('accuracy')}"+ Style.RESET_ALL)
        print(Fore.BLUE + f"Precision: {metrics.get('precision')}"+ Style.RESET_ALL)
        print(Fore.BLUE + f"Recall: {metrics.get('recall')}"+ Style.RESET_ALL)
        print(Fore.BLUE + f"F1 Score: {metrics.get('f1')}"+ Style.RESET_ALL)

    return metrics


def baseline_model(y: np.ndarray, test_split: float=0.3) -> np.ndarray:
    """
    Function computing the baseline to beat by the new model.

    Produces two baseline, one coming from a random guess and the other from predicting only the majority class

    sParameters
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

    print(Fore.BLUE + "\n ⭐️ Results to beat" + Style.RESET_ALL)
    y_train, y_test = train_test_split(y, test_size=test_split, random_state=42, stratify=y)

    print(Fore.BLUE + "\n Result for random baseline" + Style.RESET_ALL)
    y_random = np.random.randint(0, 2, size=len(y_test))
    print_results(y_test, y_random)

    print(Fore.BLUE + "\n Result for majority baseline" + Style.RESET_ALL)
    y_majority = np.zeros(len(y_test))
    print_results(y_test, y_majority)

    return y_random, y_majority


def build_pipeline(numeric_cols: list[str], text_cols: list[str], other_cols: list[str],
                   classifier: str='logistic', max_features_tfidf: int=10000, max_kbest: int=1000) -> Pipeline:
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
    classifier : str, optional
        The classifier to use ('logistic', 'gbt', 'random_forest'), by default 'logistic'
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
            ('text1', TfidfVectorizer(max_features=max_features_tfidf, ngram_range = (1, 3), max_df=0.8, norm="l1", strip_accents="unicode"), text_cols[0]),
            ('text2', TfidfVectorizer(max_features=max_features_tfidf, ngram_range = (1, 3), max_df=0.8, norm="l1", strip_accents="unicode"), text_cols[1]),
            ('text3', TfidfVectorizer(max_features=max_features_tfidf, ngram_range = (1, 3), max_df=0.8, norm="l1", strip_accents="unicode"), text_cols[2])
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

    # Create the final preprocessing pipeline
    pipeline = Pipeline([
        ("balancing", RandomUnderSampler(random_state=1830)),
        ('preprocessing', column_transformer)]
                        )

    #('smote', SMOTE(random_state=42, k_neighbors=20)),

    #Add the "head" of the pipeline from the potential classifiers
    classifiers = dict(logistic= LogisticRegression(penalty='l2', C=0.9, class_weight='balanced', random_state=1830, solver='liblinear', max_iter=1000),
                       gbt= HistGradientBoostingClassifier(random_state=1830),
                       random_forest= RandomForestClassifier(random_state=1830, class_weight="balanced"),
                       sgd= SGDClassifier(random_state=1830, eta0=1)
                       )
    

    if classifier == "stacked":
        estimators = [('rf', classifiers.get("random_forest")),
                      ("gbt", classifiers.get("gbt")),
                      ("sgd", classifiers.get("sgd"))
                      ]
        clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
        classifiers[classifier] = clf

    classifier_model = classifiers.get(classifier, None)
    if classifier not in classifiers.keys():
        raise ValueError("Invalid classifier name. Choose 'logistic', 'gbt', 'random_forest', 'sgd' or 'stacked'.")

    if (classifier == "gbt") | (classifier == 'stacked'): #Adding an additional step for classifiers that require dense array
        sparse_to_dense_transformer = FunctionTransformer(func=sparse_to_dense, validate=False)
        pipeline.steps.append(['dense', sparse_to_dense_transformer])

    pipeline.steps.append(['classifier', classifier_model]) #Adding a classifier head

    return pipeline

def tune_model(X: pd.DataFrame, y: pd.Series, max_features: int=1000, n_iter: int=20, cv: int=5, classifier: str='logistic') -> Pipeline:
    """
    Tune a machine learning model with hyperparameter optimization.

    Parameters
    ----------
    X : pd.DataFrame
        The feature matrix.
    y : pd.Series
        The target variable.
    max_features : int, optional
        The maximum number of features for tf-idf vectorization, by default 1000.
    n_iter : int, optional
        The number of iterations for hyperparameter optimization, by default 20.
    cv : int, optional
        The number of folds for each combination of hyperparameters on which to cross validate
    classifier : str, optional
        The classifier to use in the pipeline ('logistic', 'gbt', or 'random_forest', 'sgd' or 'stacked'), by default 'logistic'.

    Returns
    -------
    Pipeline : imblearn.pipeline.Pipeline/sklearn.pipeline.Pipeline
        A scikit-learn pipeline containing a tuned machine learning model.
    """

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    text_cols = ["amenities", "description", "host_about"]
    other_cols = list(set(X.columns) - set(numeric_cols) - set(text_cols))

    pipe_model = build_pipeline(numeric_cols, text_cols, other_cols, max_features_tfidf = max_features, classifier=classifier)

    pipe_params = params_combiner(classifier=classifier)
    scoring = dict(AUC="roc_auc",
                   accuracy=make_scorer(accuracy_score),
                   precision=make_scorer(precision_score)
                   )

    rand_search = RandomizedSearchCV(pipe_model,
                                     param_distributions=pipe_params,
                                     cv=cv,
                                     n_iter=n_iter,
                                     scoring=scoring,
                                     refit="precision",
                                     random_state=1830,
                                     verbose=2)

    rand_search.fit(X, y)
    print(Fore.BLUE + f"Precision is :{rand_search.best_score_}" + Style.RESET_ALL )

    return rand_search


def train_model(X: pd.DataFrame, y: pd.Series, test_split: float=0.3, max_features: int=1000, n_splits: int = 5) -> Pipeline:
    """
    Fit the passed model with the passed data and return a tuple (fitted_model, history)

    Parameters
    ----------
    X : pd.DataFrame
        The dataframe of features
    y : pd.Series
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
        A dataframe with the mean cross-validated metrics (4 in total + the fold number)
    """

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    text_cols = ["amenities", "description", "host_about"]
    other_cols = list(set(X.columns) - set(numeric_cols) - set(text_cols))

    pipe_model = build_pipeline(numeric_cols, text_cols, other_cols, max_features_tfidf = max_features)

    print(Fore.BLUE + "\nLaunching CV" + Style.RESET_ALL)

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=2, random_state=42)

    res = []
    pred_list = []
    test_list = []

    for fold, (train, test) in enumerate(cv.split(X, y)):
        start_time = time.time()  # Record the start time
        pipe_model.fit(X.loc[train,:], y[train])
        y_pred = pipe_model.predict(X.loc[test,:])

        res.append(print_results(y[test], y_pred, verbose=False, fold=fold))

        pred_list.append(y_pred)
        test_list.append(y[test])

        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time

        print(f"CV Number {fold + 1} done. Time elapsed: {elapsed_time:.2f} seconds")

    res = pd.DataFrame(res)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=1830, stratify=y)
    pipe_model.fit(X_train, y_train)

    #size_data = y_train.value_counts().sort_values(ascending=False).iloc[0] * 16
    print(f"✅ Model trained on \n {len(X_train)} original rows")
    print(f"Mean cross_validated accuracy: {round(np.mean(res['accuracy']), 2)}")
    print(f"Mean cross_validated precision: {round(np.mean(res['precision']), 2)}")

    return pipe_model, res, (pred_list, test_list)

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
