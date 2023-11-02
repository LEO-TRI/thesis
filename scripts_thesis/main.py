import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle
import json

from colorama import Fore, Style

from scripts_thesis.data import DataLoader, LoadDataMixin
from scripts_thesis.model_ML import train_model, evaluate_model, predict_model, load_model, tune_model
from scripts_thesis.utils import get_top_features, params_extracter
from scripts_thesis.graphs import model_explainer, plot_confusion_matrix, graphs_cross_val, probability_distribution, feature_importance_plotting

from scripts_thesis.params import *
from scripts_thesis.preproc import *

from sklearn.model_selection import train_test_split

import tensorflow as tf

#####LAUNCH#####
def main():
    '''
    Method to input the parameters for the programme.
    '''

    print(Fore.MAGENTA + "\n ⭐️ Do you want to use default parameter? (agreement=0.8, target=sdg)" + Style.RESET_ALL)

    yes = bool(int(input("Enter 0 for no and 1 for yes: ")))

    if not yes:
        agreement = float(input("Enter agreement value (float between 0 and 1s): "))
        target = input("Enter target (sdg or esg): ")

    return agreement, target

#####SETUP#####
def local_setup()-> None:
    '''
    Method to create the directories for the package.

    Takes and returns no objects

    Relies on the files in LOCAL_PATHS to do so
    '''
    for file_path in LOCAL_PATHS:
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)

#####PROCESS#####
class ModelFlow(LoadDataMixin, DataLoader):
    """
    A class to manage the model flow.

    Keeps as instance information some processed data to be used for prediction.

    Inherits the loading functions from DataLoader thanks to the mixin LoadDataMixin
    """

    def __init__(self, file_name: str=None, target="license", test_split: float=0.3) -> None:
        """
        Initiates a ModelFlow class. ModelFlow is used to avoid data leakages
        by holding train and test sets and then pulling on them depending on needs.

        Parameters
        ----------
        file_name : str, optional
            The file from which to pull data, by default None
        target : str, optional
            The target feature, by default "license"
        test_split : float, optional
            The train/test split, by default 0.3
        """
        super().__init__() #Brings back load_raw_data. Used as a test for mixin
        self.file_name = None
        self.best_threshold = dict()
        self.test_data = None
        self.test_size=test_split
        self.load_model = load_model

        X, y = self.prep_data(file_name=file_name, target=target)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=1830, stratify=y)
        self.test_data = (X_test, y_test)
        self.train_data = (X_train, y_train)

    def preprocess(self, has_model: bool=True) -> pd.DataFrame:
        """
        Load the raw data from the raw_data folder.\n
        Save the data locally if not in the raw data folder.\n
        Process query data.\n
        Store processed data in the processed directory.\n
        Keeps 10 lines of data within the class to do some pred later on.

        Parameters
        ----------
        has_model : bool, optional
            Additional preprocessing steps are taken if True, by default True

        Returns
        -------
        data_clean_train : pd.DataFrame
            A cleaned df saved locally and that can be used for training and testing afterwards.
        data_clean_pred: pd.DataFrame
            A cleaned df saved in the class and that is used for predicted afterwards
        """

        print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

        # Process data
        df = self.load_raw_data() #Brings back load_raw_data. Used as a test for mixin
        #reviews = self.load_raw_data(target="reviews")

        data_clean = preprocess_data(df)
        #data_clean = data_clean.sample(frac=0.1)

        df_attendance = pd.read_csv("/home/leotricaud/code/LEO-TRI/perso_projects/thesis/data/attendance.csv", sep=";")
        df_attendance.columns = ["venue", "visitors", "latitude", "longitude"]

        coordinates_tree = (df_attendance[["latitude"]].values, df_attendance[["longitude"]].values)
        coordinates_targets = (data_clean[["latitude"]].values, data_clean[["longitude"]].values)

        data_clean["monument"] = add_distance(coordinates_tree, coordinates_targets)

        print(Fore.MAGENTA + "\n preprocess Part 1 done"+ Style.RESET_ALL)


        if has_model:
            data_clean = clean_target_feature(data_clean)
            data_clean = clean_variables_features(data_clean) #, reviews)

        data_clean_pred = data_clean.sample(n=10, random_state=1830, replace=False)
        data_clean_train = data_clean.drop(index=data_clean_pred.index)

        k = len(data_clean)
        now = datetime.now()

        file_name = f"processed_{k}_rows_{now.strftime('%d-%m-%Y-%H-%M')}.parquet.gzip"
        full_file_path = os.path.join(LOCAL_DATA_PATH, file_name)

        data_clean_train.to_parquet(full_file_path, compression='gzip')

        print("✅ preprocess() done \n Saved localy")

        return data_clean_pred

    #####MODEL#####
    def optimise(self,
                 classifier: list[str]=None,
                 n_iter: int=50):
        """
        A method to perform hyperparameters tuning on several classifiers

        Returns a fitted and optimised classifier

        Parameters
        ----------
        classifiers : list[str], optional
            The classifier to use ('logistic', 'gbt', 'random_forest'), if None will test all, by default None
            Must be passed as a list even if only one classifier is passed.
        n_iter: int, optional
            Number of iterations from RandomizedSearch, by default 50

        Returns
        -------
        Pipeline
            A sklearn pipeline
        """

        print(Fore.MAGENTA + "\n⭐️ Use case: optimise" + Style.RESET_ALL)
        print(Fore.MAGENTA + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

        X_train, y_train = self.train_data
        if X_train is None: #Used to exit the function and trigger an error if load_processed_data fails
            return None

        if classifier is None:
            classifier=["logistic", "gbt", "random_forest", "sgd", "gNB", "xgb", "stacked"]

        if type(classifier)==str: #Converts to list format for iteration
            classifier = [classifier]

        print(Fore.MAGENTA + f"\nTuning {len(classifier)} model(s)..." + Style.RESET_ALL)

        tuned_results = {key: tune_model(X_train, y_train, n_iter=n_iter, classifier=key) for key in classifier} #Test the pipeline with hyperparameters for three potential classifiers
        tuned_results = {key: [model.best_params_, params_extracter(model), model] for key, model in tuned_results.items()} #Extract the best results, and parameters from the fitted pipelines
        tuned_results = {key: value for key, value in sorted(tuned_results.items(), key= lambda x : x[1][1].get("precision"), reverse=True)}

        print(Fore.MAGENTA + "Models' results are:" + Style.RESET_ALL)
        for key in tuned_results.keys():
            tuned_results.get(key)[1] = {key: np.round(value, 2) for key, value in tuned_results.get(key)[1].items()}
            print(f"{key} : {tuned_results.get(key)[1]}\n")

        best_model_ind = list(tuned_results.keys())[0] #Select the key of the best model

        model_iteration = len(os.listdir(LOCAL_MODEL_PATH)) + 1
        file_name = f'model_train_{best_model_ind}_V{model_iteration}'
        full_file_path = os.path.join(LOCAL_MODEL_PATH, file_name)

        print(tuned_results.get(best_model_ind)[0])
        print(type(tuned_results.get(best_model_ind)[0]))

        with open(full_file_path, 'wb') as f:
            pickle.dump(tuned_results.get(best_model_ind)[0], f)

        return tuned_results.get(best_model_ind) #Return the best model


    def train(self,
              file_name: str = None,
              n_splits: int=5,
              classifier: str="logistic",
              is_rebalance: bool=False,
              is_save_graph: bool=False) -> None:
        """
        Load data from the data folder.

        Train the instantiated model on the train set.

        Store training results and model weights as a pickle and a csv respectively.

        Parameters
        ----------
        file_name : str, optional
            The file from which to load the processed data. If none, will load the last file, by default None
        target : str, optional
            The target y of the model, by default "license"
        test_split : float, optional
            The train test split, by default 0.3
        classifier : str, optional
            The classifier to use in the pipeline ('logistic', 'gbt', 'random_forest', 'gNB', 'xgb', or 'stacked'), by default 'logistic'.
        is_save_graph: bool, by default False
            Whether to save the outputed graphs locally
        """

        print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)

        print(Fore.MAGENTA + "\nLoading preprocessed validation data..." + Style.RESET_ALL)
        X_train, y_train = self.train_data

        print(Fore.MAGENTA + "\nTraining model..." + Style.RESET_ALL)

        model, results, auc_metrics = train_model(X_train, y_train, classifier=classifier, n_splits=n_splits, is_rebalance=is_rebalance) #auc_metrics = {test_list, proba_list}
        fig1, fig2, best_threshold = graphs_cross_val(auc_metrics) #Producing the cross_val metrics
        self.best_threshold[classifier] = best_threshold

        print(Fore.MAGENTA + "\n ✅ Training finished, saving model and graph..." + Style.RESET_ALL)

        model_iteration = len(os.listdir(LOCAL_MODEL_PATH)) + 1
        file_name = f'model_V{model_iteration}_{classifier}.pkl'
        full_file_path = os.path.join(LOCAL_MODEL_PATH, file_name)
        pickle.dump(model, open(full_file_path, 'wb'))

        if is_save_graph:
            file_name = f'model_train_V{model_iteration}_{classifier}'
            full_file_path = os.path.join(LOCAL_RESULT_PATH, file_name)
            results.to_csv(full_file_path)

            filename1 = f'auc_curve_{model_iteration}_{classifier}'
            full_file_path = os.path.join(LOCAL_IMAGE_PATH, filename1)
            fig1.write_image(file =full_file_path, format="png")

            filename2 = f'prc_curve_{model_iteration}_{classifier}'
            full_file_path = os.path.join(LOCAL_IMAGE_PATH, filename2)
            fig2.write_image(file =full_file_path, format="png")

        print(Fore.MAGENTA + "\nSaving done..." + Style.RESET_ALL)
        print(Fore.MAGENTA + "\n  ✅ Training accomplished, well done Captain..." + Style.RESET_ALL)

    def evaluate(self, file_name: str = None, classifier: str="logistic") -> pd.DataFrame:
        """
        Evaluate the performance of the latest production model on processed data.\n

        Return accuracy, recall, precision and f1 as a pd.DataFrame.

        Parameters
        ----------
        file_name : str, optional
            The file from which to load the data, by default None
        target : str, optional
            The feature column, by default "license"

        Returns
        -------
        pd.DataFrame
            A pd.DataFrame integrating the results from the evaluation
        """

        print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

        #TODO Understand filename not working. file_name = class(MlFlow) instead of none

        X_test, y_test = self.test_data

        model = self.load_model(classifier = classifier)
        best_threshold = self.best_threshold.get(classifier)

        if classifier == "xgb":
            features = model["classifier"].feature_importances_
            fig = feature_importance_plotting(features)
            fig.show()

        results, y_pred, y_test = evaluate_model(model, X_test, y_test, threshold = best_threshold)
        y_test = y_test.to_numpy()

        plot_confusion_matrix(y_test, y_pred, threshold = best_threshold)
        probability_distribution(y_test, y_pred)
        #metrics_on_one_plot(y_test, y_pred)

        model_iteration = len(os.listdir(LOCAL_EVALUATE_PATH)) + 1
        file_name = f'model_evaluate_V{model_iteration}'

        full_file_path = os.path.join(LOCAL_EVALUATE_PATH, file_name)
        results.to_csv(full_file_path)

        print("✅ evaluate() done \n")

        return results

    def pred(self, X_pred:pd.DataFrame = None) -> np.array:
        """
        Make a prediction using the latest trained model and provided data.

        Parameters
        ----------
        X_pred : pd.DataFrame, optional
            The dataframe with the rows to predict, by default None
            If none, some built-in examples will be used.

        Returns
        -------
        y_pred : np.array
            An array of predicted values based on X_pred
        """

        if X_pred is None:
            X_pred = self.pred_data #10 rows of the original data removed during preprocessing and never seen by the model.

        print("\n⭐️ Use case: predict")

        model = load_model()
        assert model is not None

        X_pred = X_pred.drop(columns=["license"])
        if len(X_pred) == 0:
            print("Error on the input X_pred, please review your input")
            return None

        y_pred, y_pred_proba = predict_model(model, X_pred)


        print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
        print("\n✅ Proba: ", y_pred_proba, "\n")

        return y_pred

    @staticmethod
    def model_viz()-> None:
        """
        Notes
        -------
        Method that produces a graph showing main features used to predict fraud.

        Takes the last saved ML model as input by default.

        Saves a csv file with the model coefficient + an jpeg version.

        Coordinates for saved documents are given in params (LOCAL_COEFS_PATH and LOCAL_IMAGE_PATH respectively).
        """

        model = load_model()

        df = get_top_features(model)
        df = df.sort_values(['SDG', 'coef'], ignore_index = True)

        model_iteration = len(os.listdir(LOCAL_MODEL_PATH)) + 1
        file_name = f'coefs_model_V{model_iteration}.csv'

        full_file_path = os.path.join(LOCAL_COEFS_PATH, file_name)
        df.to_csv(full_file_path, index=False)

        fig = model_explainer(df=df)

        file_name = f'coefs_model_V{model_iteration}.jpeg'
        full_file_path = os.path.join(LOCAL_IMAGE_PATH, file_name)
        fig.write_image(full_file_path)


if __name__ == '__main__':
    #agreement, target = main()
    local_setup()
    print("✅ Setup done")
    ml = ModelFlow()
    ml.pred_data = ml.preprocess() #Used to keep some prediction data unseen by the model
    print("✅ Process done")
    ml.train()
    print("✅ Train done")
    ModelFlow.model_viz()
    print("✅ Viz created")
    ml.evaluate()
    print("✅ Evaluate done")
    #pred()
    print("✅ Pred done")
