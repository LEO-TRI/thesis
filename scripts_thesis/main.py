import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle

from colorama import Fore, Style

from scripts_thesis.data import DataLoader, LoadDataMixin
from scripts_thesis.model_ML import train_model, evaluate_model, predict_model, load_model
from scripts_thesis.utils import get_top_features, model_explainer, plot_confusion_matrix
from scripts_thesis.params import *
from scripts_thesis.preproc import *

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
    '''
    for file_path in LOCAL_PATHS:
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)

#####PROCESS#####
class ModelFlow(LoadDataMixin, DataLoader):
    """
    A class to manage the model flow.
    Keeps as instance information some processed data to be used for prediction
    """

    def __init__(self) -> None:
        super().__init__() #Brings back load_raw_data. Used as a test for mixin
        self.pred_data = self.preprocess() #Used to keep some prediction data unseen by the model


    def preprocess(self, model: bool=True) -> pd.DataFrame:
        """      
        Load the raw data from the raw_data folder.\n
        Save the data locally if not in the raw data folder.\n
        Process query data.\n
        Store processed data in the processed directory.\n
        Keeps 10 lines of data within the class to do some pred later on. 

        Parameters
        ----------
        model : bool, optional
            Additional preprocessing steps are taken if True, by default True

        Returns
        -------
        data_clean_pred : pd.DataFrame
            A cleaned df saved locally and that can be used for training and testing afterwards. 
        """
        print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

        # Process data
        df = self.load_raw_data() #Brings back load_raw_data. Used as a test for mixin

        data_clean = preprocess_data(df)

        if model:
            data_clean = clean_target_feature(data_clean)
            data_clean = clean_variables_features(data_clean)

        data_clean_pred = data_clean.sample(n=10, random_state=1830, replace=False)
        data_clean_train = data_clean.drop(data_clean_pred.index)

        k = len(data_clean)
        now = datetime.now()

        file_name = f"processed_{k}_rows_{now.strftime('%d-%m-%Y-%H-%M')}.parquet.gzip"
        full_file_path = os.path.join(LOCAL_DATA_PATH, file_name)

        data_clean_train.to_parquet(full_file_path,
                    compression='gzip')

        print("✅ preprocess() done \n Saved localy")
        
        return data_clean_pred

    #####MODEL#####
    def train(file_name: str = None, target: str = "license", test_split: float = 0.3) -> None:
        """
        Load data from the data folder.\n
        Train the instantiated model on the train set.\n
        Store training results and model weights as a pickle and a csv respectively.\n

        Parameters
        ----------
        file_name : str, optional
            The file from which to load the processed data. If none, will load the last file, by default None
        target : str, optional
            The target y of the model, by default "license"
        test_split : float, optional
            The train test split, by default 0.3
        """

        print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
        print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

        data_processed = DataLoader.load_processed_data(file_name=file_name)
        if data_processed is None: #Used to exit the function and trigger an error if load_processed_data fails
            return None

        y= data_processed[target].astype(int)
        X = data_processed.drop(columns=[target])

        print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)
        model, res = train_model(X, y, test_split)

        model_iteration = len(os.listdir(LOCAL_MODEL_PATH)) + 1
        file_name = f'model_V{model_iteration}.pkl'
        full_file_path = os.path.join(LOCAL_MODEL_PATH, file_name)
        pickle.dump(model, open(full_file_path, 'wb'))

        file_name = f'model_train_V{model_iteration}'
        full_file_path = os.path.join(LOCAL_RESULT_PATH, file_name)
        res.to_csv(full_file_path)

    def model_viz()-> None:
        '''
        Method that produces a graph showing main words used to predict SDG categories.\n
        Takes the last saved ML model as input by default.\n
        Saves a csv file with the model coefficient + an jpeg version.\n
        Coordinates for saved documents are given in params (LOCAL_COEFS_PATH and LOCAL_IMAGE_PATH respectively).
        '''
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

    def evaluate(file_name:str = None,
        target:str = "sdg"
        ) -> pd.DataFrame:
        """
        Evaluate the performance of the latest production model on processed data.\n
        Return accuracy, recall, precision and f1 as a pd.DataFrame.
        """
        print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

        data_processed = DataLoader.load_processed_data(file_name=file_name)
        if data_processed is None:
            return None

        y= data_processed[target]
        X = data_processed["lemma"]

        model = load_model()
        results, y_pred, y_test = evaluate_model(model, X, y)

        plot_confusion_matrix(y_test, y_pred)

        model_iteration = len(os.listdir(LOCAL_EVALUATE_PATH)) + 1
        file_name = f'model_evaluate_V{model_iteration}'

        full_file_path = os.path.join(LOCAL_EVALUATE_PATH, file_name)
        results.to_csv(full_file_path)

        print("✅ evaluate() done \n")

        return results

    def pred(X_pred:pd.DataFrame = None) -> np.array:
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
            X_pred = np.array(
                ["The UN debated a new plan to increase poverty-relief efforts in poor and emerging countries",
                "Results of the conference on the protection of biodiversity have stalled, with measures for large mammals especially problematic"
                ]
                    )
            
        print("\n⭐️ Use case: predict")

        model = load_model()
        assert model is not None

        X_pred = X_pred.drop(columns=column_selector(X_pred))
        X_pred = clean_variables_features(X_pred)
        if len(X_pred) == 0:
            print("Error on the input X_pred, please review your input")
            return None

        y_pred, y_pred_proba = predict_model(model, X_pred)


        print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
        print("\n✅ Proba: ", y_pred_proba, "\n")

        return y_pred


if __name__ == '__main__':
    #agreement, target = main()
    local_setup()
    print("✅ Setup done")
    preprocess()
    print("✅ Process done")
    #train(target=target)
    print("✅ Train done")
    #model_viz()
    print("✅ Viz created")
    #evaluate(target=target)
    print("✅ Evaluate done")
    #pred()
    print("✅ Pred done")
