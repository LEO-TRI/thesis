import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle

from colorama import Fore, Style

from scripts_thesis.data import load_raw_data, load_processed_data
from scripts_thesis.model_ML import train_model, evaluate_model, predict_model, load_model
#from scripts_thesis.cleaning import clean_predict
from scripts_thesis.params import *
from scripts_thesis.preproc import *

#####LAUNCH#####
def main(agreement=0.8, target="sdg"):
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
def preprocess(model:bool=True) -> None:
    """
    Load the raw data from the raw_data folder.\n
    Save the data locally if not in the raw data folder.\n
    Process query data.\n
    Store processed data in the processed directory.\n
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Process data
    df = load_raw_data()
    data_clean = preprocess_data(df)

    if model:
        data_clean = clean_target_feature(data_clean)
        data_clean = clean_variables_features(data_clean)

    k = len(data_clean)
    now = datetime.now()

    file_name = f"processed_{k}_rows_{now.strftime('%d-%m-%Y-%H-%M')}.parquet.gzip"
    full_file_path = os.path.join(LOCAL_DATA_PATH, file_name)

    data_clean.to_parquet(full_file_path,
                compression='gzip')


    print("✅ preprocess() done \n Saved localy")

#####MODEL#####
def train(file_name:str = None,
          target:str = "license",
          test_split:float = 0.3) -> None:

    """
    Load data from the data folder.\n
    Train the instantiated model on the train set.\n
    Store training results and model weights.\n
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    data_processed = load_processed_data(file_name=file_name)
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

    df = get_top_features(model['tf_idf'], model['clf'], model['selector'], how = 'long')
    df = df.sort_values(['SDG', 'coef'], ignore_index = True)

    model_iteration = len(os.listdir(LOCAL_MODEL_PATH)) + 1
    file_name = f'coefs_model_V{model_iteration}.csv'

    full_file_path = os.path.join(LOCAL_COEFS_PATH, file_name)
    df.to_csv(full_file_path, index=False)

    fig = sdg_explainer(df=df)

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

    data_processed = load_processed_data(file_name=file_name)
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

    #X_pred = clean_predict(X_pred)
    y_pred, y_pred_proba = predict_model(model, X_pred)

    sdg_dict = DataProcess().sdg
    sdg_dict = {int(key): value for key, value in sdg_dict.items()}

    print("\n✅ prediction done: ", y_pred, [sdg_dict[pred] for pred in y_pred], y_pred.shape, "\n")
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
