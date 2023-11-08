import pandas as pd
import numpy as np
from scripts_thesis.data import DataLoader
from scripts_thesis.model_ML import load_model, train_model

from sklearn.model_selection import train_test_split

class FeatureExplainer():

    def __init__(self, model):
        self.model = model

    @classmethod
    def from_model(cls, classifier):
        model = load_model(classifier = classifier)

        fe = FeatureExplainer(model)
        fe.plot_top_features()



    @classmethod
    def from_train(cls, file_name: str=None, target: str="license", test_split: float=0.3):

        X, y = DataLoader.prep_data(file_name=file_name, target=target)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=1830, stratify=y)
        model, _, _ = train_model(X_train, y_train)

        fe = FeatureExplainer(model)

    def plot_top_features(self):
        features = self.model[:-2].get_feature_names_out()
        coefs = self.model[-1].feature_importances_

        res = pd.DataFrame(np.vstack((features, coefs)).T, columns = ["features", "coefs"])
        res["coefs"] = res["coefs"].astype(float)
