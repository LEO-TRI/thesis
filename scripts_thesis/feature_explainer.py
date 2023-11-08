import pandas as pd
import numpy as np
from scripts_thesis.data import DataLoader
from scripts_thesis.model_ML import load_model, train_model
from scripts_thesis.graphs import feature_importance_plotting, feature_cols_plotting
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score


class FeatureExplainer():

    def __init__(self, model, features, coefs=None, cols=None):
        self.model = model
        self.features = features
        self.coefs = coefs
        self.cols = cols
        self.permut_coefs = None
        self.fig = None

    @staticmethod
    def check_has_importance(model):
        
        features = model[:-2].get_feature_names_out()

        if hasattr(model[-1], "feature_importances_"):
        
            coefs = model[-1].feature_importances_
            coef_df = pd.DataFrame(np.vstack((features, coefs)).T, columns = ["features", "coefs"])
            coef_df = coef_df.astype({"coefs": float, "features": str}).sort_values("coefs", ascending=False)
            features = coef_df["features"].to_numpy()
            coefs = coef_df["coefs"].to_numpy()

        else: 
            coefs = None
        
        return (features, coefs)

    @classmethod
    def from_model(cls, classifier, is_show: bool = False):
        model = load_model(classifier = classifier)

        features, coefs = FeatureExplainer.check_has_importance(model)

        fe = FeatureExplainer(model, features, coefs)

        
        fig = fe.plot_top_features(fe.coefs)
        if is_show:
            fig.show()

        return fe 

        
    @classmethod
    def from_train(cls, file_name: str=None, target: str="license", test_split: float=0.3, is_show: bool = False):

        X, y = DataLoader.prep_data(file_name=file_name, target=target)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=1830, stratify=y)
        model, _, _ = train_model(X_train, y_train)
        cols = X_test.columns.to_list()


        features, coefs = FeatureExplainer.check_has_importance(model)
        fe = FeatureExplainer(model, features, coefs, cols)

        fig = fe.plot_top_features(fe.coefs)
        if is_show:
            fig.show()

        return fe 
    
    def compute_permutation(self, 
                            X_test: np.ndarray, 
                            y_test: np.ndarray,
                            n : int = 14,
                            n_repeat: int = 5, 
                            is_show: bool = False,
                            threshold: float = 0.5) -> tuple[np.ndarray]:
        
        features = X_test.columns.to_list()[:n]

        y_pred_proba = self.model.predict_proba(X_test)[:,1]
        y_pred = np.where(y_pred_proba>=threshold, 1, 0)

        score = fbeta_score(y_test, y_pred, beta = 0.5)

        res = np.ones((len(features), n_repeat))

        for i, feature in enumerate(features):
            
            shuffled_df = X_test.copy()

            #TODO Add a feature transfo step

            print(f"Testing column {feature}")

            for repeat in range(n_repeat):

                shuffled_df[feature] = shuffled_df[feature].sample(frac=1).to_numpy()
                y_pred_proba = self.model.predict_proba(X_test)[:,1]
                y_pred = np.where(y_pred_proba>=threshold, 1, 0)
                res[i, repeat] = fbeta_score(y_test, y_pred, beta = 0.5)

        res_features_cv = np.mean(res, axis=1)
        res_final = -1 * res_features_cv + score

        self.fig = self.plot_top_columns(features, res_final)
        if is_show:
            self.fig.show()

        self.permut_coefs = res_final
        self.cols = features

        return (features, res_final)


    def plot_top_features(self, coefs):

        self.fig = feature_importance_plotting(coefs)

        return self.fig
    
    def plot_top_columns(self, cols, coefs):

        self.fig = feature_cols_plotting(cols, coefs)
        return self.fig

        

        
