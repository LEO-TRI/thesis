########################### ML TEMPLATE ##############################
from sklearn.experimental import enable_iterative_imputer #Required to import IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import FeatureUnion#, Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import HistGradientBoostingClassifier , RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

import xgboost as xgb
import time
from scripts_thesis.utils import sparse_to_dense
from scripts_thesis.model_ML import print_results


class AdvancedPipeline():
    def __init__(self, classifier: list[str], numeric_cols: list[str], text_cols: list[str], other_cols: list[str], size: int=2) -> None:
        self.classifier = classifier
        self.model = self.combine_branches(classifier,
                                           numeric_cols,
                                           text_cols,
                                           other_cols,
                                           size)

    def build_preprocessing(numeric_cols: list[str],
                            text_cols: list[str],
                            other_cols: list[str],
                            max_features_tfidf: int=10000,
                            max_kbest: int=1000) -> Pipeline:
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
                ('num', numeric_transformer, numeric_cols),],
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

        # Create the final preprocessing pipeline. Further steps can be added with append later
        pipeline = Pipeline([
            ("balancing", RandomUnderSampler(random_state=1830)),
            ('preprocessing', column_transformer),
                            ]
                            )

        return pipeline


    def add_classifier(self, classifier, pipeline):
        #Set the "head" of the pipeline from the potential classifiers
        classifiers = dict(logistic= LogisticRegression(penalty='l2', C=0.9, random_state=1830, solver='liblinear', max_iter=1000, class_weight="balanced"),
                        gbt= HistGradientBoostingClassifier(random_state=1830),
                        random_forest= RandomForestClassifier(random_state=1830, class_weight="balanced"),
                        sgd= SGDClassifier(random_state=1830, max_iter=1000),
                        xgb=xgb.XGBClassifier(random_state=1830, tree_method="hist"),
                        gNB = GaussianNB()
                        )

        if classifier == "stacked":
            estimators = [('rf', classifiers.get("random_forest")),
                        ("gbt", classifiers.get("gbt")),
                        ("gNB", classifiers.get("gNB"))
                        ]

        #Adding the stacked classifier to the dict of classifiers
            clf = StackingClassifier(estimators=estimators, final_estimator=classifiers.get("logistic"))
            classifiers[classifier] = clf

        if classifier not in classifiers.keys():
            raise ValueError("Invalid classifier name. Choose 'logistic', 'gbt', 'random_forest', 'gNB', 'xgb' or 'stacked'.")

        classifier_model = classifiers.get(classifier, None)

        #Adding an additional step for classifiers that require dense array
        if (classifier == "gbt") | (classifier == 'stacked') | (classifier == "gNB") | (classifier == "xgb"):
            sparse_to_dense_transformer = FunctionTransformer(func=sparse_to_dense, validate=False)
            pipeline.steps.append(['dense', sparse_to_dense_transformer])

        pipeline.steps.append(['classifier', classifier_model])

    def combine_branches(self,
                         classifier: list[str],
                         numeric_cols: list[str],
                         text_cols: list[str],
                         other_cols: list[str],
                         size: int=2):

        if isinstance(classifier, str):
            classifier = [classifier] * size

        full_model = []

        for i, model in enumerate(classifier):
            pipe = self.build_preprocessing(numeric_cols, text_cols, other_cols)
            pipe = self.add_classifier(model, pipe)
            full_model.append((f'pipe_{i}', pipe))

        VotingClassifier(estimators=full_model, voting="soft")

        return VotingClassifier(estimators=full_model, voting="soft")

    def make_output(self, X, y, n_splits: int=5):

        pipe_model = self.model
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=2, random_state=42)

        res = []
        pred_list = []
        test_list = []

        #Filters for models that cannot produce probabilities estimates
        has_proba = True

        for fold, (train, test) in enumerate(cv.split(X, y)):
            start_time = time.time()  # Record the start time

            pipe_model.fit(X.iloc[train,:], y[train])
            y_pred = pipe_model.predict(X.iloc[test,:])
            res.append(print_results(y[test], y_pred, verbose=False, fold=fold))

            test_list.append(y[test])

            if has_proba:
                y_proba = pipe_model.predict_proba(X.loc[test,:])
                pred_list.append(y_proba[:,1])
            else :
                pred_list.append(y_pred)


            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time  # Calculate elapsed time

            print(f"CV Number {fold + 1} done. Time elapsed: {elapsed_time:.2f} seconds")
