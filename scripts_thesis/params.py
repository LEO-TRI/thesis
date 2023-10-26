import os

######################PATHS######################
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code", "LEO-TRI", "perso_projects", "thesis")
ROOT_PATH  = os.path.join(os.path.expanduser('~'), "thesis")
LOCAL_RAW_PATH = os.path.join(ROOT_PATH, "data", "raw_data")
LOCAL_DATA_PATH = os.path.join(ROOT_PATH, "data", "processed_data")
LOCAL_MODEL_PATH = os.path.join(ROOT_PATH, "models", "saves")
LOCAL_RESULT_PATH = os.path.join(ROOT_PATH, "models", "results", "train")
LOCAL_EVALUATE_PATH = os.path.join(ROOT_PATH, "models", "results", "evaluate")
LOCAL_COEFS_PATH = os.path.join(ROOT_PATH, "models", "results", "coefs")
LOCAL_IMAGE_PATH = os.path.join(ROOT_PATH, "models", "results", "images")

LOCAL_PATHS = [LOCAL_RAW_PATH, LOCAL_DATA_PATH, LOCAL_MODEL_PATH,
               LOCAL_RESULT_PATH, LOCAL_EVALUATE_PATH, LOCAL_COEFS_PATH,
               LOCAL_IMAGE_PATH]


logistic_dict = {'classifier__C': 3.927997954578607, 'classifier__penalty': 'l1', 'preprocessing__text__selectkbest__k': 1700, 'preprocessing__text__text_preprocessing__text1__ngram_range': (1, 1), 'preprocessing__text__text_preprocessing__text1__norm': 'l1', 'preprocessing__text__text_preprocessing__text2__ngram_range': (1, 1), 'preprocessing__text__text_preprocessing__text2__norm': 'l2', 'preprocessing__text__text_preprocessing__text3__ngram_range': (1, 3), 'preprocessing__text__text_preprocessing__text3__norm': 'l2'}
gbt_dict = {'classifier__l2_regularization': 0.9626393637566706, 'classifier__learning_rate': 0.3719726103526537, 'classifier__max_bins': 139, 'classifier__max_depth': 4, 'classifier__max_leaf_nodes': 19, 'preprocessing__text__selectkbest__k': 1500, 'preprocessing__text__text_preprocessing__text1__ngram_range': (1, 2), 'preprocessing__text__text_preprocessing__text1__norm': 'l2', 'preprocessing__text__text_preprocessing__text2__ngram_range': (1, 2), 'preprocessing__text__text_preprocessing__text2__norm': 'l2', 'preprocessing__text__text_preprocessing__text3__ngram_range': (1, 2), 'preprocessing__text__text_preprocessing__text3__norm': 'l1'}
#gbt_dict = {'classifier__l2_regularization': 0.09399567969872136, 'classifier__learning_rate': 0.03830828475716885, 'classifier__max_bins': 77, 'classifier__max_depth': 2, 'classifier__max_leaf_nodes': 31, 'pca__n_components': 180, 'preprocessing__text__selectkbest__k': 1400, 'preprocessing__text__text_preprocessing__text1__ngram_range': (1, 2), 'preprocessing__text__text_preprocessing__text1__norm': 'l1', 'preprocessing__text__text_preprocessing__text2__ngram_range': (1, 2), 'preprocessing__text__text_preprocessing__text2__norm': 'l1', 'preprocessing__text__text_preprocessing__text3__ngram_range': (1, 1), 'preprocessing__text__text_preprocessing__text3__norm': 'l1'}
#gbt_dict = {'classifier__l2_regularization': 0.9533783877176355, 'classifier__learning_rate': 0.010474942262980624,   'classifier__max_bins': 211, 'classifier__max_depth': 3,'classifier__max_leaf_nodes': 8,'preprocessing__text__selectkbest__k': 1100,'preprocessing__text__text_preprocessing__text1__ngram_range': (1, 3),'preprocessing__text__text_preprocessing__text1__norm': 'l2','preprocessing__text__text_preprocessing__text2__ngram_range': (1, 2),'preprocessing__text__text_preprocessing__text2__norm': 'l2','preprocessing__text__text_preprocessing__text3__ngram_range': (1, 1),'preprocessing__text__text_preprocessing__text3__norm': 'l2'}

random_forest_dict = {'preprocessing__text__text_preprocessing__text3__norm': 'l1', 'preprocessing__text__text_preprocessing__text3__ngram_range': (1, 1), 'preprocessing__text__text_preprocessing__text2__norm': 'l2', 'preprocessing__text__text_preprocessing__text2__ngram_range': (1, 2), 'preprocessing__text__text_preprocessing__text1__norm': 'l2', 'preprocessing__text__text_preprocessing__text1__ngram_range': (1, 3), 'preprocessing__text__selectkbest__k': 400, 'classifier__n_estimators': 230, 'classifier__min_samples_split': 35, 'classifier__min_samples_leaf': 15, 'classifier__max_leaf_nodes': 62, 'classifier__max_features': 'log2', 'classifier__max_depth': 9}
gNB_dict = {'preprocessing__text__text_preprocessing__text3__norm': 'l1', 'preprocessing__text__text_preprocessing__text3__ngram_range': (1, 3), 'preprocessing__text__text_preprocessing__text2__norm': 'l2', 'preprocessing__text__text_preprocessing__text2__ngram_range': (1, 3), 'preprocessing__text__text_preprocessing__text1__norm': 'l2', 'preprocessing__text__text_preprocessing__text1__ngram_range': (1, 1), 'preprocessing__text__selectkbest__k': 500}
stacked_dict = {'classifier__final_estimator__C': 3.6168169810197197, 'classifier__final_estimator__penalty': 'l1', 'classifier__gbt__l2_regularization': 0.6936181331909276, 'classifier__gbt__learning_rate': 0.20498488736308185, 'classifier__gbt__max_bins': 210, 'classifier__gbt__max_depth': 2, 'classifier__gbt__max_leaf_nodes': 24, 'classifier__rf__max_depth': 2, 'classifier__rf__max_features': 'sqrt', 'classifier__rf__max_leaf_nodes': 89, 'classifier__rf__min_samples_leaf': 22, 'classifier__rf__min_samples_split': 13, 'classifier__rf__n_estimators': 260, 'preprocessing__text__selectkbest__k': 1800, 'preprocessing__text__text_preprocessing__text1__ngram_range': (1, 1), 'preprocessing__text__text_preprocessing__text1__norm': 'l1', 'preprocessing__text__text_preprocessing__text2__ngram_range': (1, 1), 'preprocessing__text__text_preprocessing__text2__norm': 'l1', 'preprocessing__text__text_preprocessing__text3__ngram_range': (1, 2), 'preprocessing__text__text_preprocessing__text3__norm': 'l1'}
xgb_dict = {'classifier__booster': 'dart', 'classifier__learning_rate': 0.3973684249270416, 'classifier__max_depth': 4, 'classifier__n_estimators': 35, 'classifier__reg_alpha': 0.8866560287286961, 'preprocessing__text__selectkbest__k': 800, 'preprocessing__text__text_preprocessing__text1__ngram_range': (1, 2), 'preprocessing__text__text_preprocessing__text1__norm': 'l1', 'preprocessing__text__text_preprocessing__text2__ngram_range': (1, 2), 'preprocessing__text__text_preprocessing__text2__norm': 'l2', 'preprocessing__text__text_preprocessing__text3__ngram_range': (1, 1), 'preprocessing__text__text_preprocessing__text3__norm': 'l2'}

hyperparams_dict = dict(logistic=logistic_dict,
                        gbt=gbt_dict,
                        xgb=xgb_dict,
                        random_forest=random_forest_dict,
                        gNB=gNB_dict,
                        stacked=stacked_dict)
