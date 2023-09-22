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