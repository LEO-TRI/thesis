from scripts_thesis.data import load_data

import pandas as pd
import numpy as np
import re

from scripts_thesis.data import load_data
from scripts_thesis.cleaning import clean_target_feature

from scripts_thesis.utils import custom_combiner

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, OneHotEncoder


###Train model###
def train():
    df = load_data()
    print(df.head())
    print(df.shape)

if __name__ == "__main__":
    print("Launching model")
    print("Starting training sequence")
    train()
    print("Finished training sequence")
