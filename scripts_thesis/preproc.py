import numpy as np
import pandas as pd
from scripts_thesis.cleaning import clean_vec

def clean_variables_features(df: pd.DataFrame, features: [str] =None) -> pd.DataFrame:

    upper_bound = np.quantile(df["price"], 0.75) * 10
    mask = (df["price"]< upper_bound) & (5<df["price"]) & (df["number_of_reviews"]!=0)
    df = df.loc[mask,:].reset_index(drop=True)

    df["host_identity_verified"] = np.where(df["host_identity_verified"]=="t", 1, 0)
    df["host_is_superhost"] = np.where(df["host_is_superhost"]=="t", 1, 0)

    df["nb_amenities"] = df["amenities"].map(lambda row: len(row.split(",")))
    df["amenities"] = clean_vec(df["amenities"].values)

    df["description"] = df["description"].fillna(" ")
    df["description"] = clean_vec(df["description"].values)

    if features is None:
        features = ["host_is_superhost", "host_identity_verified",
            "accommodates", "beds", "price", "number_of_reviews",
            "review_scores_rating", "amenities", "license",
            "reviews_per_month", "neighbourhood_cleansed",
            "host_listings_count", "description", "host_about"]

    df = df.loc[:, features]
    return df.fillna(value=np.nan)


def clean_target_feature(df:pd.DataFrame) -> pd.DataFrame:

    df = df[df["room_type"] == "Entire home/apt"]

    mask = (df.license.isna())
    df_na = df[mask]
    df = df[~mask]

    mask = (df["license"]=='Available with a mobility lease only ("bail mobilité")') | (df["license"]=="Exempt - hotel-type listing")
    df_valid = df.loc[mask]
    df = df[~mask]

    mask = df.license.map(lambda row: len(row)==13)
    df_valid = pd.concat([df_valid, df.loc[mask]])
    df = df[~mask]

    mask = ~df.sort_values("first_review").duplicated(subset="license").values
    df_valid = pd.concat([df_valid, df.loc[mask]])
    df_na = pd.concat([df_na, df.loc[~mask]])

    df_valid["license"] = 0
    df_na["license"] = 1

    return pd.concat([df_valid, df_na], axis=0).reset_index(drop=True)
