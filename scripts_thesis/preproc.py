import pandas as pd
import numpy as np
from thefuzz import fuzz

from scripts_thesis.cleaning import CleanData, SpacyClean
from scripts_thesis.distance_calculation import CoordinateCounterTree

from tqdm import tqdm
import string

cd = CleanData()
sc = SpacyClean()

def column_selector(df: pd.DataFrame) -> list:
    """
    A convenience function used to determine which columns to drop from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The initial dataframe

    Returns
    -------
    drop_list : list
        A list of column names
    """
    df_summary = df.describe(include="all").transpose()
    drop_list = [elem for elem in df_summary.index if df_summary.loc[elem, "count"] ==0]
    drop_list = drop_list + ["neighborhood_overview", "listing_url", "scrape_id", "host_url", "host_picture_url",\
                            "picture_url", "neighbourhood", "host_neighbourhood", "host_thumbnail_url",\
                            "host_has_profile_pic", "picture_url", "host_name", "neighborhood_overview",\
                            "host_location", "host_response_time", "bathrooms_text", "calendar_last_scraped", "source"]

    return drop_list

def text_selector(df:pd.DataFrame)-> pd.DataFrame:
    df["host_about"] = df["host_about"].replace(np.nan, "")
    df["mask"] = df["host_about"].map(lambda row: len(row))

    temp_df = df.groupby("host_id").agg(count_obj=("name", "count"))
    df = df.merge(temp_df, on="host_id", how="left")
    unique_val = df.groupby(["host_about"]).agg(mask = ("mask", np.mean)).reset_index()

    unique_val["group"] = np.nan
    unique_val["host_about_2"] = unique_val["host_about"].map(lambda row: cd.remove_proper_nouns(row))
    unique_val["mask"] = unique_val["host_about_2"].map(lambda row: len(row))

    unique_val = unique_val[unique_val["mask"]>50].reset_index(names="index_base")

    lengths = unique_val.iloc[:, 2]  # Precompute the length values
    df_copy = unique_val.copy()

    for ind, elem in tqdm(enumerate(unique_val["host_about_2"])):
        temp_len = lengths[ind] # Use precomputed length value
        temp_df = df_copy[(df_copy["mask"] < temp_len + 15) & (df_copy["mask"] > temp_len - 15)]

        ratios = temp_df["host_about_2"].map(lambda row: fuzz.ratio(elem, row))  # Use vectorized operations
        temp_mask = ratios > 75
        similar_texts = temp_df.loc[temp_mask, ["host_about", "host_about_2"]]
        similar_indices = temp_df[temp_mask].index

        df_copy = df_copy.drop(similar_indices, axis=0)

        if len(similar_texts["host_about_2"])>1:
            similar_texts["group"] = ind
            similar_texts = similar_texts.drop_duplicates("host_about_2")
            mask = unique_val["host_about_2"].isin(similar_texts["host_about_2"])
            unique_val.loc[mask, "group"] = unique_val.loc[mask, "host_about"].map(similar_texts.set_index("host_about")["group"])

    return unique_val

def text_desc_selector(df:pd.DataFrame) -> pd.DataFrame:
    unique_val_de = df[df["count_obj"]==1]
    unique_val_de["description"] = unique_val_de["description"].replace(np.nan, "")
    unique_val_de["mask"] = unique_val_de["description"].map(lambda row: len(row))
    unique_val_de = unique_val_de.groupby(["description"]).agg(count_obj_de=("name", np.count_nonzero),
                                                mask = ("mask", np.mean)).reset_index()

    return unique_val_de

def create_occupancy_rate(df:pd.DataFrame)-> pd.DataFrame:
    df.loc[:, ["first_review", "last_review"]] = df[["first_review", "last_review"]].apply(pd.to_datetime, axis=0)
    df["time_difference"] = (df["last_review"] - df['first_review']) / np.timedelta64(1, 'D')
    df["nb_of_nights_cons"] = df["number_of_reviews"] / 0.5
    df["nb_of_nights_large"] = df["number_of_reviews"] / 0.72

    df["occupancy_rate_cons"] = df["nb_of_nights_cons"] * df["minimum_nights"] / df["time_difference"]
    df["occupancy_rate_large"] = df["nb_of_nights_large"] * df["minimum_nights"] / df["time_difference"]
    df["occupancy_rate_large"] = np.where(df["occupancy_rate_large"]>1, 1, df["occupancy_rate_large"])
    df["occupancy_rate_cons"] = np.where(df["occupancy_rate_cons"]>1, 1, df["occupancy_rate_cons"])

    df["exit"] = df["last_review"] + np.timedelta64(6, 'M')
    df["entry"] = df["first_review"] - np.timedelta64(6, 'M')
    return df

def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:

    df = df.drop(column_selector(df), axis=1)

    df= df.sort_values(["id", "last_scraped"])
    df = df.drop_duplicates(subset =["id", "host_id", "license"])
    print("Step 1 done")

    df["price"] = cd.clean_price_vec(df["price"])
    #upper_bound = np.quantile(df["price"], 0.75) * 10
    #mask = (df["price"]< upper_bound) & (5<df["price"]) & (df["number_of_reviews"]!=0)
    #df = df.loc[mask,:].reset_index(drop=True)

    df_text = df.drop_duplicates(subset =["id"]).copy()
    temp_df = df.groupby("host_id").agg(count_obj=("name", "count")).reset_index()
    df_text = df_text.merge(temp_df.loc[:, ["count_obj", "host_id"]], on="host_id", how="left")
    print("Step 2 done")

    unique_val = text_selector(df_text) #Selecting unique_val
    df_merged = df_text.merge(unique_val.loc[:, ["host_about", "group"]], on="host_about", how="left")
    print("Step 3 done")

    unique_val_de = text_desc_selector(df_merged) #Selecting unique_val_de
    mask = df_merged["description"].isin(unique_val_de["description"])
    condition = (mask) & (df_merged["count_obj"] < df_merged["description"].map(unique_val_de.set_index("description")["count_obj_de"]))
    df_merged.loc[condition, "count_obj"] = df_merged.loc[condition, "description"].map(unique_val_de.set_index("description")["count_obj_de"])
    print("Step 4 done")

    df_merged["group"] = np.where(df_merged["group"].isna(), 0, df_merged["group"])
    condition = (df_merged["group"]==0) & (df_merged["count_obj"]>1)

    temp_df = df_merged.loc[condition, ["host_id", "description", "name", "host_about"]]
    temp_df = temp_df.merge(temp_df.groupby("host_id").agg(count_obj_host=("name", np.count_nonzero)).reset_index())
    temp_df = temp_df.merge(temp_df.groupby("description").agg(count_obj_desc=("name", np.count_nonzero)).reset_index())
    temp_df.loc[temp_df["count_obj_desc"]==66, "count_obj_desc"]= 1
    print("Step 5 done")

    #Merging groups
    condition = temp_df["count_obj_desc"]<temp_df["count_obj_host"]

    ref = max(df_merged["group"])
    mask = temp_df[condition]

    for ind, elem in enumerate(mask["host_id"].unique()):
        df_merged.loc[df_merged["host_id"]==elem, "group"] = ref + ind
        df_merged.loc[df_merged["host_id"]==elem, "count_obj"] =mask.loc[mask["host_id"]==elem, "count_obj_host"]

    ref = max(df_merged["group"])
    mask = temp_df[condition == False]

    for ind, elem in enumerate(mask["description"].unique()):
        df_merged.loc[df_merged["description"]==elem, "group"] = ref + ind
        df_merged.loc[df_merged["description"]==elem, "count_obj"] = mask.loc[mask["description"]==elem, "count_obj_desc"]

    for ind, elem in enumerate(df_merged["group"].unique()):
        df_merged.loc[df_merged["group"]==elem, "group"] = ind

    temp_df = df_merged["group"].value_counts().reset_index()
    temp_df.columns = ["group", "group_count"]

    df_merged = df_merged.merge(temp_df, on="group", how="left")
    df_merged["group"] = df_merged["group"].astype(int)

    df_merged = create_occupancy_rate(df_merged)

    return df_merged

###Additional preproc for model preparation###

def clean_variables_features(df: pd.DataFrame, reviews: pd.DataFrame = None,
                             features: [str] = None, train_set:bool = True) -> pd.DataFrame:

    if train_set:
        upper_bound = np.quantile(df["price"], 0.75) * 10
        df["description"] = df["description"].fillna(" ")

        mask = (df["price"]< upper_bound) & (5<df["price"]) & (df["number_of_reviews"]!=0) & (df["description"].map(lambda row : len(row)>1))
        df = df.loc[mask,:].reset_index(drop=True)

    df = df.set_index("id")
    #reviews = reviews.set_index("listing_id")[["comments"]]
    #df = df.join(reviews, how="left")

    df["host_identity_verified"] = np.where(df["host_identity_verified"]=="t", 1, 0)
    df["host_is_superhost"] = np.where(df["host_is_superhost"]=="t", 1, 0)

    df["nb_amenities"] = df["amenities"].map(lambda row: len(row.split(",")))
    df["amenities"] = cd.clean_vec(df["amenities"].values)

    df["description"] = df["description"].fillna(" ")

    df['word_count'] = [len(review.split()) for review in df['description']]
    df['upper_char_count'] = [sum(char.isupper() for char in review) for review in df['description']]
    df['special_char_count'] = [sum(char in string.punctuation for char in review) for review in df['description']]

    df["description"] = df["description"].str.replace("license number", "")
    df["description"] = cd.clean_vec(df["description"].values)
    df["description"] = [text for text in sc.preprocess_spacy(df["description"].values)]

    df["host_about"] = df["host_about"].fillna(" ")

    df['word_count_host'] = [len(review.split()) for review in df['host_about']]
    df['upper_char_count_host'] = [sum(char.isupper() for char in review) for review in df['host_about']]
    df['special_char_count_host'] = [sum(char in string.punctuation for char in review) for review in df['host_about']]

    df["has_host_about"] = np.where(df["host_about"].map(lambda row : len(row)>20), 1, 0)
    df["host_about"] = cd.clean_vec(df["host_about"].values)
    df["host_about"] = [text for text in sc.preprocess_spacy(df["host_about"].values)]

    #df["comments"] = cd.clean_vec(df["comments"].values)
    #df["comments"] = sc.preprocess_spacy(df["comments"].values)

    if features is None:
        features = ["host_is_superhost", "host_identity_verified",
            "accommodates", "beds", "price", "number_of_reviews",
            "review_scores_rating", "amenities", "license",
            "reviews_per_month", "neighbourhood_cleansed",
            "host_listings_count", "description", "has_host_about",
            "host_about", "word_count", "upper_char_count", "special_char_count",
            "occupancy_rate_cons", "monument"] #, "comments"]

    df = df.loc[:, features]

    return df.fillna(value=np.nan)

def add_distance(coordinates_tree: tuple, coordinates_targets: tuple) -> list:

    cct = CoordinateCounterTree.from_data_points(*coordinates_tree)

    target_lat, target_long = coordinates_targets
    coordinates = np.concatenate([target_lat, target_long], axis=1)

    res = cct.calculate_points_within_distance(coordinates, distance_km=2.5)
    count = [len(val) for val in res]
    return count


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
