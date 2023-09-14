import numpy as np
import pandas as pd
import geopandas as gpd
import os

from scripts_thesis.preproc import *

###Methods for descriptive part####

class GraphLoader:

    def get_geodata(self):
        tmp_ = gpd.read_file(os.path.join("data", 'quartier_paris.geojson'))
        tmp_["c_quinsee"] = pd.to_numeric(tmp_["c_quinsee"])
        return tmp_

    def get_data(self, graph=True):
        """
        This function returns a Python dict.
        Its keys should be 'sellers', 'orders', 'order_items' etc...
        Its values should be pandas.DataFrames loaded from csv files
        """

        df_graph = pd.read_parquet(os.path.join("data", 'df_merged_geo.parquet.gzip'))

        if graph:
            df_graph = df_graph.loc[:, ["id", "group", "group_count", "c_quinsee", "license",
                                "room_type", "count_obj", "price", "description", "number_of_reviews",
                                "first_review", "last_review", "minimum_nights", "review_scores_rating",
                                'host_id', 'calculated_host_listings_count', 'property_type', 'availability_365']]

        df_geo = self.get_geodata()

        df_geo["c_quinsee"] = pd.to_numeric(df_geo["c_quinsee"])
        df_graph["c_quinsee"] = pd.to_numeric(df_graph["c_quinsee"])
        df_graph = df_graph.merge(df_geo.loc[:, ["l_qu", "c_quinsee"]],
                          on="c_quinsee")

        return df_graph

    def clean_data(self, graph:bool=True):
        df_graph = self.get_data(graph=graph)
        df_graph["arr"] = df_graph["c_quinsee"].map(lambda row: str(row)[2:4])

        upper_bound = np.quantile(df_graph["price"], 0.75) * 10
        mask = (df_graph["price"]< upper_bound) & (5<df_graph["price"]) & (df_graph["number_of_reviews"]!=0)

        return df_graph.loc[mask,:].reset_index(drop=True)

    def get_occupancy_rate(self):
        df_time = self.clean_data()

        df_time.loc[:, ["first_review", "last_review"]] = df_time[["first_review", "last_review"]].apply(pd.to_datetime, axis=0)
        df_time["time_difference"] = (df_time["last_review"] - df_time['first_review']) / np.timedelta64(1, 'D')

        df_time["nb_of_nights_cons"] = df_time["number_of_reviews"] / 0.5
        df_time["nb_of_nights_large"] = df_time["number_of_reviews"] / 0.72

        df_time["occupancy_rate_cons"] = df_time["nb_of_nights_cons"] * df_time["minimum_nights"]/df_time["time_difference"]
        df_time["occupancy_rate_large"] = df_time["nb_of_nights_large"]* df_time["minimum_nights"]/df_time["time_difference"]

        df_time["exit"] = df_time["last_review"] + np.timedelta64(3, 'M')
        df_time["entry"] = df_time["first_review"] - np.timedelta64(3, 'M')

        return df_time

    def get_neighourhood(self):
        df_graph = self.clean_data()
        df_geo = self.get_geodata()

        df_graph = df_graph.groupby("l_qu").agg(count_properties=("l_qu", np.count_nonzero),
                                       price = ("price", np.mean),
                                       c_quinsee = ("c_quinsee", np.mean)
                                       )

        df_graph = df_graph.merge(df_geo, on = "c_quinsee")
        df_graph["price_bin"] = pd.qcut(df_graph['price'], q=[0, .2, .4, .6, .8, 1],
                                        labels=["Q" + str(i + 1) for i, num in enumerate(np.linspace(0, 100, num=5))])
        df_graph = df_graph.sort_values("price", ascending=False)
        df_graph["surface"] = df_graph["surface"].map(lambda row: np.round(row, 2))

        return df_graph

    def get_folium(self):
        df_graph = self.clean_data()
        df_geo = self.get_geodata()

        df_graph = df_graph.groupby("l_qu").agg(count_properties=("l_qu", np.count_nonzero),
                                       price = ("price", np.mean),
                                       c_quinsee = ("c_quinsee", np.mean)
                                       )

        return df_geo.merge(df_graph, on = "c_quinsee")

    def get_arr(self, df):
        df_geo = self.get_geodata()
        df = df[df["room_type"]=="Entire home/apt"]
        df_graph = df.groupby("l_qu").agg(count_properties=("l_qu", np.count_nonzero),
                                price = ("price", np.mean),
                                c_quinsee = ("c_quinsee", np.mean)
                                )

        df_graph = df_graph.merge(df_geo, on = "c_quinsee")
        df_graph["price_bin"] = pd.qcut(df_graph['price'], q=[0, .2, .4, .6, .8, 1],
                                        labels=["Q" + str(i + 1) for i, num in enumerate(np.linspace(0, 100, num=5))])
        df_graph = df_graph.sort_values("price", ascending=False)
        df_graph["surface"] = df_graph["surface"].map(lambda row: np.round(row, 2))

        df_arr = df_graph.groupby("c_ar").agg(price=("price", "mean"),
                                            surface=("surface", "sum"),
                                            count_properties = ("count_properties", "sum")).reset_index()

        return df_arr

###Methods for predictive parts####

#Building target#
def load_data() -> pd.DataFrame:

    df = pd.read_parquet(os.path.join("data", 'df_merged_geo.parquet.gzip'))
    df = clean_target_feature(df)
    df = clean_variables_features(df)

    return df
