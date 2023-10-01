import numpy as np
import pandas as pd
import geopandas as gpd
import os
import glob
import re

from scripts_thesis.params import *

###Methods for descriptive part####

class GraphLoader:
    """
    Convenience class used to load the dataset used for graph representation at different
    levels of processing.
    """

    def get_geodata(self) -> pd.DataFrame:
        tmp_ = gpd.read_file(os.path.join("data", 'quartier_paris.geojson'))
        tmp_["c_quinsee"] = pd.to_numeric(tmp_["c_quinsee"])
        return tmp_

    def get_data(self, is_graph: bool=True) -> pd.DataFrame:
        """
        This function returns a pd.DataFrame.

        Parameters
        ----------
        is_graph : bool, optional
            A boolean determining whether additional preprocessing will be done, by default True

        Returns
        -------
        df_graph : pd.DataFrame
            A DataFrame suitable for graphical exploration
        """

        df_graph = pd.read_parquet(os.path.join("data", 'df_merged_geo.parquet.gzip'))

        if is_graph:
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

    def clean_data(self, graph: bool=True) -> pd.DataFrame:
        """
        This function returns a pd.DataFrame.

        Parameters
        ----------
        is_graph : bool, optional
            A boolean determining whether additional preprocessing will be done, by default True

        Returns
        -------
        df_graph : pd.DataFrame
            A DataFrame suitable for graphical exploration
        """

        df_graph = self.get_data(graph=graph)
        df_graph["arr"] = df_graph["c_quinsee"].map(lambda row: str(row)[2:4])

        upper_bound = np.quantile(df_graph["price"], 0.75) * 10
        mask = (df_graph["price"]< upper_bound) & (5<df_graph["price"]) & (df_graph["number_of_reviews"]!=0)

        return df_graph.loc[mask,:].reset_index(drop=True)

    def get_occupancy_rate(self) -> pd.DataFrame:
        """
        This function returns a pd.DataFrame using self.clean_data() and adds survival rates to it

        Returns
        -------
        df_graph : pd.DataFrame
            A DataFrame suitable for graphical exploration
        """

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

    def get_neighourhood(self) -> pd.DataFrame:
        """
        This function returns a pd.DataFrame using self.clean_data() and adds geopgraphical data using self.get_geodata()

        Returns
        -------
        pd.DataFrame
            A DataFrame suitable for graphical exploration
        """
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

    def get_folium(self) -> pd.DataFrame:
        df_graph = self.clean_data()
        df_geo = self.get_geodata()

        df_graph = df_graph.groupby("l_qu").agg(count_properties=("l_qu", np.count_nonzero),
                                       price = ("price", np.mean),
                                       c_quinsee = ("c_quinsee", np.mean)
                                       )

        return df_geo.merge(df_graph, on = "c_quinsee")

    def get_arr(self, df) -> pd.DataFrame:
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

###Cleaning raw data###
class DataLoader:
    """
    A class centralising data loading methods
    """

    def __init__(self, path: str = LOCAL_RAW_PATH, target :str="listing"):
        """
        Parameters
        ----------
        path : str, optional
            The path of the raw data file, by default LOCAL_RAW_PATH stored in params.py
        target : str, optional
            The type of data being seeked
        """
        self.path = path
        self.target = target

    def load_folder(self, path: str = None, target :str="listing") -> [pd.DataFrame]:
        """
        A function to faciliate iterations over all available csvs in a given folder

        Parameters
        ----------
        path : str, optional
            The path to the directory with the raw data, if None defaults to the LOCAL_RAW_DATA_PATH, by default None
        target : str, optional
            The type of data being seeked, by default listing

        Returns
        -------
        [pd.DataFrame]
            A list of pd.DataFrame

        Yields
        ------
        Iterator[[pd.DataFrame]]
            One instance of the processed dataframe. Yields 1 by iteration over the list of csv files
        """
        if path is None:
            path = self.path

        files = glob.glob(path + '/*.csv.gz')
        for f in files:
                # get filename
                stock = os.path.basename(f)
                if len(re.findall(target, stock))>0:
                    temp_df = pd.read_csv(f)
                    # create new column with filename
                    temp_df['ticker'] = stock
                    temp_df['ticker'] = temp_df['ticker'].replace('.csv.gz', '', regex=True)
                    temp_df['ticker'] = temp_df['ticker'].replace('listings_', '', regex=True)
                    yield temp_df

    def load_raw_data(self, target :str="listing") -> pd.DataFrame:
        """
        A function to convert the list of dataframe yielded by load_folder()

        Leverages load_folder() and its yield structure to concatenate [pd.DataFrame] into 1 pd.DataFrame

        Parameters
        ----------
        target : str, optional
            The type of data being seeked, by default listing

        Returns
        -------
        pd.DataFrame
            The full pd.DataFrame of raw data
        """
        if len(os.listdir(self.path))>0:
            li = [df for df in self.load_folder(target=target)]

        else:
            full_file_path = os.path.join(os.getcwd(), "data")
            li = [df for df in self.load_folder(full_file_path, target)]

        return pd.concat(li)

    #Building target#
    @staticmethod
    def load_processed_data(file_name: str= None) -> pd.DataFrame:
        """
        A convenience function to load the processed data

        Parameters
        ----------
        file_name : str, optional
            The file to be loaded, if None returns the latest saved, by default None

        Returns
        -------
        data_processed : pd.DataFrame
            The loaded DataFrame
        """

        if file_name==None:
            full_file_path = os.path.join(LOCAL_DATA_PATH, "None")
        else:
            full_file_path = os.path.join(LOCAL_DATA_PATH, file_name)

        if not os.path.exists(full_file_path):
            files = [os.path.join(LOCAL_DATA_PATH, file) for file in os.listdir(LOCAL_DATA_PATH) if file.endswith(".parquet.gzip")]

            if len(files) == 0:
                print("No processed data, please use preprocess first")
                return None #Used to exit the function

            print("No corresponding parquet file, returning latest saved parquet")
            full_file_path = max(files, key=os.path.getctime)

        data_processed = pd.read_parquet(full_file_path)

        if data_processed.shape[0] < 10:
            print("❌ Not enough processed data retrieved to train on")
            return None #Used to exit the function

        return data_processed.reset_index(drop=True)

    def prep_data(self, file_name: str= None, target: str= "license") -> tuple[pd.DataFrame, pd.Series]:
        """
        A convenience function leveraging load_processed_data in the same class to provide additional processing.

        Exists to avoid cluttering main.py

        Parameters
        ----------
        file_name : str, optional
            the path of the file with the data, if None, will default to the latest file, by default None
        target : str, optional
            The target feature to be defined as y, by default "license"

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            A tuple containing the pd.DataFrame X of features and the pd.Series y of the target
        """
        df = self.load_processed_data(file_name=file_name)
        if df is None: #Used to exit the function and trigger an error if load_processed_data fails
            return None, None

        y= df[target].astype(int)
        X = df.drop(columns=[target])

        return X, y

class LoadDataMixin():
    """
    A test class to check how mixins work
    Is supposed to allow the ModelFlow class in main.py to inherit a method from DataLoader
    """
    def load_raw_data(self, target :str= "listing"):
        return super().load_raw_data(target)

    def prep_data(self, file_name: str= None, target: str= "license"):
        return super().prep_data(file_name, target)
