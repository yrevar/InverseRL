import numpy as np
import os, os.path as osp
import pandas as pd
import seaborn as sns
from .geospatial_utils import df_crop_trips
from mapsplotlib import mapsplot as mplt


def find_dirs_with_labels(dataset_dir):
    """Find trajectory directories containing "labels.txt"."""
    labeled_dirs = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file == "labels.txt":
                labeled_dirs.append(root)
    return labeled_dirs


def get_dataframe_grouped_by_user(traj_dir_with_labels,
                                  select_user=None,
                                  process_labels=False):
    """Parse trajectories and return data frame grouped by user."""
    assert (isinstance(traj_dir_with_labels, list))
    data = []
    for traj_dir in traj_dir_with_labels:

        print("Processing ", traj_dir)
        labels_file = os.path.join(traj_dir, "labels.txt")
        df_labels = parse_trajectory_labels(labels_file)
        t_dir = os.path.join(traj_dir, "Trajectory")
        user_id = traj_dir.split("/")[-1]

        if select_user is not None and user_id != select_user:
            continue

        for file in os.listdir(t_dir):

            if file.endswith(".plt"):
                df_traj = parse_trajectory_file(
                    os.path.join(t_dir, file))
                df_traj.dropna(inplace=True)
                df_traj["transport_mode"] = np.nan
                df_traj["trip_id"] = file.split(".")[0]
                df_traj["user_id"] = user_id
                label_exists = not (df_traj.iloc[0]["date_time"] >
                                    df_labels.iloc[-1]["end_time"] or
                                    df_traj.iloc[-1]["date_time"]
                                    < df_labels.iloc[0]["start_time"])
                if label_exists and process_labels:
                    for index, row in df_labels.iterrows():
                        mask = (df_traj['date_time'] >= row["start_time"]) \
                               & (df_traj['date_time'] <= row["end_time"])
                        if sum(mask) > 0:
                            df_traj.at[mask, "transport_mode"] = row["transport_mode"]
                data.append(df_traj)
    return pd.concat(data, ignore_index=True)


def parse_trajectory_file(file_path):
    """Parse trajectory file "<trip_id>.plt"."""
    return pd.read_csv(file_path,
                       skiprows=6, usecols=[0, 1, 3, 4, 5, 6],
                       parse_dates={'date_time': [4, 5]},
                       infer_datetime_format=True,
                       header=None, names=["latitude", "longitude", "x",
                                           "altitude", "n_days", "date",
                                           "time"])


def parse_trajectory_labels(file_path):
    """Parse trajectory labels file "labels.txt"."""
    return pd.read_csv(file_path, parse_dates=[0, 1],
                       infer_datetime_format=True, sep='\t',
                       skiprows=1, header=None,
                       names=["start_time", "end_time", "transport_mode"])


class GeoLifeDataset:

    def __init__(self, dir="../datasets/GeolifeTrajectories1.3/",
                 parsed_hdf5_file="geolife_data_parsed.h5",
                 parsed_hdf5_name="/geolife_trajectories_labelled",
                 process_labels=True):
        """Parse geolife data grouped by user.
            Store the datatframe in HDF store to speed up subsequent retrievals.
        """
        self.dataset_dir = dir
        data_store_dir = osp.join(dir, parsed_hdf5_file)
        store = pd.HDFStore(data_store_dir)
        if parsed_hdf5_name in store.keys():
            print("Loading from HDFS store: {}: {}".format(data_store_dir, parsed_hdf5_name))
            data = store[parsed_hdf5_name]
        else:
            print("Creating HDFS store from data: {}".format(dir))
            dirs_with_labels = find_dirs_with_labels(dir)
            data = get_dataframe_grouped_by_user(
                dirs_with_labels, process_labels)
            store[parsed_hdf5_name] = data
        store.close()
        self.update_data_cache(self.__cleanup_data_frame(data))

    def __cleanup_data_frame(self, df):
        df = df.fillna({"transport_mode": "N/A"})
        # Sort by Date Time
        df = df.sort_values(by="date_time")
        """
        Some taxi samples have multiple samples recorded on single timestamp
        E.g.,
        2008-10-31 10:43:08	39.919698	116.349964	20081031101923	taxi
        2008-10-31 10:43:11	39.919694	116.349964	20081031101923	taxi
        2008-10-31 10:43:11	39.919693	116.349963	20081031101923	taxi
        2008-10-31 10:43:13	39.919690	116.349963	20081031101923	taxi
        """
        df.drop_duplicates(subset=["date_time"], keep="last", inplace=True)
        return df

    def get_user_ids(self):
        return np.unique(self.data.user_id)

    def get_trip_ids(self):
        return np.unique(self.data.trip_id)

    def get_transport_modes(self):
        return np.unique(self.data.transport_mode)

    def select_user_ids(self, user_id_list, update_data_cache=False):
        df = self.data[self.data["user_id"].isin(user_id_list)]
        if update_data_cache:
            self.update_data_cache(df)
        return df

    def select_transport_modes(self, transport_mode_list, update_data_cache=False):
        df = self.data[self.data["transport_mode"].isin(transport_mode_list)]
        if update_data_cache:
            self.update_data_cache(df)
        return df

    def crop_by_lat_lng(self, lat=39.9059631, lng=116.391248,
                        lat_span_miles=10, lng_span_miles=10,
                        update_data_cache=False):
        cropped_df, boundary_coords = df_crop_trips(self.data, lat, lng,
                                  lat_span_miles=lat_span_miles,
                                  lng_span_miles=lng_span_miles)
        if update_data_cache:
            self.update_data_cache(cropped_df)
        return cropped_df, boundary_coords

    def update_data_cache(self, new_df):
        print("Updating data cache...")
        self.data = new_df
        self.lat_min = self.data.latitude.min()
        self.lat_max = self.data.latitude.max()
        self.lng_min = self.data.longitude.min()
        self.lng_max = self.data.longitude.max()
        # # filter by transport mode
        # self.tmode_by_user = pd.DataFrame(
        #     self.data.groupby("user_id")["transport_mode"].value_counts())
        # self.tmode_by_user.columns = ["counts"]
        # self.tmode_by_user.reset_index(inplace=True)
        # self.tmode_by_user_pivot = self.tmode_by_user.pivot(
        #     index="user_id", columns="transport_mode", values="counts")

    def plot_scatter(self, title="", kind="hex", height=10, **kwargs):
        g = sns.jointplot(data=self.data, x="longitude", y="latitude", kind=kind, height=height, **kwargs)
        g.fig.suptitle(title + "\nlat: [{:.4f}, {:.4f}], lng: [{:.4f}, {:.4f}]".format(
            self.lat_min, self.lat_max, self.lng_min, self.lng_max))
        return g

    def update_gmap_api_key(self, api_key):
        mplt.register_api_key(api_key)

    def plot_gmap_density(self, latitude_list, longitude_list, **kwargs):
        mplt.density_plot(latitude_list, longitude_list, **kwargs)