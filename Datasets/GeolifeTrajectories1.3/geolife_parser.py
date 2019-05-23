"""
The code in this file is received from: https://heremaps.github.io/pptk/tutorials/viewer/geolife.html
"""
import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os


# Some user directories have labels.txt, which indicates the transport mode used for a specific duration.
# However, the labelled duration don't fully encompass all the points in all trips by a user. 
# 0 is reserved for such points, whereas np.nan is when none of the trajectory by a given user has labels (user directory has no labels.txt)
mode_names = ['walk', 'bike', 'bus', 'car', 'subway','train', 'airplane', 'boat', 'run', 'motorcycle', 'taxi']
mode_ids = {s : i + 1 for i, s in enumerate(mode_names)}

def get_transport_label(t_id):
    return mode_names[int(t_id)-1]

def get_transport_id(t_label):
    return mode_ids[t_label]

def read_plt(plt_file, user_id):
    trip_id = plt_file.split("/")[-1].split(".")[0]
    points = pd.read_csv(plt_file, skiprows=6, header=None,
                         parse_dates=[[5, 6]], infer_datetime_format=True)

    # for clarity rename columns
    points.rename(inplace=True, columns={'5_6': 'time', 0: 'lat', 1: 'lon', 3: 'alt'})

    # remove unused columns
    points.drop(inplace=True, columns=[2, 4])
    
    points["trip_id"] = user_id + trip_id

    return points

def read_labels(labels_file):
    labels = pd.read_csv(labels_file, skiprows=1, header=None,
                         parse_dates=[[0, 1], [2, 3]],
                         infer_datetime_format=True, delim_whitespace=True)

    # for clarity rename columns
    labels.columns = ['start_time', 'end_time', 'transport_mode']

    # replace 'label' column with integer encoding
    labels['transport_mode'] = [mode_ids[i] for i in labels['transport_mode']]

    return labels

def apply_labels(points, labels):
    indices = labels['start_time'].searchsorted(points['time'], side='right') - 1
    no_label = (indices < 0) | (points['time'].values >= labels['end_time'].iloc[indices].values)
    points['transport_mode'] = labels['transport_mode'].iloc[indices].values
    points['transport_mode'][no_label] = 0

def read_user(user_folder, user_id):
    labels = None

    plt_files = glob.glob(os.path.join(user_folder, 'Trajectory', '*.plt'))
    df = pd.concat([read_plt(f, user_id) for f in plt_files])

    labels_file = os.path.join(user_folder, 'labels.txt')
    if os.path.exists(labels_file):
        labels = read_labels(labels_file)
        apply_labels(df, labels)
    else:
        df['transport_mode'] = np.nan

    return df

def read_all_users(folder):
    subfolders = glob.glob(os.path.join(folder, '*'))
    dfs = []
    for i, sf in enumerate(subfolders):
        user_id = sf.split("/")[-1]
        print('[%d/%d] processing user %s' % (i + 1, len(subfolders), user_id))
        df = read_user(sf, user_id)
        df['user_id'] = user_id
        dfs.append(df)
    return pd.concat(dfs)