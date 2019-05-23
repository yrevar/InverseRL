#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from geolife_parser import *
from geolife_data import upsample_df

# %matplotlib inline
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import pandas as pd
if os.path.exists("geolife.pkl"):
    data = pd.read_pickle('geolife.pkl')  # reads 'geolife.pkl' into df
else:
    data = read_all_users("./Data")
    data.to_pickle('geolife.pkl')


# In[3]:


data.head()


# There are several trips in which users have used multiple transport modes. To identify them uniquely let's augment trip_ids with transport_mode.

# In[4]:


data["trip_id"] = data["trip_id"] + "#" +  data["transport_mode"].astype(str)


# In[5]:


data.head()


# In[7]:


print("Transport modes: ", list(map(lambda x: get_transport_label(x), sorted(data.transport_mode.dropna().unique().astype(np.int)))))


# We're looking for trajectories with interesting behaviors, so that we can learn those behaviors. Now the question is, what is interesting? 
# 
# Let's consider a 2d state space. Are straight lines interesting? Of course, no. Straight lines don't tell us anything new that we don't already know. Any trivial reward function can explain such trajectories e.g., $R(s) = 0, \forall s \in S$. For all trajectories that can be explained without knowing the reward function, in other words, those that can be explained assuming a constant reward function over the state space, we don't need Markov Decision Processes. It's because there's no utility to be maximized, they are simply processes which under the Markov assumption can also be studied without any animal interaction.
# 
# To illustrate, consider a sink in some gravitational environment, the water is going to go down in it. There's a lot of value in modeling that, but in the contex of IRL we want to learn from expert demonstrations, so modeling human behavior is very useful. We don't want to model inanimate things, we want to model intelligent animal behavior and learn from it.
# 
# To do this, we're going to have to filter interesting trajectories first. To keep filtering simple, let's only consider trajectories with transport mode specified, and let's further limit ourself to only "car" trajectories.

# In[8]:


data = data[~pd.isna(data.transport_mode)]
data.transport_mode = data.transport_mode.astype(np.int).values


# In[9]:


# let's use string representation for transport mode
data.loc[:, "transport_mode"] = data["transport_mode"].apply(lambda x: get_transport_label(x)).values


# In[10]:


data.head()


# In[11]:


def plot_trips_by_user(data):
    trips_by_user = data.groupby(["user_id"])["trip_id"].nunique()
    trips_by_user.name = "ntrip"
    ax = trips_by_user.plot(kind="bar", figsize=(16,10), title="Trips by User");
    ax.set_ylabel("Number of trips")
    return trips_by_user

def plot_trips_by_transport(data):
    trips_by_transport = data.groupby(["transport_mode"])["trip_id"].nunique()
    trips_by_transport.name = "ntrip"
    ax = trips_by_transport.plot(kind="bar", figsize=(16,10), title="Trips by Transport Mode");
    ax.set_ylabel("Number of trips")
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    return trips_by_transport


# In[12]:


trips_by_user = plot_trips_by_user(data)


# In[13]:


trips_by_transport = plot_trips_by_transport(data)


# ## Select car trajectories

# In[14]:


select_transport_mode = "car"
data = data[data.transport_mode == select_transport_mode]


# In[15]:


print("No. of {} trajectories: {}".format(select_transport_mode, data.trip_id.nunique()))


# In[16]:


data.to_csv("./geolife_car_trips.csv")

print("Done.")