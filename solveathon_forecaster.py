#!/usr/bin/env python
# coding: utf-8

# In[67]:


from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import *
import numpy as np
import os
import matplotlib.pyplot as plt


# In[68]:


df = pd.read_csv(r'C:\Users\susha\OneDrive\Desktop\dataset.csv', encoding='latin-1')
df.head()


# In[70]:


features = df[['hour', 'month', 'day', 'stationName']]


# In[71]:


stations = df['stationName'].unique()
days = sorted(df['day'].unique())

# Station encoding
station_le = LabelEncoder()
stations_int = station_le.fit_transform(stations)
station_ohe = OneHotEncoder().fit(stations_int.reshape(-1, 1))

# Days encoding
days_le=  LabelEncoder()
days_int = days_le.fit_transform(days)
days_ohe = OneHotEncoder().fit(days_int.reshape(-1, 1))


# In[72]:


features.head()


# In[73]:


def create_feature(df):
    day_label_encoded = days_le.transform(df['day'])
    day_feature = days_ohe.transform(day_label_encoded.reshape(-1, 1)).toarray()
    station_label_encoded = station_le.transform(df['stationName'])
    station_feature = station_ohe.transform(station_label_encoded.reshape(-1, 1)).toarray()
    time_features = df[['hour', 'month']].values
    return np.concatenate([day_feature, time_features, station_feature], axis=1)


# In[74]:


def predict(X):
    df = pd.DataFrame([X], columns=['hour', 'month', 'day', 'stationName'])
    X = create_feature(df)
    return model.predict(X)[0]


# In[75]:


X = create_feature(features)


# In[76]:


Y = df['health_implication'].values


# In[77]:


model = RandomForestClassifier(random_state=42)


# In[78]:


model.fit(X, Y)


# In[79]:


model.score(X, Y)


# In[80]:


# hour, month, day, station
predict([0, 11, 'Sunday', 'Ratna'])

