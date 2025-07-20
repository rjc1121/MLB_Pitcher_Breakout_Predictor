# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

import sklearn
import xgboost as xgb
import matplotlib





#Reading excel files 
Pitching_2023 = pd.read_excel('Pitching_2023.xlsx', skiprows=1)
Pitching_2024 = pd.read_excel('Pitching_2024.xlsx', skiprows=1)

print(Pitching_2023.head())

print(Pitching_2024.head())

# For 2023
Pitching_2023[['Last', 'First']] = Pitching_2023['last_name, first_name'].str.split(', ', expand=True)
Pitching_2023['Name'] = Pitching_2023['First'] + ' ' + Pitching_2023['Last']
Pitching_2023 = Pitching_2023.drop(columns=['First', 'Last'])

# For 2024
Pitching_2024[['Last', 'First']] = Pitching_2024['last_name, first_name'].str.split(', ', expand=True)
Pitching_2024['Name'] = Pitching_2024['First'] + ' ' + Pitching_2024['Last']
Pitching_2024 = Pitching_2024.drop(columns=['First', 'Last'])

print(Pitching_2023.head())

print(Pitching_2024.head())

merged_Pitching = pd.merge(Pitching_2023, Pitching_2024, on="Name", suffixes=("_2023", "_2024"))

pitchers_filtered = merged_Pitching[
    (merged_Pitching['player_age_2024'] <= 28) & 
    (merged_Pitching['pa_2023'] >= 150) & 
    (merged_Pitching['pa_2023'] <= 500)
]

X = merged_Pitching[['xslgdiff_2023', 'xbadiff_2023']]
y = merged_Pitching['p_era_2024']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=21) 
model.fit(X_train, y_train)