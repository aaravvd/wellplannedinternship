import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle
from datetime import datetime


import argparse
import pandas as pd

# Define the argument parser
parser = argparse.ArgumentParser(description="Flight Fare Prediction")  # argparse for flexible command line arguments to update dataset
parser.add_argument('--train_file', type=str, help="Input file path", default = "/Users/aaravdixit/flightpred/Data/data.csv")  # Define the argument for the input file path

args = parser.parse_args()

df = pd.read_csv(args.train_file)







df.sort_values(by=['city1', 'city2','Year', 'quarter',], inplace=True)




df.dropna(inplace = True)




df['city_pair'] = df['city1']+df['city2'] #creating city pair feature
real_df = df.sort_values(by=['city_pair', 'Year'])

df['previous_fare'] = real_df.groupby('city_pair')['fare'].shift(1)




df['city_pair'] = df.apply(lambda row: '-'.join(sorted([row['city1'], row['city2']])), axis=1)

sorted_df = df.sort_values(by=['city_pair', 'Year']) #sorting by the citypair and year

sorted_df['fare_difference'] = sorted_df.groupby('city_pair')['fare'].diff() #finding the difference in fare between quarters
df = sorted_df[['city_pair', 'Year', 'fare_difference', 'quarter', 'passengers', 'nsmiles', 'previous_fare', 'fare', 'fare_low']]
df.dropna(inplace = True)






numerical_columns = ['Year', 'quarter', 'nsmiles', 'passengers', 'fare', 'previous_fare', 'fare_difference'] 
numerical_df = df[numerical_columns]






numerical_df = numerical_df.dropna(subset=['previous_fare'])




features = ['previous_fare', 'nsmiles', 'passengers', 'quarter']
target = 'fare_difference'

X = numerical_df[features]
y = numerical_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #create testing and training data





model = RandomForestRegressor(random_state=42) #define rf model

param_grid = { #did gridsearch once, then hardcoded the parameters
    'n_estimators': [100],            
    'max_depth': [10],            
    'min_samples_split': [2],       
    'min_samples_leaf': [2],               
    'max_features': ['sqrt'],     
}


grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)


grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
timestamp = timestamp = int(round(datetime.now().timestamp())) #best model
output_file_name = f'model_{timestamp}.pkl' #timestamp from datetime
#to save new model everytime

with open(f'../Model/{output_file_name}', 'wb') as f: #save model
    pickle.dump(best_model, f)
