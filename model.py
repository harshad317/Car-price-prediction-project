import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

df= pd.read_csv('car data.csv')

def preprocessing(dataset):
    dataset['Selling_Price']= np.log1p(dataset['Selling_Price'])
    dataset['Present_Price']= np.log1p(dataset['Present_Price'])
    dataset['Kms_Driven']= np.log1p(dataset['Kms_Driven'])
    return dataset

def feature_engg(dataset):
    dataset['kms_driven_by_car_name_and_year']= dataset.groupby(['Year', 'Car_Name'])['Kms_Driven'].transform('mean')
    dataset['present_price_by_car_name_and_year']= dataset.groupby(['Year', 'Car_Name'])['Present_Price'].transform('mean')
    dataset['present_price_by_transmission_car_name']= dataset.groupby(['Transmission', 'Car_Name'])['Present_Price'].transform('mean')
    dataset['present_price_by_fuel_type_car_name']= dataset.groupby(['Fuel_Type', 'Car_Name'])['Present_Price'].transform('mean')
    dataset['present_price_by_seller_type_car_name']= dataset.groupby(['Seller_Type', 'Car_Name'])['Present_Price'].transform('mean')

    df.Transmission= df.Transmission.map({'Manual': 0, 'Automatic': 1})
    df.Seller_Type= df.Seller_Type.map({'Dealer':0, 'Individual': 1})
    df.Fuel_Type= df.Fuel_Type.map({'CNG':0, 'Diesel':1, 'Petrol':2})
    return dataset

df= preprocessing(dataset= df)
df= feature_engg(dataset= df)
cont_cols= [var for var in df.columns if df[var].dtypes != 'O']

X= df.drop(['Car_Name', 'Selling_Price'], axis= 1)
y= df['Selling_Price']

def my_model():
    model = CatBoostRegressor(eval_metric='RMSE', n_estimators=2500, verbose=0)
    model.fit(X, y)
    return model

regressor= my_model()

pickle.dump(regressor, open('regressor.pkl', 'wb'))