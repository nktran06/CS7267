import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv('portland_housing.csv', encoding='utf-8')
df.dropna(subset = ["zestimate", "latitude", "longitude", "yearBuilt", "livingArea (sqft)", "lotSize", "bathrooms", "bedrooms", "daysOnZillow"], inplace=True)
def age(x):
    return float(2021-x)
df['age'] = df['yearBuilt'].apply(age)

scaler = preprocessing.MinMaxScaler()
df['latitude scl']=scaler.fit_transform(df[['latitude']])
df['longitude scl']=scaler.fit_transform(df[['longitude']])
df['house age scl']=scaler.fit_transform(df[['age']])
df['livingArea scl']=scaler.fit_transform(df[['livingArea (sqft)']])
df['lotSize scl']=scaler.fit_transform(df[['lotSize']])
df['bathrooms scl']=scaler.fit_transform(df[['bathrooms']])
df['bedrooms scl']=scaler.fit_transform(df[['bedrooms']])
df['daysListed scl']=scaler.fit_transform(df[['daysOnZillow']])
df['soldPrice scl']=scaler.fit_transform(df[['lastSoldPrice']])

df_train, df_test = sklearn.model_selection.train_test_split(df, test_size=0.20, random_state=41)
X_train=df_train[['latitude scl', 'longitude scl', 'house age scl', 'livingArea scl', 'lotSize scl', 'bathrooms scl', 'bedrooms scl', 'daysListed scl', 'soldPrice scl']]
X_test=df_test[['latitude scl', 'longitude scl', 'house age scl', 'livingArea scl', 'lotSize scl', 'bathrooms scl', 'bedrooms scl', 'daysListed scl', 'soldPrice scl']]
Y_train=df_train['zestimate'].ravel()
Y_test=df_test['zestimate'].ravel()

model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=9, weights='distance', algorithm='auto', p=2, n_jobs=-1)

reg = model.fit(X_train, Y_train)
pred_values_train = model.predict(X_train)
pred_values_train = pred_values_train.astype(int)
pred_values_test = model.predict(X_test)
pred_values_test = pred_values_test.astype(int)

pd.set_option("display.width", 400)
pd.set_option("display.max_columns", 20)

print(pd.DataFrame(Y_test))
print(pd.DataFrame(pred_values_test))
print(pd.DataFrame(Y_train))
print(pd.DataFrame(pred_values_train))
print("")
print('---------------------------------------------------------')
print('TEST')
print('r2 Squared Score: \t\t\t', sklearn.metrics.r2_score(Y_test, pred_values_test))
print('Root Mean Squared Error: \t', sklearn.metrics.mean_squared_error(Y_test, pred_values_test, squared=0))
print("Mean Squared Error: \t\t", sklearn.metrics.mean_squared_error(Y_test, pred_values_test))
print("Mean Absolute Error: \t\t", sklearn.metrics.mean_absolute_error(Y_test, pred_values_test))
print("")
print('TRAIN')
print('r2 Squared Score: \t\t\t', sklearn.metrics.r2_score(Y_train, pred_values_train))
print('Root Mean Squared Error: \t', sklearn.metrics.mean_squared_error(Y_train, pred_values_train, squared=0))
print("Mean Squared Error: \t\t", sklearn.metrics.mean_squared_error(Y_train, pred_values_train))
print("Mean Absolute Error: \t\t", sklearn.metrics.mean_absolute_error(Y_train, pred_values_train))
print('---------------------------------------------------------')