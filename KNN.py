import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv('portland_housing.csv', encoding='utf-8')
df.dropna(subset = ["zestimate", "latitude", "longitude", "yearBuilt", "livingArea (sqft)", "lotSize", "bathrooms", "bedrooms", "daysOnZillow"], inplace=True)
def age(x):
    return float(2021-x)
df['age'] = df['yearBuilt'].apply(age)

scaler = MinMaxScaler()
df['latitude scl']=scaler.fit_transform(df[['latitude']])
df['longitude scl']=scaler.fit_transform(df[['longitude']])
df['house age scl']=scaler.fit_transform(df[['age']])
df['livingArea scl']=scaler.fit_transform(df[['livingArea (sqft)']])
df['lotSize scl']=scaler.fit_transform(df[['lotSize']])
df['bathrooms scl']=scaler.fit_transform(df[['bathrooms']])
df['bedrooms scl']=scaler.fit_transform(df[['bedrooms']])
df['daysListed scl']=scaler.fit_transform(df[['daysOnZillow']])

df_train, df_test = train_test_split(df, test_size=0.25, random_state=41)
X_train=df_train[['latitude scl', 'longitude scl', 'house age scl', 'livingArea scl', 'lotSize scl', 'bathrooms scl', 'bedrooms scl', 'daysListed scl']]
X_test=df_test[['latitude scl', 'longitude scl', 'house age scl', 'livingArea scl', 'lotSize scl', 'bathrooms scl', 'bedrooms scl', 'daysListed scl']]
Y_train=df_train['zestimate'].ravel()
Y_test=df_test['zestimate'].ravel()

model = KNeighborsRegressor(n_neighbors=3, weights='uniform', algorithm='auto', p=2, n_jobs=-1)

reg = model.fit(X_train, Y_train)
pred_values_train = model.predict(X_train)
pred_values_test = model.predict(X_test)

print(pd.DataFrame(Y_test))
print(pd.DataFrame(pred_values_test))
print(pd.DataFrame(Y_train))
print(pd.DataFrame(pred_values_train))

pd.set_option("display.width", 400)
pd.set_option("display.max_columns", 20)
print('---------------------------------------------------------')
print('Effective Metric: \t\t\t', reg.effective_metric_)
print('Effective Metric Params: \t', reg.effective_metric_params_)
print('No. of Samples Fit: \t\t', reg.n_samples_fit_)
print("")
print('Test r2 Squared Score: \t\t', sklearn.metrics.r2_score(Y_test, pred_values_test))
print("Test Mean Squared Error: \t", mean_squared_error(Y_test, pred_values_test))
print("")
print('Train r2 Squared Score: \t', sklearn.metrics.r2_score(Y_train, pred_values_train))
print("Train Mean Squared Error: \t", mean_squared_error(Y_train, pred_values_train))
print('---------------------------------------------------------')