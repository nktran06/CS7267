import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv('portland_housing.csv', encoding='utf-8')
df.dropna(subset = ["zestimate", "latitude", "longitude", "yearBuilt", "livingArea (sqft)", "lotSize", "bathrooms", "bedrooms"], inplace=True)
df['Price Bracket'] = pd.qcut(df['zestimate'], 3, labels=['bottom 33', 'middle 33', 'top 33'])
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
enc=OrdinalEncoder()
df['Price Bracket enc']=enc.fit_transform(df[['Price Bracket']]) # encode categorical values

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
X_train=df_train[['latitude scl', 'longitude scl', 'house age scl', 'livingArea scl', 'lotSize scl', 'bathrooms scl', 'bedrooms scl']]
X_test=df_test[['latitude scl', 'longitude scl', 'house age scl', 'livingArea scl', 'lotSize scl', 'bathrooms scl', 'bedrooms scl']]
y_train=df_train['zestimate'].ravel()
y_test=df_test['zestimate'].ravel()

model = KNeighborsRegressor(n_neighbors=3, weights='uniform', algorithm='auto', p=2, n_jobs=-1)

reg = model.fit(X_train, y_train)
pred_values_train = model.predict(X_train)
pred_values_test = model.predict(X_test)

print(df)
print('---------------------------------------------------------')
print('Number of Samples Fit: ', reg.n_samples_fit_)
score_test = model.score(X_test, y_test)
print('Test Accuracy Score: ', score_test)
score_train = model.score(X_train, y_train)
print('Training Accuracy Score: ', score_train)
print('---------------------------------------------------------')