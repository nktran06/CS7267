import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

def importdata():
    data = pd.read_csv('portland_housing.csv')
    return data

def dataDrop(data):
    return data.dropna()

def dataInterpolate(data):
    return data.interpolate(method='linear', limit_direction='forward')

def splitData(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)
    return X_train, X_test, Y_train, Y_test

def main():
    data = importdata()

    data = data.drop(columns=['priceHistory/0/price', 'priceHistory/2/price', 'lastSoldPrice'])
    data = data.reindex(columns=['zpid', 'bedrooms', 'bathrooms', 'livingArea (sqft)', 'lotSize', 'daysOnZillow', 'yearBuilt', 'zestimate' ])

    #dropping rows with null values
    data = dataDrop(data)

    #filling values with via linear interpolation
    #data = dataInterpolate(data)

    X = data.iloc[:, 1:-1].values
    Y = data.iloc[:, -1].values

    X_train, X_test, Y_train, Y_test = splitData(X, Y)

    print(len(X_train))
    print(len(X_test))

    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, Y_train)

    y_predict = regressor.predict(X_test)

    print("Accuracy: " + str(accuracy_score(Y_test, y_predict)))

    df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_predict})
    print((df))

    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_predict))
    print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_predict))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_predict)))

    from sklearn.tree import export_graphviz
    export_graphviz(regressor, out_file='tree.dot', feature_names=['bedrooms', 'bathrooms', 'livingArea (sqft)', 'lotSize', 'daysOnZillow', 'yearBuilt',])

if __name__=="__main__":
    main()