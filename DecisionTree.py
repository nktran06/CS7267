import numpy as np
import pandas as pd
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

    #dropping rows with null values
    #data = dataDrop(data)

    #filling values with via linear interpolation
    data = dataInterpolate(data)

    X = data.iloc[:, 1:10].values
    Y = data.iloc[:, 10].values

    X_train, X_test, Y_train, Y_test = splitData(X, Y)

    print(len(X_train))
    print(len(X_test))

    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, Y_train)


if __name__=="__main__":
    main()