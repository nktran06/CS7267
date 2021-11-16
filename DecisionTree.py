import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing


def importdata():
    data = pd.read_csv('portland_housing_full.csv')
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

    # smaller data set
    # data = data.drop(columns=['priceHistory/0/price', 'priceHistory/2/price', 'lastSoldPrice'])
    # data = data.reindex(columns=['zpid', 'bedrooms', 'bathrooms', 'livingArea (sqft)', 'lotSize', 'daysOnZillow', 'yearBuilt', 'zestimate' ])

    # data preprocessing with normalization
    # d = preprocessing.normalize(data, axis=0)  # normalize along columns
    # data = pd.DataFrame(d, columns=['zpid', 'bedrooms', 'bathrooms', 'livingArea (sqft)',
    #                              'lotSize', 'daysOnZillow', 'yearBuilt', 'zestimate'])

    data = data.drop(columns=['abbreviatedAddress', 'address/city'])
    data = data.reindex(columns=['zpid', 'bedrooms', 'bathrooms', 'daysOnZillow', 'lastSoldPrice', 'livingArea (sqft)',
                                 'latitude', 'longitude', 'lotSize', 'price', 'priceHistory/0/price',
                                 'priceHistory/0/priceChangeRate', 'priceHistory/0/pricePerSquareFoot',
                                 'priceHistory/1/pricePerSquareFoot', 'priceHistory/2/price',
                                 'priceHistory/2/priceChangeRate', 'priceHistory/2/pricePerSquareFoot',
                                 'priceHistory/3/price', 'priceHistory/3/priceChangeRate',
                                 'priceHistory/3/pricePerSquareFoot',
                                 'yearBuilt', 'zestimate'])

    # dropping rows with null values
    # data = dataDrop(data)

    # filling values with via linear interpolation
    data = dataInterpolate(data)

    # data preprocessing with normalization
    d = preprocessing.normalize(data, axis=0)  # normalize along columns
    data = pd.DataFrame(d,
                        columns=['zpid', 'bedrooms', 'bathrooms', 'daysOnZillow', 'lastSoldPrice', 'livingArea (sqft)',
                                 'latitude', 'longitude', 'lotSize', 'price', 'priceHistory/0/price',
                                 'priceHistory/0/priceChangeRate', 'priceHistory/0/pricePerSquareFoot',
                                 'priceHistory/1/pricePerSquareFoot', 'priceHistory/2/price',
                                 'priceHistory/2/priceChangeRate', 'priceHistory/2/pricePerSquareFoot',
                                 'priceHistory/3/price', 'priceHistory/3/priceChangeRate',
                                 'priceHistory/3/pricePerSquareFoot',
                                 'yearBuilt', 'zestimate'])

    zpid = data.iloc[:, 0:1].values

    X = data.iloc[:, 1:-1].values
    Y = data.iloc[:, -1].values

    X_train, X_test, Y_train, Y_test = splitData(X, Y)

    print("Number of Training Points:", len(X_train))
    print("Number of Test Points: ", len(X_test))

    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, Y_train)

    y_predict = regressor.predict(X_test)

    # print("Accuracy: " + str(accuracy_score(Y_test, y_predict)))

    r2score = r2_score(Y_test, y_predict)
    print('R2_Score: ', r2score)

    df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_predict, 'Difference': np.abs(Y_test - y_predict),
                       'MSE': np.sqrt(np.abs(Y_test - y_predict))})
    pd.set_option('display.max_rows', None)
    outFile = open('result_interpolate.json', 'w')
    # outFile = open('result_drop.json', 'w')
    outFile.write(str(df))
    outFile.close()

    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_predict))
    print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_predict))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_predict)))


if __name__ == "__main__":
    main()