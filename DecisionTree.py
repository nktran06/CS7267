import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
import matplotlib.pyplot as plt

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
    #Uncomment for graph
    # data = data.drop(columns=['abbreviatedAddress', 'address/city', 'latitude', 'longitude', 'price', 'priceHistory/0/price',
    #                              'priceHistory/0/priceChangeRate', 'priceHistory/0/pricePerSquareFoot',
    #                              'priceHistory/1/pricePerSquareFoot', 'priceHistory/2/price',
    #                              'priceHistory/2/priceChangeRate', 'priceHistory/2/pricePerSquareFoot',
    #                              'priceHistory/3/price', 'priceHistory/3/priceChangeRate', 'priceHistory/3/pricePerSquareFoot' ,'lastSoldPrice', 'livingArea (sqft)',
    #                              'yearBuilt', 'zpid'])
    #
    # data = data.reindex(columns=['bedrooms', 'bathrooms', 'daysOnZillow', 'lotSize', 'zestimate'])

    data = data.drop(
        columns=['abbreviatedAddress', 'address/city', 'latitude', 'longitude', 'price', 'priceHistory/0/price',
                 'priceHistory/0/priceChangeRate', 'priceHistory/0/pricePerSquareFoot',
                 'priceHistory/1/pricePerSquareFoot', 'priceHistory/2/price',
                 'priceHistory/2/priceChangeRate', 'priceHistory/2/pricePerSquareFoot',
                 'priceHistory/3/price', 'priceHistory/3/priceChangeRate', 'priceHistory/3/pricePerSquareFoot',
                  'zpid'])

    data = data.reindex(columns=['bedrooms', 'bathrooms', 'lotSize', 'livingArea (sqft)',
                                 'yearBuilt','daysOnZillow', 'lastSoldPrice', 'zestimate'])

    #dropping rows with null values
    #data = dataDrop(data)

    #filling values with via linear interpolation
    data = dataInterpolate(data)

    #data preprocessing with normalization
    d = preprocessing.normalize(data, axis=0) #normalize along columns
    data = pd.DataFrame(d, columns=['bedrooms', 'bathrooms', 'lotSize', 'livingArea (sqft)',
                                 'yearBuilt','daysOnZillow', 'lastSoldPrice', 'zestimate'])


    zpid = data.iloc[0:50, 0:1].values
    X = data.iloc[:, 0:-1].values
    Y = data.iloc[:, -1].values

    X_train, X_test, Y_train, Y_test = splitData(X, Y)

    print("Number of Training Points:", len(X_train))
    print("Number of Test Points: ", len(X_test))

    regressor_1 = DecisionTreeRegressor(random_state=1, max_depth=2)
    regressor_1.fit(X_train, Y_train)
    y_predict_1 = regressor_1.predict(X_test)

    regressor_2 = DecisionTreeRegressor(random_state=1, max_depth=6)
    regressor_2.fit(X_train, Y_train)
    y_predict_2 = regressor_2.predict(X_test)


    print('\nDecsion Tree 1')
    r2score = r2_score(Y_test, y_predict_1)
    print('R2_Score: ', r2score)
    df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_predict_1,
                       'Difference': np.abs(Y_test-y_predict_1),
                       'MSE': np.sqrt(np.abs(Y_test-y_predict_1))})
    pd.set_option('display.max_rows', None)
    outFile = open('result_interpolate.json', 'w')
    #outFile = open('result_drop.json', 'w')
    outFile.write(str(df))
    outFile.close()

    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_predict_1))
    print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_predict_1))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_predict_1)))


    # regression 2
    print('\nDecsion Tree 2')
    r2score = r2_score(Y_test, y_predict_2)
    print('R2_Score: ', r2score)
    df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_predict_2,
                       'Difference': np.abs(Y_test - y_predict_2),
                       'MSE': np.sqrt(np.abs(Y_test - y_predict_2))})
    pd.set_option('display.max_rows', None)

    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_predict_2))
    print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_predict_2))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_predict_2)))

    #uncomment for decision tree graph
    # from sklearn import tree
    #
    # featNames = ['bedrooms', 'bathrooms', 'daysOnZillow', 'lotSize','zestimate']
    # fig1 = plt.figure(figsize=(9, 10))
    # tree.plot_tree(regressor_1.fit(X_train, Y_train), feature_names=featNames,
    #                class_names=Y_test,
    #                filled=True)
    # fig1.savefig('tree1.png', bbox_inches='tight')
    #
    # fig2 = plt.figure(figsize=(9, 10))
    # tree.plot_tree(regressor_2.fit(X_train, Y_train), feature_names=featNames,
    #                class_names=Y_test,
    #                filled=True)
    # fig2.savefig('tree2.png', bbox_inches='tight')

if __name__=="__main__":
    main()