import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Dense
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import seaborn as sb
from scipy import stats


def main():
    cols = ["bathroom", "bedroom", "days on zillow", "last sold price",
            "latitude", "living area", "longitude", "lot size", "year built",
            "zestimate"]
    Portland_housing = pd.read_csv(
        r'C:\Users\Matthew Zinkil\Documents\portland_housing_clean.csv',
        header=None, names=cols, skiprows=1)

    # drops the rows with zero values
    nan_value = float("NaN")
    Portland_housing.replace("", nan_value, inplace=True)
    Portland_housing.dropna(subset=cols, inplace=True)

    # randomize the dataframe and then use a sample for the testing, training
    Portland_housing.sample(frac=1)
    X = Portland_housing[
        ["bathroom", "bedroom", "days on zillow", "last sold price",
         "living area", "lot size", "year built", "zestimate"]]

    X[(np.abs(stats.zscore(X)) < 3).all(axis=1)]
    #visualize_data(X, "zestimate")



    X = X.values
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled)
    Y = X.iloc[:, -1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3,
                                                        random_state=30)

    # turns the dfs to Numpy Arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    LR(X_train, Y_train, X_test, Y_test)
    neural_network(X_train, X_test, Y_train, Y_test)


def visualize_data(X, y_var):
    scatter_df = X.drop(y_var, axis=1)
    i = X.columns

    plot1 = sb.scatterplot(i[0], y_var, data=X, color='orange', edgecolor='b',
                           s=150)
    plt.title('{} / Zestimate'.format(i[0]), fontsize=16)
    plt.xlabel('{}'.format(i[0]), fontsize=14)
    plt.ylabel('Zestimate', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter1.png')
    plt.show()

    plot2 = sb.scatterplot(i[1], y_var, data=X, color='yellow', edgecolor='b',
                           s=150)
    plt.title('{} / Zestimate'.format(i[1]), fontsize=16)
    plt.xlabel('{}'.format(i[1]), fontsize=14)
    plt.ylabel('Zestimate', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter2.png')
    plt.show()

    plot3 = sb.scatterplot(i[2], y_var, data=X, color='aquamarine',
                           edgecolor='b',
                           s=150)
    plt.title('{} / Zestimate'.format(i[2]), fontsize=16)
    plt.xlabel('{}'.format(i[2]), fontsize=14)
    plt.ylabel('Zestimate', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter3.png')
    plt.show()

    plot4 = sb.scatterplot(i[3],
                           y_var, data=X, color='deepskyblue', edgecolor='b',
                           s=150)
    plt.title('{} / Zestimate'.format(i[3]), fontsize=16)
    plt.xlabel('{}'.format(i[3]), fontsize=14)
    plt.ylabel('Zestimate', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter4.png')
    plt.show()

    plot5 = sb.scatterplot(i[4], y_var, data=X, color='crimson',
                           edgecolor='white',
                           s=150)
    plt.title('{} / Zestimate'.format(i[4]), fontsize=16)
    plt.xlabel('{}'.format(i[4]), fontsize=14)
    plt.ylabel('Zestimate', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter5.png')
    plt.show()

    plot6 = sb.scatterplot(i[5],
                           y_var, data=X, color='darkviolet', edgecolor='white',
                           s=150)
    plt.title('{} / Zestimate'.format(i[5]), fontsize=16)
    plt.xlabel('{}'.format(i[5]), fontsize=14)
    plt.ylabel('Zestimate', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter6.png')
    plt.show()

    plot7 = sb.scatterplot(i[6], y_var, data=X, color='khaki', edgecolor='b',
                           s=150)
    plt.title('{} / Zestimate'.format(i[6]), fontsize=16)
    plt.xlabel('{}'.format(i[6]), fontsize=14)
    plt.ylabel('Zestimate', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('scatter7.png')
    plt.show()

    sb.distplot(X['zestimate'], color='r')
    plt.title('Zestimate Distribution', fontsize=16)
    plt.xlabel('Zestimate', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks([200000, 400000, 600000, 800000, 1000000, 1200000], fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig('distplot.png')
    plt.show()

def LR(X_train, Y_train, X_test, Y_test):
    train_rows, train_cols = X_train.shape
    test_rows, test_cols = X_test.shape
    test_points = train_rows + test_rows
    LR = LinearRegression()
    LR.fit(X_train, Y_train)
    LR_predict = LR.predict(X_test)
    print("Linear Regression Results")
    print("Number of Test Points: ", test_points)
    print_evaluate(Y_test, LR_predict)


def neural_network(X_train, X_test, Y_train, Y_test):
    model = Sequential()
    model.add(
        Dense(X_train.shape[1], input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer=Adam(), loss='mae')

    r = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                  batch_size=100,
                  epochs=70)

    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)

    print("Test set evaluation: ")
    print_evaluate(Y_test, test_pred)

    print("Train Set Evaluation: ")
    print_evaluate(Y_train, train_pred)

    plt.plot(r.history['loss'])
    plt.plot(r.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['Y_test', 'train_pred'])
    plt.show()


def print_evaluate(true, predicted):
    MAE = metrics.mean_absolute_error(true, predicted)
    MSE = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_squared = metrics.r2_score(true, predicted)

    print("MAE: ", MAE)
    print("MSE: ", MSE)
    print("RMSE: ", rmse)
    print("R2 Square: ", r2_squared)


def print_loss(r):
    plot = pd.DataFrame(r.history).hvplot.line(y=['loss', 'val_loss'])
    print(type(plot))


if __name__ == "__main__":
    main()
