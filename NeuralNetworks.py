import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Dense
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
import hvplot.pandas


def main():
    cols = ["bathroom", "bedroom", "days_on_zillow", "last_sold_price",
            "latitude", "living_area", "longitude", "lot_size", "year_built",
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
        ["bathroom", "bedroom", "days_on_zillow", "last_sold_price",
         "living_area", "lot_size", "year_built", "zestimate"]]


    X = X.values
    print(type(X))
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled)
    Y = X.iloc[:, -1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2,
                                                        random_state=30)

    # turns the dfs to Numpy Arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    neural_network(X_train, X_test, Y_train, Y_test)


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
                  batch_size=10,
                  epochs=30)

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
