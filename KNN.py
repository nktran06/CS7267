import numpy as np
import pandas as pd
from scipy.stats import mode


def main():
    Continue = 0
    x_train, y_train, x_test, y_test = prepare_dataset()
    while Continue == 0:
        k = int(input("Enter k-value: "))

        KNNInstance = KNN(x_train, y_train, x_test, y_test, k)
        y_predict = KNNInstance.classify_test_data()
        KNNInstance.test_accuracy(y_predict, y_test)

        answer = input("Would you like to test different K values? ")
        if answer.upper() == 'NO':
            Continue = 1


class KNN:
    def __init__(self, x_train, y_train, x_test, y_test, k):
        self.k = k
        self.instances = int(len(x_train) + len(x_test))
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def distance_calculation(self, p1, p2):
        distance = np.sqrt(np.sum((p1 - p2) ** 2))
        return distance

    def classify_test_data(self):
        created_labels = []
        for test_point in self.x_test:
            dist_array = []


            for train_point in self.x_train:
                distance = self.distance_calculation(test_point, train_point)
                dist_array.append(distance)

            dist_array = np.array(dist_array)
            distances = np.argsort(dist_array)[:self.k]

            y = self.y_train[distances]

            lab = mode(y)
            lab = lab[0]
            created_labels.append(lab)
        return created_labels

    def test_accuracy(self, y_predict, y_test):
        true_positives = 0
        true_negatives = 0
        false_positives =0
        false_negatives = 0

        for i in range(len(y_test)):

            if y_predict[i] == y_test[i]:
                if y_predict[i] == 1 and y_test[i] == 1:
                    true_positives += 1
                elif y_predict[i] == -1 and y_test[i] == -1:
                    true_negatives += 1
            elif y_predict[i] != y_test[i]:
                if y_predict[i] == 1 and y_test[i] == -1:
                    false_positives += 1
                elif y_predict[i] == -1 and y_test[i] == 1:
                    false_negatives += 1
        print("Total samples: ", len(y_test))
        print("This is the confusion matrix: \nTrue Positives: ",
              true_positives, "\nTrue Negatives: ", true_negatives,
              "\nFalse Positives: " , false_positives, "\nFalse Negatives: ",
              false_negatives, "\n Accuracy: ",
              float((true_negatives+true_positives)/len(y_test)))


def prepare_dataset():
    # loads in data set randomly assigns 70 percent of the data set to the
    # training group and assigns the other 30% of the file to the test group
    X = pd.read_csv('wdbc.data.mb.csv')
    LengthOfList = len(X)
    # First randomize the dataframe
    X = X.sample(frac = 1)
    # seperate the class column
    classvalue = X.iloc[:, -1:]
    X = X.iloc[:, :-2]
    X = (X - X.mean()) / X.std()
    X = X.to_numpy()
    classvalue = classvalue.to_numpy()
    LengthOfList = len(X)
    training_size = int(LengthOfList * .7)
    x_train = X[:training_size]
    y_train = classvalue[:training_size]
    x_test = X[training_size:]
    y_test = classvalue[training_size:]
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    main()
