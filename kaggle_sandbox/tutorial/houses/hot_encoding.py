import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

FILE = './melb_data.csv'


def getTrainAndTestData(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        target,
                                                        train_size=0.7,
                                                        test_size=0.3,
                                                        random_state=0)
    return (X_train, y_train, X_test, y_test)


def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(50)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


def getOneHotEncoderTrainTestData(data):
    return pd.get_dummies(data)


if __name__ == "__main__":
    melbourne_data = pd.read_csv(FILE)
    melbourne_data_hot_encoded = getOneHotEncoderTrainTestData(melbourne_data)
    test_input = getTrainAndTestData(melbourne_data_hot_encoded, melbourne_data_hot_encoded.Price)
    print(score_dataset(test_input[0], test_input[2], test_input[1], test_input[3]))
