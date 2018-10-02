import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

FILE = './melb_data.csv'


def getMissingCols(data):
    return [col for col in data.columns
            if data[col].isnull().any()]


def getMelbPredictors(data):
    melb_predictors = data.drop(['Price'], axis=1)
    return melb_predictors.select_dtypes(exclude=['object'])


def getTrainAndTestData(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        target,
                                                        train_size=0.7,
                                                        test_size=0.3,
                                                        random_state=0)
    return (X_train, y_train, X_test, y_test)


def getReducedData(input, missing_cols):
    return input.drop(missing_cols, axis=1)


def getReducedTrainTestData(input, missing_cols):
    return input[0].drop(missing_cols, axis=1), input[2].drop(missing_cols, axis=1)


def getImputedTrainTestData(input):
    my_imputer = SimpleImputer()
    return my_imputer.fit_transform(input[0]), my_imputer.transform(input[2])


def getImputedWithExtraColumnsTrainTestData(input, missing_cols):
    X_train = input[0].copy()
    X_test = input[2].copy()
    my_imputer = SimpleImputer()
    return my_imputer.fit_transform(input[0]), my_imputer.transform(input[2])

    for col in missing_cols:
        X_train[col + '_was_missing'] = X_train[col].isnull()
        X_test[col + '_was_missing'] = X_test[col].isnull()

    return my_imputer.fit_transform(X_train), my_imputer.transform(X_test)


def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


if __name__ == "__main__":
    # load data
    melbourne_data = pd.read_csv(FILE)
    melb_numeric_predictors = getMelbPredictors(melbourne_data)
    test_input = getTrainAndTestData(melb_numeric_predictors, melbourne_data.Price)
    missing_cols = getMissingCols(test_input[0])

    X_train, X_test = getReducedTrainTestData(test_input, missing_cols)
    print(score_dataset(X_train, X_test, test_input[1], test_input[3]))

    X_train, X_test = getImputedTrainTestData(test_input)
    print(score_dataset(X_train, X_test, test_input[1], test_input[3]))

    X_train, X_test = getImputedWithExtraColumnsTrainTestData(test_input, missing_cols)
    print(score_dataset(X_train, X_test, test_input[1], test_input[3]))
