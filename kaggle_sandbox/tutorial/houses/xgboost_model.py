import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

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


def getImputedTrainTestData(input):
    my_imputer = SimpleImputer()
    return my_imputer.fit_transform(input[0]), my_imputer.transform(input[2])


def score_dataset_xgboost(X_train, X_test, y_train, y_test):
    model = XGBRegressor()
    model.fit(X_train, y_train, verbose=False)
    predictions = model.predict(X_test)

    return mean_absolute_error(predictions, y_test)


if __name__ == "__main__":
    # load data
    melbourne_data = pd.read_csv(FILE)
    melb_numeric_predictors = getMelbPredictors(melbourne_data)
    test_input = getTrainAndTestData(melb_numeric_predictors, melbourne_data.Price)
    missing_cols = getMissingCols(test_input[0])

    X_train, X_test = getImputedTrainTestData(test_input)
    print(score_dataset_xgboost(X_train, X_test, test_input[1], test_input[3]))
