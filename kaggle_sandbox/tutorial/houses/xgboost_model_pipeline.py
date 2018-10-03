import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor

FILE = './melb_data.csv'


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


if __name__ == "__main__":
    melbourne_data = pd.read_csv(FILE)
    melb_numeric_predictors = getMelbPredictors(melbourne_data)
    test_input = getTrainAndTestData(melb_numeric_predictors, melbourne_data.Price)

    pipeline = make_pipeline(Imputer(), RandomForestRegressor())
    pipeline.fit(test_input[0],test_input[1])
    predictions = pipeline.predict(test_input[2])
    print(mean_absolute_error(predictions, test_input[3]))


