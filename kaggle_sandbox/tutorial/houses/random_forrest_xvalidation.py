import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer

FILE = './melb_data.csv'

if __name__ == "__main__":
    # load data
    melbourne_data = pd.read_csv(FILE)

    # remove empty data
    melbourne_data = melbourne_data.dropna(axis=0)

    melbourne_features = ['Rooms', 'YearBuilt', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    X = melbourne_data[melbourne_features]
    y = melbourne_data.Price

    pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor())
    scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error')
    # print(scores)
    print('Mean Absolute Error %2f' %(-1 * scores.mean()))

