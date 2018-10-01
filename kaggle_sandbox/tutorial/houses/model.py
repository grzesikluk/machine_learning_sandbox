import pandas as pd
from sklearn.tree import DecisionTreeRegressor

FILE = './melb_data.csv'

if __name__ == "__main__":

    #load data
    melbourne_data = pd.read_csv(FILE)

    #remove empty data
    melbourne_data = melbourne_data.dropna(axis=0)

    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    # X is features
    X = melbourne_data[melbourne_features]
    # y is prediction target
    y = melbourne_data.Price

    #creating model
    melbourne_model = DecisionTreeRegressor(random_state=1)
    print(melbourne_model.fit(X, y))

    # make predictions
    print("Making predictions for the following 5 houses:")
    print(X.head())
    print("The predictions are")
    print(melbourne_model.predict(X.head()))
