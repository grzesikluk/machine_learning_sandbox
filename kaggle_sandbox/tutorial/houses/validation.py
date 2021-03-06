import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

FILE = './melb_data.csv'

if __name__ == "__main__":
    # load data
    melbourne_data = pd.read_csv(FILE)

    # remove empty data
    melbourne_data = melbourne_data.dropna(axis=0)

    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    # X is features
    X = melbourne_data[melbourne_features]
    # y is prediction target
    y = melbourne_data.Price

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=2)

    # creating model
    melbourne_model = DecisionTreeRegressor(random_state=1)
    melbourne_model.fit(train_X, train_y)

    predicted_prices = melbourne_model.predict(val_X)
    print(mean_absolute_error(val_y, predicted_prices))
