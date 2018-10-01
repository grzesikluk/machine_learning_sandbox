import sys

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

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



    for max_leaf_nodes in [5, 50, 200, 500, 700, 5000]:
        my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))