import pandas as pd
FILE = './melb_data.csv'

if __name__ == "__main__":
    melbourne_data = pd.read_csv(FILE)
    print(melbourne_data.columns)

    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    X = melbourne_data[melbourne_features]

    y = melbourne_data.Price
    print(y)
    print(X.describe())



