import pandas as pd
import numpy as np


if __name__ == "__main__":
    # save filepath to variable for easier access
    melbourne_file_path = 'melb_data.csv'
    # read the data and store data in DataFrame titled melbourne_data
    melbourne_data = pd.read_csv(melbourne_file_path)
    # print a summary of the data in Melbourne data
    # print(melbourne_data)
    # print(melbourne_data.describe())
    # print(melbourne_data.T)
    # melbourne_data.to_hdf('melb_data.h5','melbourne_data')
    print(melbourne_data.mean(1))