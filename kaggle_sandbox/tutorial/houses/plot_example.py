import sys
import pandas as pd
import matplotlib as plt
import numpy as np

if __name__ == "__main__":
    ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
    ts = ts.cumsum()
    ts.plot()
    # df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,columns=['A', 'B', 'C', 'D'])
    # df = df.cumsum()
    # plt.figure(); df.plot(); plt.legend(loc='best')