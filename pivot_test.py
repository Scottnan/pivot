import pandas as pd
import time
import numpy as np
from pivot_cython import pivot

data = pd.read_csv("data.csv")[["TRADE_DT",  "S_INFO_WINDCODE",  "S_DQ_CLOSE"]]
print(data)
data_cython= data.values.astype("float32")
data_python = data.astype("float")
print(data_cython[:, 0])
print(np.unique(data_cython[:, 0]))

time_start = time.time()
pivot(data_cython, data_cython[:, 0], data_cython[:, 1])
time_end = time.time()
print('cython cost', time_end-time_start)

time_start = time.time()
data.pivot(index='TRADE_DT', columns='S_INFO_WINDCODE', values='S_DQ_CLOSE')
time_end = time.time()
print('pandas cost', time_end-time_start)

