# distutils: language = c++
cimport numpy as np
cimport cython
import time
import numpy as np
from cython.parallel import prange
from libcpp.pair cimport pair
from libcpp.map cimport map
import numba


cdef void build_doublemap(map[pair[float, float], float] m, np.ndarray[np.float32_t, ndim=2] data) nogil:
    cdef int orgrows
    orgrows = data.shape[0]
    for i in prange(orgrows):
        p1 = (data[i, 0], data[i, 1])
        p2 = (p1, data[i, 2])
        m.insert(p2)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=2] _pivot(np.ndarray[np.float32_t, ndim=2] data,
                                             np.ndarray[np.float32_t, ndim=1] dates,
                                             np.ndarray[np.float32_t, ndim=1] stocks):
    cdef int rows, cols, orgrows, orgcols
    cdef float n, date, stock, data_
    cdef np.ndarray[np.float32_t, ndim=1] unique_date, unique_stock
    cdef np.ndarray[np.float32_t, ndim=2] result
    orgrows, orgcols = data.shape[0], data.shape[1]
    unique_date, unique_stock = np.unique(dates), np.unique(stocks)
    rows, cols = unique_date.shape[0], unique_stock.shape[0]
    result = np.zeros((rows, cols), dtype=np.float32)
    cdef pair[float, float] p1
    cdef pair[pair[float, float], float] p2
    cdef map[pair[float, float], float] m
    cdef map[pair[float, float], float].iterator it

    build_doublemap(m, data)



    start_time = time.time()
    for i in range(rows):
        for j in range(cols):
            p1 = (unique_date[i], unique_stock[j])
            it = m.find(p1)
            if it == m.end():
               result[i, j] = np.nan
            else:
              result[i, j] = m.at(p1)
    end_time = time.time()
    print('part2:', end_time-start_time)
    return result

def pivot(data, dates, stocks):
    return _pivot(data, dates, stocks)
