# distutils: language = c++
cimport numpy as np
cimport cython
import time
import numpy as np
from cython.parallel import prange
from libcpp.pair cimport pair
from libcpp.map cimport map

DTYPE = np.float32


cdef void build_doublemap(map[pair[float, float], float] m, float date, float stock, float data) nogil:
    cdef pair[float, float] p1
    p1.first = date
    p1.second = stock
    cdef pair[pair[float, float], float] p2
    p2.first = p1
    p2.second = data
    m.insert(p2)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _pivot(float[:, :] data, float[:] dates, float[:] stocks):
    cdef Py_ssize_t rows, cols
    cdef Py_ssize_t orgrows = data.shape[0]
    cdef Py_ssize_t orgcols = data.shape[1]
    cdef float[:] unique_date = np.unique(dates)
    cdef float[:] unique_stock = np.unique(stocks)

    rows, cols = unique_date.shape[0], unique_stock.shape[0]
    result = np.zeros((rows, cols), dtype=np.float32)
    cdef float[:, ::1] result_view = result
    cdef map[pair[float, float], float] m

    time_start = time.time()
    cdef Py_ssize_t i
    for i in prange(orgrows, nogil=True):
        build_doublemap(m, data[i, 0], data[i, 1], data[i, 2])
    time_end = time.time()
    print('part1', time_end-time_start)
    time_start = time.time()
    for i in range(rows):
        for j in range(cols):
            p1 = (unique_date[i], unique_stock[j])
            it = m.find(p1)
            if it == m.end():
                result_view[i, j] = np.nan
            else:
                result_view[i, j] = m.at(p1)
    time_end = time.time()
    print('part2', time_end-time_start)
    return result

def pivot(data, dates, stocks):
    return _pivot(data, dates, stocks)