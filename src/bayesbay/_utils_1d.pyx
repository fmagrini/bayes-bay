# cython: embedsignature=True

cimport cython
from cython cimport cdivision, boundscheck, wraparound
from libcpp cimport bool as bool_cpp
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.algorithm cimport sort
from libc.stdlib cimport malloc, free
from libc.math cimport fabs
import numpy as np
cimport numpy as np
    


@boundscheck(False)
@wraparound(False) 
cpdef bool_cpp is_sorted(double[:] arr):
    cdef long i
    cdef size_t size = arr.shape[0]
    for i in range(size - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True


@boundscheck(False)
@wraparound(False) 
@cdivision(True)  
cpdef compute_voronoi1d_cell_extents(double[:] depth, 
                                     double lb=0, 
                                     double ub=-1,
                                     double fill_value=0):
    cdef size_t size = depth.shape[0]
    cdef int i
    cdef double d1, d2
    cdef double[:] thickness = np.zeros(size, dtype=np.double)
    d1 = lb
    for i in range(size - 1):
        d2 = (depth[i] + depth[i+1]) / 2.
        thickness[i] = d2 - d1 if d1>=0 else fill_value
        d1 = d2
    thickness[i+1] = ub - d2 if ub>=0 else fill_value
    return np.asarray(thickness)


@boundscheck(False)
@wraparound(False) 
cdef int floor_index(double xp, double[:] x, ssize_t xlen):
    cdef int i
    for i in range(xlen):
        if x[i] <= xp <= x[i+1]:
            return i


@boundscheck(False)
@wraparound(False) 
@cdivision(True)  
cdef double _interpolate_linear_1d(double xp, double[:] x, double[:] y):
    cdef size_t xlen = x.shape[0]
    cdef int i
    cdef double x0, x1, y0, y1
    
    if xp <= x[0]:
        return y[0]
    elif xp >= x[xlen-1]:
        return y[xlen-1]
    i = floor_index(xp, x, xlen)
    x0 = x[i]
    x1 = x[i+1]
    y0 = y[i]
    y1 = y[i+1]
    return y0 + (xp - x0) * (y1 - y0) / (x1 - x0)


@boundscheck(False)
@wraparound(False) 
@cdivision(True)  
cdef double _interpolate_nearest_1d(double xp, double[:] x, double[:] y):
    cdef ssize_t xlen = x.shape[0]
    cdef int i
    cdef double x0, x1, y0, y1
    
    if xp <= x[0]:
        return y[0]
    elif xp >= x[xlen-1]:
        return y[xlen-1]
    i = nearest_neighbour_1d(xp, x, xlen)
    return y[i]


@boundscheck(False)
@wraparound(False) 
cpdef int nearest_neighbour_1d(double xp, double[:] x, ssize_t xlen):
    cdef int i
    cdef double x0, x1
    if xlen==1 or xp<x[0]:
        return 0
    for i in range(xlen - 1):
        x0 = x[i]
        x1 = x[i + 1]
        if x0 <= xp <= x1:
            if fabs(x0-xp) < fabs(x1-xp):
                return i
            return i + 1
    return i + 1


@boundscheck(False)
@wraparound(False) 
cpdef interpolate_depth_profile(double[:] x, double[:] y, double[:] x0):
    cdef size_t ilayer = 0
    cdef size_t i
    cdef size_t x0_len = x0.shape[0]
    cdef size_t x_len = x.shape[0]
    cdef double interface_lower = 0
    cdef double interface_upper = x[0]
    cdef double x_i
    cdef double[:] ynew = np.zeros(x0_len, dtype=np.double)
    
    for i in range(x0_len):
        x_i = x0[i]
        while True:
            if interface_lower <= x_i < interface_upper or \
                ilayer + 1 >= x_len:
                ynew[i] = y[ilayer]
                break
            else:
                ilayer += 1
                interface_lower = interface_upper
                interface_upper += x[ilayer]
    return np.asarray(ynew)


@boundscheck(False)
@wraparound(False) 
cpdef interpolate_linear_1d(xp, double[:] x, double[:] y):
    """ Linear interpolation in 1-D
    Parameters
    ----------
    xp : float, ndarray of floats
        Interpolation point(s)
        
    x, y : ndarray of floats
        Discrete set of known data points
    
    Returns
    -------
    yp : float
        Interpolated value(s)
    """
    if np.isscalar(xp):
        return _interpolate_linear_1d(xp, x, y)
    cdef int i, size = xp.shape[0]
    cdef double[:] yp = np.zeros(size, dtype=np.double)
    for i in range(size):
        yp[i] = _interpolate_linear_1d(xp[i], x, y)
    return np.asarray(yp)


@boundscheck(False)
@wraparound(False) 
cpdef interpolate_nearest_1d(xp, double[:] x, double[:] y):
    """ Nearest-neighbour interpolation in 1-D
    Parameters
    ----------
    xp : float, ndarray of floats
        Interpolation point(s)
        
    x, y : ndarray of floats
        Discrete set of known data points
    
    Returns
    -------
    yp : float
        Interpolated value(s)
    """
    if np.isscalar(xp):
        return _interpolate_nearest_1d(xp, x, y)
    cdef int i, size = xp.shape[0]
    cdef double[:] yp = np.zeros(size, dtype=np.double)
    for i in range(size):
        yp[i] = _interpolate_nearest_1d(xp[i], x, y)
    return np.asarray(yp)


@boundscheck(False)
@wraparound(False) 
@cdivision(True)
cpdef inverse_covariance(double sigma, double r, size_t n):
    cdef size_t i
    cdef double factor = 1 / (sigma**2 * (1 - r**2))
    cdef double[:, ::1] matrix = np.zeros((n, n), dtype=np.double)
    
    for i in range(1, n-1):
        matrix[i, i] = (1 + r**2) * factor
        matrix[i, i-1] = -r * factor
        matrix[i-1, i] = -r * factor
    matrix[0, 0] = factor
    matrix[n-1, n-1] = factor
    matrix[i+1, i] = -r * factor
    matrix[i, i+1] = -r * factor
    return np.asarray(matrix)
    

@boundscheck(False)
@wraparound(False) 
cpdef insert_1d(double[:] values, long index, double value):
    cdef size_t new_size = values.shape[0] + 1
    cdef double[:] new_values = np.zeros(new_size, dtype=np.double)
    cdef size_t i = 0
    cdef size_t j = 0
    while i < new_size:
        if i == index:
            new_values[i] = value
        else:
            new_values[i] = values[j]
            j += 1
        i += 1
    return np.asarray(new_values)


@boundscheck(False)
@wraparound(False) 
cpdef delete_1d(double[:] values, long index):
    cdef size_t size = values.shape[0]
    cdef double[:] new_values = np.zeros(size - 1, dtype=np.double)
    cdef size_t i, j = 0
    for i in range(size):
        if i != index:
            new_values[j] = values[i]
            j += 1
    return np.asarray(new_values)

