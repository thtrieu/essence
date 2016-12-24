import numpy as np
from cython.parallel import prange, parallel
cimport numpy as np
cimport cython

cdef extern void conv_vko(float* v, float* k, float* o, 
    int n, int h, int w, int f, 
    int ph, int pw, int out_f,
    int kh, int kw, int sh, int sw)
cdef extern void conv_vok(float* v, float* k, float* o, 
    int n, int h, int w, int f,  
    int ph, int pw, int out_f,
    int kh, int kw, int sh, int sw)
cdef extern void conv_kov(float* v, float* k, float* o, 
    int n, int h, int w, int f,  
    int ph, int pw, int out_f,
    int kh, int kw, int sh, int sw)
cdef extern void xpool2_vo(float* v, int* m, float* o, 
    int n, int h, int w, int f)
cdef extern void xpool2_ov(float* v, int* m, float* o, 
    int n, int h, int w, int f)

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
ctypedef Py_ssize_t uint

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef c_xpool2( 
    np.ndarray[DTYPE_t, ndim = 4, mode = "c"] volume, # n, h, w, f
    np.ndarray[int, ndim = 4, mode = "c"] marked, # kh, kw, f, f_out
    np.ndarray[DTYPE_t, ndim = 4, mode = "c"] pooled, # n h_out, w_out, f_out
    int h, int w, int f):

    cdef uint n = volume.shape[0]
    xpool2_vo(&volume[0,0,0,0], &marked[0,0,0,0], &pooled[0,0,0,0],
        n, h, w, f)
    return None

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef c_gradxp2( 
    np.ndarray[DTYPE_t, ndim = 4, mode = "c"] volume, # n, h, w, f
    np.ndarray[int, ndim = 4, mode = "c"] marked, # kh, kw, f, f_out
    np.ndarray[DTYPE_t, ndim = 4, mode = "c"] pooled, # n h_out, w_out, f_out
    int h, int w, int f):

    cdef uint n = volume.shape[0]
    xpool2_ov(&volume[0,0,0,0], &marked[0,0,0,0], &pooled[0,0,0,0],
        n, h, w, f)
    return None

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef c_conv2d( 
    np.ndarray[DTYPE_t, ndim = 4, mode = "c"] volume, # n, h, w, f
    np.ndarray[DTYPE_t, ndim = 4, mode = "c"] kernel, # kh, kw, f, f_out
    np.ndarray[DTYPE_t, ndim = 4, mode = "c"] conved, # n h_out, w_out, f_out
    int h, int w, int f, 
    int ph, int pw, int out_f,
    int kh, int kw, int sh, int sw):
    
    cdef uint n = volume.shape[0]
    conv_vko(&volume[0,0,0,0], &kernel[0,0,0,0], &conved[0,0,0,0],
        n, h, w, f, ph, pw, out_f, kh, kw, sh, sw)
    return None

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef c_gradk( 
    np.ndarray[DTYPE_t, ndim = 4, mode = "c"] volume, # n, h, w, f
    np.ndarray[DTYPE_t, ndim = 4, mode = "c"] kernel, # kh, kw, f, f_out
    np.ndarray[DTYPE_t, ndim = 4, mode = "c"] conved, # n h_out, w_out, f_out
    int h, int w, int f, 
    int ph, int pw, int out_f,
    int kh, int kw, int sh, int sw):

    cdef uint n = volume.shape[0]
    conv_vok(&volume[0,0,0,0], &kernel[0,0,0,0], &conved[0,0,0,0],
        n, h, w, f, ph, pw, out_f, kh, kw, sh, sw)
    return None

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef c_gradx( 
    np.ndarray[DTYPE_t, ndim = 4, mode = "c"] volume, # n, h, w, f
    np.ndarray[DTYPE_t, ndim = 4, mode = "c"] kernel, # kh, kw, f, f_out
    np.ndarray[DTYPE_t, ndim = 4, mode = "c"] conved, # n h_out, w_out, f_out
    int h, int w, int f, 
    int ph, int pw, int out_f,
    int kh, int kw, int sh, int sw):

    cdef uint n = volume.shape[0]
    conv_kov(&volume[0,0,0,0], &kernel[0,0,0,0], &conved[0,0,0,0],
        n, h, w, f, ph, pw, out_f, kh, kw, sh, sw)
    return None