import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef Py_ssize_t uint

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef conv2d( 
    np.ndarray[DTYPE_t, ndim = 4] volume, # n, h, w, f
    np.ndarray[DTYPE_t, ndim = 4] kernel, # kh, kw, f, f_out
    np.ndarray[DTYPE_t, ndim = 4] conved, # n h_out, w_out, f_out
    int n, int h, int w, int f, int out_f,
    int kh, int kw, int sh, int sw):

    cdef uint out_h = (h - kh) / sh + 1
    cdef uint out_w = (w - kw) / sw + 1

    cdef uint colcol = kh * kw * f
    cdef uint fkw = kw * f
    cdef uint n_idx, ho, wo, kcoli, fo, fi, kwi, khi

    for n_idx in range(n):
        for ho in range(out_h):
            for wo in range(out_w):
                for kcoli in range(colcol):
                    fi = kcoli % f
                    kwi = kcoli / f % kw
                    khi = kcoli / fkw
                    for fo in range(out_f):
                        conved[n_idx, ho, wo, fo] += \
                            volume[n_idx, 
                                ho * sh + khi, wo * sw + kwi, fi] * \
                            kernel[khi, kwi, fi, fo]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef conv2d_grad_kernel(
    np.ndarray[DTYPE_t, ndim = 4] volume, # n, h, w, f
    np.ndarray[DTYPE_t, ndim = 4] grad_kernel, # kh, kw, f, f_out 
    np.ndarray[DTYPE_t, ndim = 4] conved, # n, h_out, w_out, f_out
    int n, int h, int w, int f, int out_f,
    int kh, int kw, int sh, int sw):

    cdef uint out_h = (h - kh) / sh + 1
    cdef uint out_w = (w - kw) / sw + 1

    cdef uint colcol = kh * kw * f
    cdef uint fkw = kw * f
    cdef uint n_idx, ho, wo, kcoli, fo, fi, kwi, khi

    for n_idx in range(n):
        for ho in range(out_h):
            for wo in range(out_w):
                for kcoli in range(colcol):
                    fi = kcoli % f
                    kwi = kcoli / f % kw
                    khi = kcoli / fkw
                    for fo in range(out_f):
                        grad_kernel[khi, kwi, fi, fo] += \
                            volume[n_idx, 
                                ho * sh + khi, wo * sw + kwi, fi] * \
                            conved[n_idx, ho, wo, fo]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef conv2d_grad_volume( 
    np.ndarray[DTYPE_t, ndim = 4] grad_volume, # n, h, w, f
    np.ndarray[DTYPE_t, ndim = 4] kernel, # kh, kw, f, f_out
    np.ndarray[DTYPE_t, ndim = 4] conved, # n, h_out, w_out, f_out
    int n, int h, int w, int f, int out_f,
    int kh, int kw, int sh, int sw):

    cdef uint out_h = (h - kh) / sh + 1
    cdef uint out_w = (w - kw) / sw + 1

    cdef uint colcol = kh * kw * f
    cdef uint fkw = kw * f
    cdef uint n_idx, ho, wo, kcoli, fo, fi, kwi, khi

    for n_idx in range(n):
        for ho in range(out_h):
            for wo in range(out_w):
                for kcoli in range(colcol):
                    fi = kcoli % f
                    kwi = kcoli / f % kw
                    khi = kcoli / fkw
                    for fo in range(out_f):
                        grad_volume[n_idx, ho * sh + khi, wo * sw + kwi, fi] += \
                            kernel[khi, kwi, fi, fo] * \
                            conved[n_idx, ho, wo, fo]