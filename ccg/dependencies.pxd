from .combinators cimport *
from .category cimport *
from .derivation cimport *

cdef class DepLink:

    cdef readonly Category head_cat
    cdef readonly int head_pos
    cdef readonly int dep_pos
    cdef readonly int dep_slot
    cdef readonly str head_word
    cdef readonly str dep_word
    cdef readonly bint is_bound
    cdef readonly bint is_unbound
    cdef readonly bint is_adj
    cdef readonly bint is_conj
    cdef readonly int _hash

    cpdef tuple __reduce__(self)
