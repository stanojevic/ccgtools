
from .categories cimport Category
from .combinators cimport BinaryCombinator, UnaryCombinator

cdef class Node:

    cdef dict __dict__

    cdef readonly:
        Category cat
        bint is_term
        bint is_binary
        bint is_unary
        tuple _span_memo
        list children

    cdef public:
        object predarg
        # object h
        # object c

    cpdef void assign_word_positions(self)

    cpdef list words(self)

    cpdef list stags(self)


cdef class Terminal(Node):

    cdef readonly unicode word
    cdef public int pos


cdef class Unary(Node):

    cdef readonly:
        UnaryCombinator comb
        Node child


cdef class Binary(Node):

    cdef readonly:
        BinaryCombinator comb
        Node left
        Node right
