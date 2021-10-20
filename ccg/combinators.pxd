
from .category cimport Category
from .derivation cimport *


cdef class Combinator:
    pass

cdef class UnaryCombinator(Combinator):
    cdef readonly:
        bint is_type_raise
        bint is_type_change
        bint is_unary_coord
        bint is_XY_to_ZY_change
        bint is_X_to_XX_change

    cpdef bint can_apply(self, Category in_cat)
    cpdef Category apply(self, Category in_cat)
    cdef Node t(self, Node in_node)

cdef class TypeChanging1(UnaryCombinator):
    cdef readonly:
        Category cat_from
        Category cat_to

cdef class TypeRaising(UnaryCombinator):
    cdef readonly:
        bint is_order_preserving
        bint is_forward
        Category cat_res
        Category cat_arg

cdef class BinaryCombinator(Combinator):

    cdef readonly:
        bint is_glue
        bint is_type_change_binary
        bint is_conj_top
        bint is_conj_bottom
        bint is_special_right_adj
        bint is_B0_bck
        bint is_B0_fwd
        bint is_B_bck
        bint is_B_fwd
        bint is_punc
        bint is_punc_right
        bint is_punc_left
        bint is_tc_X_Y_to_Xconj
        bint is_tc_X_Y_to_Yconj
        bint is_tc_A_B_to_A
        bint is_tc_A_B_to_B
        bint is_tc_A_XY_to_ZY
        bint is_tc_XY_A_to_ZY

    cpdef bint can_apply(self, Category left, Category right)
    cpdef Category apply(self, Category left, Category right)
    cdef Node t(self, Node left, Node right)

    cpdef bint is_left_adj_comb(self, Category x, Category y)
    cpdef bint is_right_adj_comb(self, Category x, Category y)

cdef class RightAdjoin(BinaryCombinator):
    cdef readonly:
        tuple span
        Category cat

cdef class Glue(BinaryCombinator):
    pass

cdef class TypeChanging2(BinaryCombinator):
    cdef readonly:
        Category left
        Category right
        Category parent

cdef class Conj(BinaryCombinator):
    cdef readonly:
        bint is_bottom
        bint is_top

cdef class Punc(BinaryCombinator):
    cdef readonly:
        bint punc_is_left

cdef class B(BinaryCombinator):
    cdef readonly:
        bint is_forward
        bint is_backward
        int order
        bint is_crossed
        bint is_harmonic

cdef class S(BinaryCombinator):
    cdef readonly:
        bint is_forward
        bint is_backward
        bint is_crossed
        bint is_harmonic
