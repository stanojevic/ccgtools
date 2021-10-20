
from cpython cimport bool, unicode


cpdef dict matching(Category cat_a , Category cat_b)  # -> Optional[Dict[str, Tuple[bool, str]]]:

cdef dict filter_substitution(dict sub, bint from_left)  # -> Dict[str, str]:

cdef class Category:

    cdef readonly:
        bint is_atomic
        bint is_functor
        bint is_conj
        bint is_punc
        bint is_left_adj_cat   # X/X
        bint is_right_adj_cat  # X\X
        bint is_adj_cat        # X|X
        bint is_np
        bint is_pp
        bint is_n
        bint is_nominal  # nominal is NP or N
        bint is_s
        bint is_verbal  # verbal category is of shape S|X
        int pa_id
        unicode pa_b

    cpdef tuple cut_args(self, int args_to_drop=*)  # -> Optional[Tuple[Category, List[Tuple[Slash, Category]]]]:

    @staticmethod
    cdef Category _process_tokens(list tokens)

    @staticmethod
    cdef list _tokenize(unicode string)

    @staticmethod
    cdef Category _combine_cats_and_slashes(list cats, list slashes)

    cdef Category apply_substitution(self, dict sub)

    cdef bint equals(self, Category y)
    cdef bint equals_featureless(self, Category y)

cdef class Atomic(Category):

    cdef readonly unicode label
    cdef readonly unicode feature
    cdef readonly bool has_feature
    cdef readonly bool is_conj_atom

    cpdef Atomic strip_features(self)

cdef class ConjCat(Category):

    cdef readonly Category sub_cat

    cpdef ConjCat strip_features(self)

cdef class Functor(Category):

    cdef readonly:
        bint slash_is_fwd
        Category res
        Category arg

    cpdef Functor strip_features(self)

