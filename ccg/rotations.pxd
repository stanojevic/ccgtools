
from .combinators cimport *
from .derivation cimport *
from .category cimport Category

cdef class TreeTransducer:

    cdef readonly:
        bint is_extreme
        int max_fwd_B
        int max_bck_B
        list all_backward_B_variations
        dict type_raising_table

    cdef tuple _extract_right_adjunct_and_punc_subtrees(self, Node node)

    cdef Node _add_revealing_rec(self, Node node)

    cdef Node _reattach_punc_to_left(self, Node root)

    cdef Node _reattach_adjunct_or_punc(self, Node core, BinaryCombinator comb, Node adj)

    cdef Node _right_modify(self, Node left, Node right, tuple span, Category cat)

    cdef Node _attach_left_punc_at_bottom(self, Node punc, Node node)

    cdef Node _attach_right_punc_at_bottom(self, Node node, Node punc)

    cdef TypeRaising _type_raise_lookup(self, bint is_fwd, Category cat_from, Category cat_result)

    cpdef Node to_left_branching(self, Node tree)

    cdef Node _to_left_branching_simplistic(self, Node node)

    cpdef Node to_right_branching(self, Node node)

    cdef Node _to_right_branching(self, Node node)

    cdef Node _remove_unneeded_type_raising_single(self, Node node)

    cdef bint _is_b0_node_that_has_tr_child(self, Node node)

    cdef Node _remove_unneeded_type_raising_for_rebalancing(self, Node node)

    cpdef Node sink_rightward(self, Node node)
    cdef Node _sink_rightward_rule_1(self, Binary node)
    cdef Node _sink_rightward_rule_2(self, Binary node)
    cdef Node _sink_rightward_rule_3(self, Binary node)
    cdef Node _sink_rightward_rule_4(self, Binary node)
    cdef Node _sink_rightward_rule_5(self, Binary node)

    cpdef Node sink_leftward(self, Node node)
    cdef Node _sink_leftward_rule_5(self, Binary node)
    cdef Node _sink_leftward_rule_4(self, Binary node)
    cdef Node _sink_leftward_rule_3(self, Binary node)
    cdef Node _sink_leftward_rule_2(self, Binary node)
    cdef Node _sink_leftward_rule_1(self, Binary node)
    cdef Node _sink_leftward_rule_1_rebuild(self, Node core_left, int parent_order, Node node)
