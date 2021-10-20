# cython: boundscheck=False

import re
from .category cimport Category, Functor, Atomic, ConjCat, matching, filter_substitution
from .derivation cimport *


cdef class Combinator:

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


cdef class UnaryCombinator(Combinator):

    cpdef bint can_apply(self, Category in_cat):
        raise NotImplementedError

    cpdef Category apply(self, Category in_cat):
        raise NotImplementedError

    cdef inline Node t(self, Node in_node):
        return Unary(self, in_node)

    def __call__(self, in_cat):
        if isinstance(in_cat, Category):
            return self.apply(in_cat)
        else:
            return Unary(self, in_cat)


cdef class TypeChanging1(UnaryCombinator):

    def __cinit__(self, Category cat_from, Category cat_to):
        self.cat_from = cat_from
        self.cat_to = cat_to

        self.is_type_raise = False
        self.is_type_change = True
        self.is_unary_coord = self.cat_to.is_conj

        cdef Category y1, y2

        if cat_from.is_functor and cat_to.is_functor:
            y1 = cat_from.arg
            y2 = cat_to.arg
            self.is_XY_to_ZY_change = (y1.equals_featureless(y2) or (y1.is_np and y2.is_n))
        else:
            self.is_XY_to_ZY_change = False

        self.is_X_to_XX_change = (cat_to.is_adj_cat and cat_from.equals_featureless((<Functor> cat_to).res))

    cpdef bint can_apply(self, in_cat: Category):
        return self.cat_from.equals(in_cat)

    cpdef Category apply(self, in_cat: Category):
        return self.cat_to

    def __reduce__(self):
        return TypeChanging1, (self.cat_from, self.cat_to)

    def __str__(self):
        return "TC"

    def __eq__(self, other):
        if isinstance(other, UnaryCombinator):
            return self.cat_to.equals(other.cat_to) and self.cat_from.equals(other.cat_from)
        else:
            return False

    def __hash__(self):
        return hash((self.cat_from, self.cat_to))


cdef class TypeRaising(UnaryCombinator):

    def __cinit__(self, Category cat_res, Category cat_arg, bint is_forward, bint is_order_preserving = True):
        self.is_order_preserving = is_order_preserving
        self.is_forward = is_forward
        self.cat_res = cat_res
        self.cat_arg = cat_arg

        self.is_type_raise = True
        self.is_type_change = False
        self.is_unary_coord = False
        self.is_XY_to_ZY_change = False
        self.is_X_to_XX_change = False

    cpdef bint can_apply(self, Category in_cat):
        return self.cat_arg.equals_featureless(in_cat)

    cpdef Category apply(self, Category in_cat):
        top_slash_is_fwd = self.is_forward
        bottom_slash_is_fwd = (not top_slash_is_fwd) if self.is_order_preserving else top_slash_is_fwd
        return Functor(
            slash_is_fwd=top_slash_is_fwd,
            res=self.cat_res,
            arg=Functor(
                slash_is_fwd=bottom_slash_is_fwd,
                res=self.cat_res,
                arg=in_cat
            )
        )

    def __reduce__(self):
        return TypeRaising, (self.cat_res, self.cat_arg, self.is_forward, self.is_order_preserving)

    def __str__(self):
        if self.is_forward:
            return ">T"
        else:
            return "<T"

    def __eq__(self, other):
        if isinstance(other, TypeRaising):
            return self.cat_arg.equals(other.cat_arg) and self.cat_res.equals(other.cat_res)
        else:
            return False

    def __hash__(self):
        return hash((self.cat_res, self.cat_arg, self.is_forward, self.is_order_preserving))


cdef class BinaryCombinator(Combinator):

    def __cinit__(self):
        self.is_punc_left = False
        self.is_punc_right = False
        self.is_punc = False
        self.is_glue = False
        self.is_type_change_binary = False
        self.is_conj_top = False
        self.is_conj_bottom = False
        self.is_special_right_adj = False
        self.is_B0_bck = False
        self.is_B0_fwd = False
        self.is_B_bck = False
        self.is_B_fwd = False
        self.is_tc_X_Y_to_Xconj = False
        self.is_tc_X_Y_to_Yconj = False
        self.is_tc_A_B_to_A = False
        self.is_tc_A_B_to_B = False
        self.is_tc_A_XY_to_ZY = False
        self.is_tc_XY_A_to_ZY = False

    cpdef bint can_apply(self, Category left, Category right):
        raise NotImplementedError

    cpdef Category apply(self, Category left, Category right):
        raise NotImplementedError

    cdef inline Node t(self, Node left, Node right):
        return Binary(self, left, right)

    def __call__(self, left, right):
        if isinstance(left, Category) and isinstance(right, Category):
            return self.apply(left, right)
        else:
            from .derivation import Node, Binary
            if isinstance(left, Node) and isinstance(right, Node):
                return Binary(self, left, right)
            else:
                raise Exception("wrong type or arguments")

    cpdef bint is_left_adj_comb(self, Category x, Category y):
        return x.is_left_adj_cat and \
               self.can_apply(x, y) and \
               self.apply(x, y).equals(y) and \
               self.is_B_fwd and (<B> self).order == 0  # TODO should we constrain it to order==0?

    cpdef bint is_right_adj_comb(self, Category x, Category y):
        return (y.is_right_adj_cat or y.is_conj) and \
               self.can_apply(x, y) and \
               self.apply(x, y).equals(x) and \
               (self.is_B_bck or self.is_conj_top)


cdef class RightAdjoin(BinaryCombinator):
    """used for revealing (tree-rotation), not for actual parsing"""

    def __reduce__(self):
        return RightAdjoin, (self.span, self.cat)

    def __cinit__(self, tuple span, Category cat):
        self.span = span
        self.cat = cat

        self.is_special_right_adj = True

    cpdef Category apply(self, Category left, Category right):
        return left

    cpdef bint can_apply(self, Category left, Category right):
        return True

    def __str__(self):
        return "right-adjoin[%s starting at %d]" % (self.cat, self.span[0])

    def __hash__(self):
        return hash((31561, self.span, self.cat))

    def __eq__(self, other):
        return isinstance(other, RightAdjoin) and \
               other.span == self.span and \
               other.cat.equals(self.cat)


cdef class Glue(BinaryCombinator):
    """used to connect components of a failed parse"""

    def __reduce__(self):
        return Glue, ()

    def __cinit__(self):
        self.is_glue = True

    cpdef bint can_apply(self, Category left, Category right):
        return True

    cpdef Category apply(self, Category left, Category right):
        return left

    def __str__(self):
        return "GLUE"

    def __eq__(self, other):
        return isinstance(other, Glue)

    def __hash__(self):
        return 152341


cdef class TypeChanging2(BinaryCombinator):

    def __reduce__(self):
        return TypeChanging2, (self.left, self.right, self.parent)

    def __cinit__(self, Category left, Category right, Category parent):
        self.left = left
        self.right = right
        self.parent = parent

        self.is_type_change_binary = True

        cdef Functor fparent, fleft, fright
        cdef ConjCat cparent

        if parent.is_conj:
            cparent = parent
            self.is_tc_X_Y_to_Xconj = cparent.sub_cat.equals(left)
            self.is_tc_X_Y_to_Yconj = cparent.sub_cat.equals(right)
        else:
            self.is_tc_X_Y_to_Xconj = False
            self.is_tc_X_Y_to_Yconj = False

        self.is_tc_A_B_to_A = parent.equals(left)
        self.is_tc_A_B_to_B = parent.equals(right)

        if parent.is_functor:
            fparent = parent
            if left.is_functor:
                fleft = left
                self.is_tc_XY_A_to_ZY = right.is_atomic and fparent.arg.equals_featureless(fleft.arg)
                self.is_tc_A_XY_to_ZY = False
            elif right.is_functor:
                fright = right
                self.is_tc_XY_A_to_ZY = False
                self.is_tc_A_XY_to_ZY = left.is_atomic and fparent.arg.equals_featureless(fright.arg)
            else:
                self.is_tc_XY_A_to_ZY = False
                self.is_tc_A_XY_to_ZY = False
        else:
            self.is_tc_XY_A_to_ZY = False
            self.is_tc_A_XY_to_ZY = False

    cpdef bint can_apply(self, Category left, Category right):
        return self.left.equals(left) and self.right.equals(right)

    cpdef Category apply(self, Category left, Category right):
        return self.parent

    def __str__(self):
        return "TC2"

    def __eq__(self, other):
        if isinstance(other, TypeChanging2):
            return self.left.equals(other.left) and \
                   self.right.equals(other.right) and \
                   self.parent.equals(other.parent)
        else:
            return False

    def __hash__(self):
        return hash((self.left, self.right, self.parent))


cdef class Conj(BinaryCombinator):

    def __reduce__(self):
        return Conj, (self.is_bottom,)

    def __cinit__(self, bint is_bottom):
        self.is_bottom = is_bottom  # TODO remove
        self.is_top = not is_bottom  # TODO remove

        self.is_glue = False
        self.is_conj_top = not is_bottom
        self.is_conj_bottom = is_bottom

    cpdef bint can_apply(self, Category left, Category right):
        if self.is_bottom:
            return left.is_atomic and left.is_conj_atom and not right.is_conj
        else:
            return right.is_conj and left.equals((<ConjCat> right).sub_cat)

    cpdef Category apply(self, Category left, Category right):
        if self.is_bottom:
            return ConjCat(sub_cat=right)
        else:
            return left

    def __str__(self):
        arrow = ">" if self.is_bottom else "<"
        return arrow+"Φ"

    def __eq__(self, other):
        return isinstance(other, Conj) and other.is_bottom == self.is_bottom

    def __hash__(self):
        return hash((31515341, self.is_bottom))


cdef class Punc(BinaryCombinator):

    def __reduce__(self):
        return Punc, (self.punc_is_left,)

    def __cinit__(self, bint punc_is_left):
        self.punc_is_left = punc_is_left

        self.is_punc_left = punc_is_left
        self.is_punc_right = not punc_is_left
        self.is_punc = True

    cpdef bint can_apply(self, Category left, Category right):
        if self.punc_is_left:
            return left.is_punc
        else:
            return right.is_punc

    cpdef Category apply(self, Category left, Category right):
        if self.punc_is_left:
            return right
        else:
            return left

    def __str__(self):
        if self.punc_is_left:
            return "<P"
        else:
            return ">P"

    def __eq__(self, other):
        if isinstance(other, Punc):
            return other.punc_is_left == self.punc_is_left
        else:
            return False

    def __hash__(self):
        return hash((3161602, self.punc_is_left))


cdef class B(BinaryCombinator):

    def __reduce__(self):
        return B, (self.is_forward, self.order, self.is_crossed)

    def __cinit__(self, bint is_forward, int order, bint is_crossed):
        self.is_forward = is_forward
        self.is_backward = not self.is_forward
        self.order = order
        if is_crossed and self.order==0:
            raise Exception("you can't have Bx0 combinator")
        self.is_crossed  = is_crossed
        self.is_harmonic = not is_crossed

        self.is_B0_bck = not is_forward and order==0
        self.is_B0_fwd = is_forward and order==0
        self.is_B_bck = not is_forward
        self.is_B_fwd = is_forward

    cpdef bint can_apply(self, Category left, Category right):
        cdef Category main, side, arg, side_cat
        cdef tuple x, y
        cdef bint slash_is_fwd
        cdef list slashargs
        cdef dict sub
        if self.is_forward:
            main, side = left, right
        else:
            main, side = right, left
        x = main.cut_args(1)
        if not x:
            return False
        main_cat, [(slash_is_fwd, arg)] = x
        # if self.is_crossed and not arg.is_verbal:
        #     return False  # constraint on crossed composition to be only over verbal cats
        # if self.order>0 and self.is_harmonic and arg.is_np:
        #     return False  # no composition over NP
        if slash_is_fwd != self.is_forward:
            return False
        y = side.cut_args(self.order)
        if not y:
            return False
        side_cat, slashargs = y
        if self.order>0 and (self.is_harmonic != (slash_is_fwd == slashargs[0][0])):
            return False
        sub = matching(arg, side_cat)
        return sub is not None

    cpdef Category apply(self, Category left, Category right):
        cdef Category main_cat, side_cat, main, side, arg
        cdef bint slash
        cdef list slashargs
        cdef dict sub_from_left, sub_from_right, sub
        if self.is_forward:
            main, side = left, right
        else:
            main, side = right, left
        main_cat, [(_, arg)] = main.cut_args(1)
        side_cat, slashargs = side.cut_args(self.order)
        sub = matching(arg, side_cat)
        sub_from_left  = filter_substitution(sub=sub, from_left=True)
        sub_from_right = filter_substitution(sub=sub, from_left=False)
        main_cat = main_cat.apply_substitution(sub_from_right)
        for slash, arg in slashargs:
            main_cat = Functor(slash_is_fwd=slash, res=main_cat, arg=arg.apply_substitution(sub_from_left))
        return main_cat

    def __str__(self):
        arrow = ">" if self.is_forward else "<"
        cross = "" if self.is_harmonic else "x"
        if self.order == 0:
            return arrow
        elif self.order == 1:
            return arrow + "B" + cross
        else:
            return arrow + "B" + str(self.order) + cross

    def __eq__(self, other):
        if isinstance(other, B):
            return self.order == other.order and \
                   self.is_harmonic == other.is_harmonic and \
                   self.is_forward == other.is_forward
        else:
            return False

    def __hash__(self):
        return hash((32115, self.order, self.is_forward, self.is_harmonic))


cdef class S(BinaryCombinator):
    """In case of <Sx rule is Y/Z (X\\Y)/Z => X/Z where X = S\\$"""

    def __reduce__(self):
        return S, (self.is_forward, self.is_crossed)

    def __cinit__(self, bint is_forward = False, bint is_crossed = False):
        self.is_forward = is_forward
        self.is_backward = not self.is_forward
        self.is_crossed = is_crossed
        self.is_harmonic = not is_crossed

    cpdef bint can_apply(self, Category left, Category right):
        cdef Category main, side, main_cat, cat1, cat2, cat3
        cdef bint slash1_is_fwd, slash2_is_fwd
        if self.is_forward:
            main, side = left, right
        else:
            main, side = right, left
        x = main.cut_args(2)
        if not x:
            return False
        main_cat, [(slash1_is_fwd, cat1), (slash2_is_fwd, cat2)] = x
        if self.is_crossed and not main_cat.is_verbal:
            return False  # you can't used crossed substitution with non-verbal cats
        if self.is_harmonic and slash1_is_fwd != slash2_is_fwd:
            return False
        if self.is_forward == (not slash1_is_fwd):
            return False
        cat3 = Functor(slash_is_fwd=slash2_is_fwd, res=cat1, arg=cat2)
        sub = matching(cat3, side)
        return sub is not None

    cpdef Category apply(self, Category left, Category right):
        cdef Category main, side, main_cat, cat1, cat2, cat3
        cdef bint slash1, slash2
        if self.is_forward:
            main, side = left, right
        else:
            main, side = right, left
        main_cat, [(slash1, cat1), (slash2, cat2)] = main.cut_args(2)
        cat3 = Functor(slash_is_fwd=slash2, res=cat1, arg=cat2)
        sub = matching(cat3, side)
        sub = filter_substitution(sub=sub, from_left=False)
        main_cat = main_cat.apply_substitution(sub)
        return main_cat

    def __str__(self):
        x = ">" if self.is_forward else "<"
        y = "" if self.is_harmonic else "x"
        return x+"S"+y

    def __hash__(self):
        return hash((self.is_forward, self.is_harmonic))

    def __eq__(self, other):
        if isinstance(other, S):
            return other.is_forward == self.is_forward and other.is_harmonic == self.is_harmonic
        else:
            return False


