# cython: boundscheck=False

from .category cimport *
from .derivation cimport *
from .combinators cimport *

from .grammar import Grammar


cdef BinaryCombinator f_app      = B(is_forward=True , is_crossed=False, order=0)
cdef BinaryCombinator f_comp     = B(is_forward=True , is_crossed=False, order=1)
cdef BinaryCombinator b_app      = B(is_forward=False, is_crossed=False, order=0)
cdef BinaryCombinator b_comp     = B(is_forward=False, is_crossed=False, order=1)
cdef BinaryCombinator b_comp_x   = B(is_forward=False, is_crossed=True , order=1)
cdef UnaryCombinator  tr         = TypeRaising(cat_res=Category.from_str("S"), cat_arg=Category.from_str("NP"), is_forward=True)
cdef list b_fwd_harmonic_lookup = [B(is_forward=True, is_crossed=False, order=0),
                                   B(is_forward=True, is_crossed=False, order=1),
                                   B(is_forward=True, is_crossed=False, order=2),
                                   B(is_forward=True, is_crossed=False, order=3)]

cdef inline Node _bsafe(BinaryCombinator c, Node x, Node y):
    if (x is not None) and (y is not None) and (c.can_apply(x.cat, y.cat)):
        return c.t(x, y)
    else:
        return None

cdef inline Node _usafe(UnaryCombinator c, Node x):
    if c is not None:
        return c.t(x)
    else:
        return None

cdef class TreeTransducer:

    def __init__(self, bint is_extreme, object g: Grammar):
        self.is_extreme = is_extreme
        self.max_fwd_B = g.max_fwd_B
        self.max_bck_B = g.max_bck_B
        self.all_backward_B_variations = g.all_backward_B_variations
        self.type_raising_table = dict()
        for t in g.unary_normal:
            if t.is_type_raise and t.is_order_preserving:
                self.type_raising_table[(t.is_forward, t.cat_arg, t.cat_res.strip_features())] = t

    cdef tuple _extract_right_adjunct_and_punc_subtrees(self, Node node):
        cdef Node core_left, core_right, core_child
        cdef list left_right_adjs, right_right_adjs, right_adjs
        cdef Binary bnode
        cdef Unary unode
        if node.is_binary:
            bnode = node
            core_left , left_right_adjs  = self._extract_right_adjunct_and_punc_subtrees(bnode.left )
            core_right, right_right_adjs = self._extract_right_adjunct_and_punc_subtrees(bnode.right)
            if bnode.comb.is_special_right_adj or bnode.comb.is_punc_right:
                return core_left, left_right_adjs+right_right_adjs+[(bnode.comb, core_right)]
            elif bnode.comb.is_punc_left:
                return core_right, left_right_adjs+right_right_adjs+[(bnode.comb, core_left)]
            else:
                return bnode.comb.t(core_left, core_right), left_right_adjs+right_right_adjs
        elif node.is_unary:
            unode = node
            core_child, right_adjs = self._extract_right_adjunct_and_punc_subtrees(unode.child)
            return unode.comb.t(core_child), right_adjs
        else:
            return node, []

    cdef Node _add_revealing_rec(self, Node node):
        cdef Unary unode
        cdef Binary bnode
        cdef BinaryCombinator c
        if node.is_term:
            return node
        elif node.is_unary:
            unode = node
            return unode.comb.t(self._add_revealing_rec(unode.child))
        else:
            bnode = node
            l = self._add_revealing_rec(bnode.left)
            r = self._add_revealing_rec(bnode.right)
            if bnode.comb.is_right_adj_comb(bnode.left.cat, bnode.right.cat):
                c = RightAdjoin(span=bnode.left.span, cat=bnode.left.cat)
            else:
                c = bnode.comb
            return c.t(l, r)

    cdef inline Node _reattach_punc_to_left(self, Node root):
        return self.to_left_branching_with_revealing(root, ignore_adjunction=True)

    def to_left_branching_with_revealing(self,
                                         Node root,
                                         bint with_left_branching = True,
                                         bint ignore_adjunction = False):
        cdef Node core_root, left_corner, n
        cdef BinaryCombinator c
        cdef list right_adjs_and_punc
        cdef tuple x

        core_root = self._to_left_branching_simplistic(root)
        if not ignore_adjunction:
            core_root = self._reattach_punc_to_left(core_root)
            core_root = self._add_revealing_rec(core_root)
        core_root, right_adjs_and_punc = self._extract_right_adjunct_and_punc_subtrees(core_root)
        if with_left_branching:
            core_root = self._to_left_branching_simplistic(core_root)
            right_adjs_and_punc = [(c, self._to_left_branching_simplistic(n)) for c, n in right_adjs_and_punc]
        right_adjs_and_punc = sorted(right_adjs_and_punc, key=lambda x: x[1].span)

        if right_adjs_and_punc and \
           right_adjs_and_punc[0][0].is_punc and \
           right_adjs_and_punc[0][1].span[0]==root.span[0]:
            left_corner = right_adjs_and_punc[0][1]
            for c, n in right_adjs_and_punc:
                if c.is_punc and left_corner.span[1] == n.span[0]:
                    left_corner = Punc(True).t(left_corner, n)
                else:
                    break
            right_adjs_and_punc = [(c, n) for c, n in right_adjs_and_punc if left_corner.span[1]<n.span[1]]
            left_corner = self._attach_left_punc_at_bottom(left_corner, core_root)
        else:
            left_corner = core_root

        for c, n in right_adjs_and_punc:
            left_corner = self._reattach_adjunct_or_punc(left_corner, c, n)
        assert left_corner.words() == root.words()
        return left_corner

    cdef Node _reattach_adjunct_or_punc(self, Node core, BinaryCombinator comb, Node adj):
        cdef Unary ucore
        cdef Binary bcore
        if core.span[1] == adj.span[0]:
            if comb.is_punc:
                return Punc(False).t(core, adj)
            else:
                return comb.t(core, adj)
        else:
            if core.is_binary:
                bcore = core
                if bcore.right.span[0]<adj.span[0]:
                    return bcore.comb.t(core.left, self._reattach_adjunct_or_punc(bcore.right, comb, adj))
                else:
                    return bcore.comb.t(self._reattach_adjunct_or_punc(bcore.left, comb, adj), bcore.right)
            elif core.is_unary:
                ucore = core
                return ucore.comb.t(self._reattach_adjunct_or_punc(ucore.child, comb, adj))
            else:
                raise Exception("should not be here")

    cdef Node _right_modify(self, Node left, Node right, tuple span, Category cat):
        # this is used for inserting adjunction (right) into the core tree (left) potentially in the middle of it
        cdef Binary bleft
        cdef Unary uleft
        cdef BinaryCombinator b, c = None
        if left.span == span:
            for b in self.all_backward_B_variations:
                if b.is_right_adj_comb(left.cat, right.cat):
                    c = b
                    break
            if c and left.cat.equals(cat):
                return self.sink_rightward(c.t(left, right))
            elif left.is_binary and left.comb.is_punc_left:
                bleft = left
                return bleft.comb.t(bleft.left, self._right_modify(bleft.right, right, bleft.right.span, cat))
            elif left.is_unary:
                uleft = left
                return uleft.comb.t(self._right_modify(uleft.child, right, span, cat))
            else:
                raise Exception("failed to find the node for right adjunction")
        else:
            if left.is_unary:
                uleft = left
                return uleft.comb.t(self._right_modify(uleft.child, right, span, cat))
            elif left.is_binary:
                bleft = left
                return bleft.comb.t(bleft.left, self._right_modify(bleft.right, right, span, cat))
            else:
                raise Exception("failed to find the node for right adjunction")

    cdef Node _attach_left_punc_at_bottom(self, Node punc, Node node):
        cdef Unary unode
        cdef Binary bnode
        if node.is_term:
            return Punc(True).t(punc, node)
        elif node.is_binary:
            bnode = node
            return bnode.comb.t(self._attach_left_punc_at_bottom(punc, bnode.left), bnode.right)
        else:
            unode = node
            return unode.comb.t(self._attach_left_punc_at_bottom(punc, unode.child))

    cdef Node _attach_right_punc_at_bottom(self, Node node, Node punc):
        cdef Unary unode
        cdef Binary bnode
        if node.is_term:
            return Punc(False).t(node, punc)
        elif node.is_binary:
            bnode = node
            return bnode.comb.t(bnode.left, self._attach_right_punc_at_bottom(bnode.right, punc))
        else:
            unode = node
            return unode.comb.t(self._attach_right_punc_at_bottom(unode.child, punc))

    cdef inline TypeRaising _type_raise_lookup(self, bint is_fwd, Category cat_from, Category cat_result):
        return self.type_raising_table.get((is_fwd, cat_from, cat_result.strip_features()), None)

    cpdef Node to_left_branching(self, Node tree):
        cdef Node node = self._to_left_branching_simplistic(tree)
        node = self._reattach_punc_to_left(node)
        assert node.words() == tree.words(), "rebranching destroyed the tree"
        return node

    cdef Node _to_left_branching_simplistic(self, Node node):
        cdef Node out_node
        cdef Binary bnode
        cdef Unary unode
        if node.is_binary:
            bnode = node
            out_node = self.sink_leftward(bnode.comb.t(
                self._to_left_branching_simplistic(bnode.left),
                self._to_left_branching_simplistic(bnode.right)))
        elif node.is_unary:
            unode = node
            out_node = self.sink_leftward(
                                unode.comb.t(
                                    self._to_left_branching_simplistic(
                                        unode.child)))
        else:
            out_node = node
        return out_node

    cpdef Node to_right_branching(self, Node node):
        cdef Node tree = self._to_right_branching(node)
        assert node.words() == tree.words(), "rebranching destroyed the tree"
        return tree

    cdef Node _to_right_branching(self, Node node):
        cdef Binary bnode
        cdef Unary unode
        if node.is_binary:
            bnode = node
            return self.sink_rightward(bnode.comb.t(self._to_right_branching(bnode.left), self._to_right_branching(bnode.right)))
        elif node.is_unary:
            unode = node
            return self.sink_rightward(unode.comb.t(self._to_right_branching(unode.child)))
        else:
            return node

    cdef inline Node _remove_unneeded_type_raising_single(self, Node node):
        cdef Binary bnode
        cdef Unary  uleft, uright
        if self.is_extreme and node.is_binary:
            bnode = node
            if bnode.comb.is_B0_bck and bnode.right.is_unary:
                uright = bnode.right
                c = uright.comb
                if c.is_type_raise and (<TypeRaising> c).is_order_preserving:
                    return f_app.t(bnode.left, uright.child)
            elif bnode.comb.is_B0_fwd and bnode.left.is_unary:
                uleft = bnode.left
                c = uleft.comb
                if c.is_type_raise and (<TypeRaising> c).is_order_preserving:
                    return b_app.t(uleft.child, bnode.right)
        return node

    cdef bint _is_b0_node_that_has_tr_child(self, Node node):
        cdef Binary bnode
        if node.is_binary:
            bnode = node
            if bnode.comb.is_B0_fwd and bnode.left.is_unary and (<Unary> bnode.left).comb.is_type_raise:
                return True
            if bnode.comb.is_B0_bck and bnode.right.is_unary and (<Unary> bnode.right).comb.is_type_raise:
                return True
        return False

    cdef Node _remove_unneeded_type_raising_for_rebalancing(self, Node node):
        cdef Binary bnode
        if node.is_binary:
            bnode = node
            if self._is_b0_node_that_has_tr_child(bnode.left) or self._is_b0_node_that_has_tr_child(bnode.right):
                return bnode.comb.t(self._remove_unneeded_type_raising_single(bnode.left),
                                    self._remove_unneeded_type_raising_single(bnode.right))
        return node

    cpdef Node sink_rightward(self, Node node):
        node = self._remove_unneeded_type_raising_for_rebalancing(node)
        cdef Node   tree, x, y, z
        cdef Binary bnode, bleft
        cdef Unary  unode
        cdef UnaryCombinator unary_comb, tr, tr1, tr2
        cdef BinaryCombinator punc_comb, c_top, c_left, a_top, a_right
        cdef int o1, o2, new_order
        cdef tuple span
        if node.is_binary:
            bnode = node
            c_top = bnode.comb
            if c_top.is_special_right_adj:
                span = (<RightAdjoin>c_top).span
                cat_to_modify = (<RightAdjoin>c_top).cat
                return self._right_modify(bnode.left, bnode.right, span, cat_to_modify)
            elif c_top.is_punc_right:
                return self._attach_right_punc_at_bottom(bnode.left, bnode.right)
            elif c_top.is_punc_left or c_top.is_glue:
                return c_top.t(bnode.left, self.sink_rightward(bnode.right))
            elif bnode.left.is_binary:
                bleft = bnode.left
                c_left = bleft.comb
                x = bleft.left
                y = bleft.right
                z = bnode.right
                if self.is_extreme:
                    if c_top.is_B0_fwd and c_left.is_B0_bck:
                        # rule 5
                        return self._sink_rightward_rule_5(bnode)
                    elif c_top.is_B_bck and c_left.is_B0_fwd and (<B>c_top).order<self.max_bck_B:
                        # rule 4
                        return self._sink_rightward_rule_4(bnode)
                    elif c_top.is_B0_fwd and c_left.is_B0_fwd:
                        # rule 3
                        return self._sink_rightward_rule_3(bnode)
                    elif c_top.is_B_bck and c_left.is_B_bck:
                        # rule 2
                        return self._sink_rightward_rule_2(bnode)
                if c_top.is_B_fwd and c_left.is_B_fwd:
                    # rule 1
                    return self._sink_rightward_rule_1(bnode)
                elif c_left.is_punc_left:
                    return c_left.t(x, self.sink_rightward(c_top.t(y, z)))
                elif c_top.is_glue and c_left.is_glue:
                    return c_top.t(x, self.sink_rightward(c_top.t(y, z)))
        if node.is_unary:
            unode = node
            if unode.child.is_binary and (<Binary>unode.child).comb.is_punc_left:
                unary_comb = unode.comb
                punc_comb = unode.child.comb
                punc = unode.child.left
                rest = unode.child.right
                return punc_comb.t(punc, unary_comb.t(rest))
        return node

    cdef inline Node _sink_rightward_rule_5(self, Binary node):
        cdef TypeRaising tr
        cdef Node tree
        cdef Binary left = node.left
        cdef Node x = left.left
        cdef Node y = left.right
        cdef Node z = node.right
        tr = self._type_raise_lookup(False, z.cat, node.cat)
        tree = _bsafe(b_app, x, _bsafe(b_app, y, _usafe(tr, z)))
        if tree is not None and tree.cat.equals(node.cat):
            return tree
        return node

    cdef inline Node _sink_rightward_rule_4(self, Binary node):
        cdef TypeRaising tr
        cdef Node tree
        cdef Binary left = node.left
        cdef Node x = left.left
        cdef Node y = left.right
        cdef Node z = node.right
        cdef B c_top = node.comb
        cdef B a_right = B(is_forward=False, is_crossed=False, order=c_top.order+1)
        tr = self._type_raise_lookup(False, z.cat, node.cat)
        tree = _bsafe(b_app, x, _bsafe(a_right, _usafe(tr, y), z))
        if tree is not None and tree.cat.equals(node.cat):
            return tree
        return node

    cdef inline Node _sink_rightward_rule_3(self, Binary node):
        cdef TypeRaising tr1, tr2
        cdef Node tree
        cdef Binary left = node.left
        cdef Node x = left.left
        cdef Node y = left.right
        cdef Node z = node.right
        tr1 = self._type_raise_lookup(False, y.cat, left.cat)
        tr2 = self._type_raise_lookup(False, z.cat, node.cat)
        tree =_bsafe(b_app, x, _bsafe(b_comp, _usafe(tr1, y), _usafe(tr2, z)))
        if tree is not None and tree.cat.equals(node.cat):
            return tree
        return node

    cdef inline Node _sink_rightward_rule_2(self, Binary node):
        cdef Binary left = node.left
        cdef Node x = left.left
        cdef Node y = left.right
        cdef Node z = node.right
        cdef B c_top = node.comb
        cdef B c_left = left.comb
        cdef B a_right, a_top = c_left
        cdef int o1 = c_top.order
        cdef int o2 = c_left.order
        cdef int new_order = o1-o2+1
        cdef Node tree
        if o1>=o2 and new_order<=self.max_bck_B:
            a_right = B(is_forward=False, is_crossed=False, order=new_order)
            if not a_right.can_apply(y.cat, z.cat):
                a_right = B(is_forward=False, is_crossed=True , order=new_order)
                if not a_right.can_apply(y.cat, z.cat):
                    return node
            tree = _bsafe(a_top, x, self.sink_rightward(a_right.t(y, z)))
            if tree is not None:
                return tree
        return node

    cdef inline Node _sink_rightward_rule_1(self, Binary node):
        cdef Binary left = node.left
        cdef Node x = left.left
        cdef Node y = left.right
        cdef Node z = node.right
        cdef B c_top = node.comb
        cdef B c_left = left.comb
        cdef B a_top, a_right
        cdef int new_order = c_top.order + c_left.order - 1
        if c_left.order!=0 and new_order<=self.max_fwd_B:
            a_top = <B> b_fwd_harmonic_lookup[new_order]
            # a_top = B(is_forward=True, is_crossed=False, order=new_order)
            a_right = c_top
            return self.sink_rightward(a_top.t(x, self.sink_rightward((a_right.t(y, z)))))
        return node

    cpdef Node sink_leftward(self, Node node):
        node = self._remove_unneeded_type_raising_for_rebalancing(node)
        cdef Binary bnode
        if node.is_binary:
            bnode = node
            c_top = bnode.comb
            if bnode.right.is_binary:
                # right = bnode.right
                right = (<Binary> bnode.right)
                c_right = right.comb
                # sink = self.sink_leftward
                x = bnode.left
                y = right.left
                z = right.right
                if c_top.is_B0_bck and c_right.is_B0_fwd:
                    # rule 5
                    return self._sink_leftward_rule_5(bnode)
                elif self.is_extreme and c_top.is_B_fwd and c_right.is_B0_bck:
                    # rule 4
                    return self._sink_leftward_rule_4(bnode)
                elif self.is_extreme and c_top.is_B0_bck and c_right.is_B0_bck:
                    # rule 3
                    return self._sink_leftward_rule_3(bnode)
                elif self.is_extreme and c_top.is_B_bck and c_right.is_B_bck:
                    # rule 2
                    return self._sink_leftward_rule_2(bnode)
                elif c_top.is_B_fwd:
                    # rule 1
                    return self._sink_leftward_rule_1(bnode)
                elif c_right.is_punc_right:
                    return c_right.t(self.sink_leftward(c_top.t(x, y)), z)
            if c_top.is_punc_left:
                return self._attach_left_punc_at_bottom(bnode.left, bnode.right)
        return node

    cdef inline Node _sink_leftward_rule_5(self, Binary node):
        cdef UnaryCombinator tr
        cdef Node tree
        cdef Node x = node.left
        cdef Node y = (<Binary> node.right).left
        cdef Node z = (<Binary> node.right).right
        tr = self._type_raise_lookup(True, x.cat, node.cat)
        tree = _bsafe(f_app, _bsafe(f_comp, _usafe(tr, x), y), z)
        if tree is not None and tree.cat.equals(node.cat):
            return tree
        return node

    cdef inline Node _sink_leftward_rule_4(self, Binary node):
        cdef int new_order = (<B> node.comb).order+1
        cdef UnaryCombinator tr
        cdef Node tree
        cdef Node x = node.left
        cdef Node y = (<Binary> node.right).left
        cdef Node z = (<Binary> node.right).right
        if new_order>self.max_fwd_B:
            return node
        cdef B a_left = <B> b_fwd_harmonic_lookup[new_order]
        # cdef B a_left = B(is_forward=True, is_crossed=False, order=new_order)
        tr = self._type_raise_lookup(True, y.cat, node.right.cat)
        tree = _bsafe(f_app, _bsafe(a_left, x, _usafe(tr, y)), z)
        if tree is not None and tree.cat.equals(node.cat):
            return tree
        return node

    cdef inline Node _sink_leftward_rule_3(self, Binary node):
        cdef UnaryCombinator tr1, tr2
        cdef Node tree
        cdef Node x = node.left
        cdef Node y = (<Binary> node.right).left
        cdef Node z = (<Binary> node.right).right
        tr1 = self._type_raise_lookup(True, x.cat, node.cat)
        tr2 = self._type_raise_lookup(True, y.cat, node.right.cat)
        tree = _bsafe(f_app, _bsafe(f_comp, _usafe(tr1, x), _usafe(tr2, y)), z)
        if tree is not None and tree.cat.equals(node.cat):
            return tree
        return node

    cdef inline Node _sink_leftward_rule_2(self, Binary node):
        cdef B c_top = node.comb
        cdef B c_right = node.right.comb
        cdef int new_order = c_top.order+c_right.order-1
        cdef bint new_is_crossed = c_top.is_crossed and c_right.is_crossed and new_order!=0
        if c_right.order==0 or new_order>self.max_fwd_B:
            return node
        cdef B a_top = B(is_forward=False, is_crossed=new_is_crossed, order=new_order)
        # a_left = B(is_forward=False, is_crossed=a_top.is_crossed, order=c_top.order)
        cdef Node tree
        cdef Node x = node.left
        cdef Node y = (<Binary> node.right).left
        cdef Node z = (<Binary> node.right).right
        tree = _bsafe(a_top, self.sink_leftward(c_top.t(x, y)), z)
        if tree is not None and tree.cat.equals(node.cat):
            return tree
        return node

    cdef inline Node _sink_leftward_rule_1(self, Binary node):
        cdef Node tree = self._sink_leftward_rule_1_rebuild(node.left, (<B> node.comb).order, node.right)
        if tree is not None:
            return tree
        else:
            return node

    cdef Node _sink_leftward_rule_1_rebuild(self, Node core_left, int parent_order, Node node):
        cdef Binary bnode
        cdef Node l, tree
        cdef int new_order
        cdef B comb, a2
        if node.is_binary:
            bnode = node
            if bnode.comb.is_B_fwd:
                comb = bnode.comb
                if parent_order>=comb.order:
                    if comb.is_B0_fwd and (<Functor> bnode.left.cat).res.is_np:
                        return None
                    else:
                        new_order = parent_order-comb.order+1
                        l = self._sink_leftward_rule_1_rebuild(core_left, new_order, bnode.left)
                        if l:
                            return bnode.comb.t(l, bnode.right)
                        elif new_order<=self.max_fwd_B:
                            a1 = comb
                            a2 = <B> b_fwd_harmonic_lookup[new_order]
                            # a2 = B(is_forward=True, is_crossed=False, order=new_order)
                            tree = _bsafe(a1, _bsafe(a2, core_left, bnode.left), bnode.right)
                            if tree is not None:
                                return tree
                            return node
        return None
