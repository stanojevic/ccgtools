
from .combinators import *
from .categories import Category


class Grammar:

    def __init__(self, max_fwd_B = 2, max_bck_B = 1):
        self.max_fwd_B = max_fwd_B
        self.max_bck_B = max_bck_B
        self.unary_normal = [
            TypeRaising(  # for topicalization, not used by EasyCCG, but used by C&C and Chinese parse
                cat_arg = Category.from_str("NP"),
                cat_res = Category.from_str("S"),
                is_forward=True,
                is_order_preserving=False),
            TypeRaising(
                cat_arg = Category.from_str("NP"),
                cat_res = Category.from_str("S"),
                is_forward=True,
                is_order_preserving=True),
            TypeRaising(   # my type raising -- indirect object
                cat_arg = Category.from_str("NP"),
                cat_res = Category.from_str("S\\NP"),
                is_forward=False,
                is_order_preserving=True),
            TypeRaising(   # my type raising  -- direct object
                cat_arg = Category.from_str("NP"),
                cat_res = Category.from_str("(S\\NP)/NP"),
                is_forward=False,
                is_order_preserving=True),
            TypeRaising(   # my type raising  -- direct object
                cat_arg = Category.from_str("PP"),
                cat_res = Category.from_str("(S\\NP)/NP"),
                is_forward=False,
                is_order_preserving=True),
            TypeRaising(   # used by EasyCCG
                cat_arg = Category.from_str("NP"),
                cat_res = Category.from_str("S\\NP"),
                is_forward=True,
                is_order_preserving=False),
            TypeRaising(   # used by EasyCCG
                cat_arg = Category.from_str("PP"),
                cat_res = Category.from_str("S\\NP"),
                is_forward=True,
                is_order_preserving=False),
            TypeChanging1(   # used by EasyCCG
                cat_from = Category.from_str("S[pss]\\NP"),
                cat_to   = Category.from_str("NP\\NP"),
            ),
            TypeChanging1(   # used by EasyCCG
                cat_from=Category.from_str("S[ng]\\NP"),
                cat_to=Category.from_str("NP\\NP"),
            ),
            TypeChanging1(   # used by EasyCCG
                cat_from=Category.from_str("S[adj]\\NP"),
                cat_to=Category.from_str("NP\\NP"),
            ),
            TypeChanging1(   # used by EasyCCG
                cat_from=Category.from_str("S[to]\\NP"),
                cat_to=Category.from_str("NP\\NP"),
            ),
            TypeChanging1(   # used by EasyCCG
                cat_from=Category.from_str("S[to]\\NP"),
                cat_to=Category.from_str("N\\N"),
            ),
            TypeChanging1(   # used by EasyCCG
                cat_from=Category.from_str("S[dcl]\\NP"),
                cat_to=Category.from_str("NP\\NP"),
            ),
            TypeChanging1(   # used by EasyCCG
                cat_from=Category.from_str("S[pss]\\NP"),
                cat_to=Category.from_str("S/S"),
            ),
            TypeChanging1(   # used by EasyCCG
                cat_from=Category.from_str("S[ng]\\NP"),
                cat_to=Category.from_str("S/S"),
            ),
            TypeChanging1(   # used by EasyCCG
                cat_from=Category.from_str("S[to]\\NP"),
                cat_to=Category.from_str("S/S"),
            ),
            TypeChanging1(  # used by EasyCCG
                cat_from=Category.from_str("N"),
                cat_to=Category.from_str("NP"),
            ),
        ]
        self.binary_normal = [
            B(is_forward=True , order=0, is_crossed=False),
            B(is_forward=True , order=1, is_crossed=False),
            B(is_forward=True , order=2, is_crossed=False),
            B(is_forward=False, order=0, is_crossed=False),
            B(is_forward=False, order=1, is_crossed=False),
            B(is_forward=False, order=1, is_crossed=True ),
            Conj(is_bottom=False),
            Conj(is_bottom=True),
            # S(is_forward=False, is_crossed=True),
            Punc(punc_is_left=True),
            Punc(punc_is_left=False),
        ]
        self.all_backward_B_variations = [b for b in self.binary_normal if b.is_B_bck] + [Conj(is_bottom=False)]
        self.binary_tc = dict()
        self.unary_tc = dict()
        self.type_raisings = [tr for tr in self.unary_normal if tr.is_type_raise]

    def lookup_binary(self, left: Category, right: Category):
        options = []
        for comb in self.binary_tc.get((left, right), []):
            options.append(comb)
        for comb in self.binary_normal:
            if comb.can_apply(left, right):
                options.append(comb)
        return options

    def lookup_unary(self, child: Category):
        options = []
        for comb in self.unary_tc.get(child, []):
            options.append(comb)
        for comb in self.type_raisings:
            if comb.can_apply(child):
                options.append(comb)
        return options

    def add_unary_tc(self, comb: TypeChanging1):
        key = comb.cat_from
        if key not in self.unary_tc:
            self.unary_tc[key] = set()
        self.unary_tc[key].add(comb)

    def add_binary_tc(self, comb: TypeChanging2):
        key = (comb.left, comb.right)
        if key not in self.binary_tc:
            self.binary_tc[key] = set()
        self.binary_tc[key].add(comb)

    def recognize_unary_comb(self, cat_child: Category, cat_parent: Category) -> UnaryCombinator:
        candidates = [c for c in self.unary_normal if c.can_apply(cat_child) and c(cat_child) == cat_parent]
        if candidates:
            assert len(candidates) == 1
            return candidates[0]
        else:
            return TypeChanging1(cat_from=cat_child, cat_to=cat_parent)

    def recognize_binary_comb(self, left: Category, right: Category, parent: Category) -> BinaryCombinator:
        for c in self.binary_normal:
            if c.can_apply(left, right) and c(left, right) == parent:
                return c
        return TypeChanging2(left=left, right=right, parent=parent)
