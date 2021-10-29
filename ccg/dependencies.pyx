# cython: boundscheck=False

from .visualization import DepsDesc
from .combinators cimport *
from .categories cimport *
from .derivation cimport *

cdef dict _MEMO_ENGLISH_PRED_ARG_MAP = None

cdef dict _MEMO_CHINESE_PRED_ARG_MAP = None

cdef Category _special_cat_1 = Category.from_str("(S[to]\\NP)/(S[b]\\NP)")
cdef Category _special_cat_2 = Category.from_str("S[wq]/(S\\NP)")           # wh type 1
cdef Category _special_cat_3 = Category.from_str("S[wq]/(S/NP)")            # wh type 1
cdef Category _special_cat_4 = Category.from_str("S[wq]")
cdef Category _special_cat_5 = Category.from_str("NP/(S[dcl]\\NP)")         # wh type 2
cdef Category _special_cat_6 = Category.from_str("(S[wq]/(S[dcl]\\NP))/N")  # wh type 3
cdef Category _special_cat_7 = Category.from_str("(S[wq]/(S[q]/NP))/N")     # wh type 3
cdef Category _special_cat_8 = Category.from_str("(S[wq]/S[q])/(S[adj]\\NP)")       # wh type 4
cdef Category _special_cat_9 = Category.from_str("(S[wq]/S[q])/(S[wq]/(S[q]/NP))")  # wh type 5
cdef Category _special_cat_10 = Category.from_str("(S[wq]/(S[q]/(S[adj]\\NP)))/(S[adj]\\NP)") # wh type 6
cdef Category _special_cat_11 = Category.from_str("(S[wq]/(S[dcl]\\NP))/NP")  # wh type 3
cdef Category _special_cat_12 = Category.from_str("(S[wq]/(S[dcl]/NP))/NP")   # wh type 3

# S[wq]/(S\NP) and S[wq]/(S/NP)
cdef inline bint is_special_wh_type_1(Category cat):
    cdef bint a = cat.equals_featureless(_special_cat_2)
    cdef bint b = cat.equals_featureless(_special_cat_3)
    if a or b:
        return (<Functor> cat).res.equals(_special_cat_4)
    else:
        return False

# (S[wq]/(S[dcl]\NP))/N and (S[wq]/(S[q]/NP))/N
cdef inline bint is_special_wh_type_3(Category cat):
    cdef bint a = cat.equals_featureless(_special_cat_6)
    cdef bint b = cat.equals_featureless(_special_cat_7)
    cdef bint c = cat.equals_featureless(_special_cat_11)
    cdef bint d = cat.equals_featureless(_special_cat_12)
    if a or b or c or d:
        return (<Functor> (<Functor> cat).res).res.equals(_special_cat_4)
    else:
        return False

# NP/(S[dcl]\NP)
cdef inline bint is_special_wh_type_2(Category cat, str word):
    return cat.equals(_special_cat_5) and is_wh_word(word)

# (S[wq]/S[q])/(S[adj]\\NP)
cdef inline bint is_special_wh_type_4(Category cat):
    return cat.equals(_special_cat_8)

# (S[wq]/S[q])/(S[wq]/(S[q]/NP))
cdef inline bint is_special_wh_type_5(Category cat):
    return cat.equals(_special_cat_9)

# (S[wq]/(S[q]/(S[adj]\\NP)))/(S[adj]\\NP)
cdef inline bint is_special_wh_type_6(Category cat):
    return cat.equals(_special_cat_10)

cdef inline bint is_wh_word(str word):
    cdef str w = word.lower()
    return (w == "why") or (w == "who") or (w == "which") or \
           (w == "what") or (w == "where") or (w == "when") or (w == "how")

cdef inline dict _chinese_predarg_table():
    global _MEMO_CHINESE_PRED_ARG_MAP
    cdef dict res
    if _MEMO_CHINESE_PRED_ARG_MAP is None:
        res = dict()
        res[("被", Category.from_str("(S[dcl]\\NP)/((S[dcl]\\NP)/NP)"))] = Formula.from_str(
            "(S[dcl]\\NP_2)/((S[dcl]\\NP_2:B)/NP)_1")
        res[("被", Category.from_str("(S[dcl]\\NP)/((S[dcl]\\NP)/NP)"))] = Formula.from_str(
            "(S[dcl]\\NP_2)/((S[dcl]\\NP)/NP_2:B)_1")
        res[("将", Category.from_str("(S\\NP)/(S\\NP)"))] = Formula.from_str("(S\\NP_2)/(S\\NP_2:B)_1")
        res[("的", Category.from_str("(NP/NP)\\(S[dcl]\\NP)"))] = Formula.from_str("(NP/NP_2)\\(S[dcl]\\NP_2:B)_1")
        res[("了", Category.from_str("(S\\NP)\\(S\\NP)"))] = Formula.from_str("(S\\NP_2)\\(S\\NP_2:B)_1")
        res[("是", Category.from_str("(S[dcl]\\NP)/(S[dcl]\\NP)"))] = Formula.from_str(
            "(S[dcl]\\NP_2)/(S[dcl]\\NP_2:B)_1")
        res[("完", Category.from_str("(S\\NP)\\(S\\NP)"))] = Formula.from_str("(S\\NP_2)\\(S\\NP_2:B)_1")
        _MEMO_CHINESE_PRED_ARG_MAP = res
    return _MEMO_CHINESE_PRED_ARG_MAP


cdef inline dict _english_predarg_table():
    global _MEMO_ENGLISH_PRED_ARG_MAP
    cdef dict res
    cdef Formula formula
    if _MEMO_ENGLISH_PRED_ARG_MAP is None:
        from os.path import realpath, dirname, join
        from os import getcwd
        res = dict()
        file_loc = join(realpath(join(getcwd(), dirname(__file__))), "dependencies_mapping_english.txt")
        with open(file_loc) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                fields = line.split("\t")
                cat = Category.from_str(fields[0])

                formula = Formula.from_str(fields[1])
                replacements = {v: Var(pd_id=i) for i, v in enumerate(sorted(formula.vars, key=lambda v: v.pd_id))}
                formula = formula.replace_vars(replacements)

                res[cat] = formula
        _MEMO_ENGLISH_PRED_ARG_MAP = res
    return _MEMO_ENGLISH_PRED_ARG_MAP


cdef class PredArgAssigner:

    cdef readonly str lang
    cdef readonly bint is_English
    cdef readonly bint include_conj_term

    def __cinit__(self, str lang, bint include_conj_term = False):
        self.lang = lang
        # self.is_English = lang.lower().startswith("english")  # TODO Chinese currently doesn't work well
        self.is_English = True
        self.include_conj_term = include_conj_term

    def all_deps(self, Node tree):
        cdef set deps = set()
        cdef Node n
        for n in tree:
            n.predarg = None
        self.for_node(tree)
        for n in tree:
            deps.update(n.predarg.new_finished_deps)
        return deps

    def show_all_deps(self, node):
        return self._to_deps_desc(node)

    cdef object _to_deps_desc(self, Node node):
        deps_list = [(d.head_pos, d.dep_pos, d.label, "dashed" if d.is_adj else "solid") for d in self.all_deps(node)]
        return DepsDesc(words=node.words(), deps=deps_list, starting_position=node.span[0])

    def all_deps_visualize(self,
                           Node node,
                           str graph_label = "dependencies",
                           str renderer = "dot",
                           bint include_disconnected_words = False,
                           bint include_word_positions = False):
        self._to_deps_desc(node).visualize(graph_label=graph_label,
                                           renderer=renderer,
                                           include_disconnected_words=include_disconnected_words,
                                           include_word_positions=include_word_positions)

    def all_deps_save(self,
                      Node node,
                      str fn,
                      str renderer = "dot",
                      bint include_disconnected_words = False,
                      bint include_word_positions = False):
        self._to_deps_desc(node).save(fn,
                                      renderer=renderer,
                                      include_disconnected_words=include_disconnected_words,
                                      include_word_positions=include_word_positions)

    cpdef PredArg for_node(self, Node node):
        cdef PredArg predarg
        if node.predarg:
            predarg = node.predarg
        elif node.is_term:
            predarg = self._process_terminal_node(<Terminal> node)
        elif node.is_unary:
            self.for_node((<Unary> node).child)
            predarg = self._process_unary_node(<Unary> node)
        else:
            self.for_node((<Binary> node).left)
            self.for_node((<Binary> node).right)
            predarg = self._process_binary_node(<Binary> node)
        node.predarg = predarg
        return predarg

    @staticmethod
    cdef inline PredArg _alpha_transform_right(PredArg lparg, PredArg rparg):
        return rparg._renew_min_var(lparg.max_var_id()+1)

    @staticmethod
    cdef PredArg _trivial_combine(PredArg main_parg, PredArg non_main_parg, PredArgState state, bint with_alpha):
        if with_alpha:
            non_main_parg = PredArgAssigner._alpha_transform_right(main_parg, non_main_parg)
        return PredArg(
            formula=main_parg.formula,
            head_word=main_parg.head_word,
            main_word=main_parg.main_word,
            state=state,
            unfinished_deps=main_parg.unfinished_deps.union(non_main_parg.unfinished_deps),
            new_finished_deps=set()
        )

    @staticmethod
    cdef PredArg _coord_combine(PredArg l_parg, PredArg r_parg, Category parent_cat):
        # l_parg = PredArgAssigner._trivial_combine(l_parg, r_parg.remove_unfinished(), NormalState(), False)
        r_parg = PredArgAssigner._alpha_transform_right(l_parg, r_parg)
        cdef list slashes = [x[0] for x in parent_cat.cut_args()[1]]
        l_parg = PredArgAssigner._mark_unbounded_vars_coord(l_parg, slashes, is_left=True)
        r_parg = PredArgAssigner._mark_unbounded_vars_coord(r_parg, slashes, is_left=False)
        cdef set subs = unify(l_parg.formula, r_parg.formula)
        cdef dict subs_lookup = {x[0]: x[1] for x in subs}
        l_parg = l_parg.replace_vars(subs_lookup)
        cdef PredArg res = PredArgAssigner._trivial_combine(l_parg, r_parg, NormalState(), False)
        cdef Term lh = l_parg.formula.find_head_term()
        cdef Term rh = r_parg.formula.find_head_term()
        if lh.is_words and rh.is_words:
            coord_heads = Words(lh.words | rh.words)
        else:
            coord_heads = lh
        res = res.copy(formula=res.formula.replace_head_term(coord_heads))
        return res.replace_vars(subs_lookup)

    @staticmethod
    cdef PredArg _mark_boundness_dep(PredArg parg, set terms, bint is_bound, bint is_unbound):
        cdef set new_unfinished = set()
        cdef UnfinishedDepLink d
        for d in parg.unfinished_deps:
            if d.dep_words in terms:
                new_unfinished.add( d.change_boundness(is_bound, is_unbound) )
            else:
                new_unfinished.add(d)
        return parg.copy(unfinished_deps=new_unfinished)

    @staticmethod
    cdef PredArg _mark_unbounded_vars_coord(PredArg parg, list slashes, bint is_left):
        args = parg.formula.cut_args()[1]
        cdef set terms = {x[0].head for x in zip(args, slashes) if x[1]==is_left}
        return PredArgAssigner._mark_boundness_dep(parg, terms, is_bound=False, is_unbound=True)

    @staticmethod
    cdef PredArg _weird_coord_combine(PredArg l_parg, PredArg r_parg, Category parent_cat):
        cdef Term l_head_term = l_parg.formula.find_head_term()
        cdef Term r_head_term = r_parg.formula.find_head_term()
        r_parg = r_parg.remove_unfinished()
        if l_head_term.is_words and r_head_term.is_words:
            w = Words(l_head_term.words | r_head_term.words)
            l_parg = PredArgAssigner._trivial_combine(l_parg, r_parg, NormalState(), True)
            f = l_parg.formula.replace_head_term(w)
            return PredArgAssigner._trivial_combine(l_parg, r_parg, NormalState(), True).copy(formula=f)
        else:
            f = Formula.from_cat(parent_cat)
            return PredArgAssigner._trivial_combine(l_parg, r_parg, NormalState(), True).copy(formula=f)

    @staticmethod
    cdef PredArg _normal_combine(PredArg main_parg, PredArg non_main_parg, int order):
        cdef Formula lmatch, rmatch, hleft
        cdef list rargs
        non_main_parg = PredArgAssigner._alpha_transform_right(main_parg, non_main_parg)
        rmatch, rargs = non_main_parg.formula.cut_args(order)
        hleft, [lmatch] = main_parg.formula.cut_args(1)
        cdef set  subs = unify(lmatch, rmatch)
        cdef Term x, y
        cdef dict subs_lookup = {x:y for x, y in subs}
        cdef set lbound, lunbound
        lbound, lunbound = lmatch.all_non_local_terms()
        cdef list subs_reversed = [(y, x) for x, y in subs]
        cdef dict subs_redundant = {x: y for x, y in list(subs) + subs_reversed}
        cdef set  rbound   = {subs_redundant[v] for v in lbound   if v in subs_redundant}
        cdef set  runbound = {subs_redundant[v] for v in lunbound if v in subs_redundant}
        non_main_parg = PredArgAssigner._mark_boundness_dep(non_main_parg, rbound  , is_bound=True, is_unbound=False)
        non_main_parg = PredArgAssigner._mark_boundness_dep(non_main_parg, runbound, is_bound=False, is_unbound=True)
        cdef Formula res_unfinished = hleft
        cdef Formula f
        for f in rargs:
            res_unfinished = PAFunctor(Unavailable(), False, False, res_unfinished, f)
        cdef PredArg parg_unfinished = PredArg(formula=res_unfinished,
                                               head_word=main_parg.head_word,
                                               main_word=main_parg.main_word,
                                               state=NormalState(),
                                               unfinished_deps=(main_parg.unfinished_deps | non_main_parg.unfinished_deps),
                                               new_finished_deps=set())
        return parg_unfinished.replace_vars(subs_lookup)

    cdef PredArg _process_binary_node(self, Binary node):
        cdef PredArg l_parg, r_parg
        cdef Category cat, l_cat, r_cat
        cdef BinaryCombinator c
        cdef PredArg res
        l_parg = node.left.predarg
        r_parg = node.right.predarg
        l_cat  = node.left.cat
        r_cat  = node.right.cat
        c = node.comb
        cat = node.cat

        if c.is_special_right_adj:
            return l_parg.copy(new_finished_deps=set())
        elif c.is_conj_top:
            if r_parg.state.is_coord:
                res = PredArgAssigner._coord_combine(l_parg, r_parg, cat)
                h = l_parg.formula.find_head_term()
                if h.is_words and self.include_conj_term and r_parg.state.conj_word:
                    res = res.add_finished(heads=Words({r_parg.state.conj_word}),
                                           deps=h,
                                           slot=1,
                                           cat=Atomic("conj"),
                                           is_bound=False,
                                           is_unbound=False,
                                           is_adj=True,
                                           is_conj=True)
                return res
            elif r_parg.state.is_weird_coord:
                return PredArgAssigner._weird_coord_combine(l_parg, r_parg, cat)
            else:
                return PredArgAssigner._trivial_combine(l_parg, r_parg.remove_unfinished(), NormalState(), True)
        elif c.is_B_fwd:
            return PredArgAssigner._normal_combine(l_parg, r_parg, c.order)
        elif c.is_B_bck:
            rstate = r_parg.state
            if rstate.is_weird_coord and c.order == 0:
                return PredArgAssigner._weird_coord_combine(l_parg, r_parg, cat)
            elif rstate.is_weird_coord or rstate.is_weird_tc2:
                return PredArgAssigner._trivial_combine(l_parg,
                                                        r_parg.remove_unfinished(),
                                                        NormalState(),
                                                        True).copy(formula=Formula.from_cat(cat))
            else:
                return PredArgAssigner._normal_combine(r_parg, l_parg, c.order)
        elif c.is_conj_bottom:
            if self.include_conj_term and node.left.is_term:
                conj_word = (<Terminal> node.left).word
                conj_pos  = (<Terminal> node.left).pos
                w = Word(conj_pos, conj_word)
                res = PredArgAssigner._trivial_combine(r_parg, l_parg, CoordState(node.right, w), True)
                h = r_parg.formula.find_head_term()
                if h.is_words:
                    res = res.add_finished(heads=Words({w}),
                                           deps=h,
                                           slot=2,
                                           cat=Atomic("conj"),
                                           is_bound=False,
                                           is_unbound=False,
                                           is_adj=False,
                                           is_conj=True)
                return res
            else:
                return PredArgAssigner._trivial_combine(r_parg, l_parg, CoordState(node.right, None), True)
        elif c.is_punc_left or c.is_glue:
            a, b = r_parg, l_parg
            return PredArgAssigner._trivial_combine(a, b, a.state, True)
        elif c.is_punc_right:
            a, b = l_parg, r_parg
            return PredArgAssigner._trivial_combine(a, b, a.state, True)
        elif c.is_tc_A_B_to_B:
            a, b = r_parg, l_parg.remove_unfinished()
            return PredArgAssigner._trivial_combine(a, b, a.state, True)
        elif c.is_tc_A_B_to_A:
            a, b = l_parg, r_parg.remove_unfinished()
            return PredArgAssigner._trivial_combine(a, b, a.state, True)
        elif c.is_tc_XY_A_to_ZY:
            return PredArgAssigner._transform_XY_to_ZY(l_parg, l_cat, cat)
        elif c.is_tc_A_XY_to_ZY:
            return PredArgAssigner._transform_XY_to_ZY(r_parg, r_cat, cat)
        elif c.is_tc_X_Y_to_Yconj:
            a, b, state = r_parg, l_parg, WeirdCoordState(node.right)
            return PredArgAssigner._trivial_combine(a, b, state, True).copy(formula=Formula.from_cat(cat))
        elif c.is_tc_X_Y_to_Xconj:
            a, b, state = l_parg, r_parg, WeirdCoordState(node.left)
            return PredArgAssigner._trivial_combine(a, b, state, True).copy(formula=Formula.from_cat(cat))
        else:
            a, b, state = l_parg, r_parg, WeirdTC2State()
            return PredArgAssigner._trivial_combine(a, b, state, True).copy(formula=Formula.from_cat(cat))

    @staticmethod
    cdef inline PredArg _transform_X_to_XX(PredArg child_parg):
        # it's imperfect but all type changing cases like this are
        # consider replacing this with coordination
        cdef Formula child_f = child_parg.formula
        cdef int base = child_parg.max_var_id()
        cdef Var new_head = Var(base+1)
        cdef Var new_head2 = Var(base+2)
        child_f = child_f.replace_head_term(new_head2)
        child_f = child_f.assign_head_words(new_head)
        cdef PAFunctor res_f = PAFunctor(
            head=child_f.head,
            is_bound=child_f.is_bound,
            is_unbound=child_f.is_unbound,
            left=child_f,
            right=child_f,
        )
        return child_parg.copy(formula=res_f)

    @staticmethod
    cdef inline PredArg _transform_XY_to_ZY(PredArg child_parg, Category cat_from, Category cat_to):
        cdef PAFunctor child_f = child_parg.formula
        cdef Formula r = child_f.right
        cdef PAFunctor new_formula = Formula.from_cat(cat_to)
        cdef UnfinishedDepLink d
        cdef bint is_bound, is_unbound
        new_formula = PAFunctor(head=child_f.head,
                                is_bound=child_f.is_bound,
                                is_unbound=child_f.is_unbound,
                                left=new_formula.left,
                                right=r)  # .assign_head_words(child_f.find_head_term())
        S_fwd_S  = Functor(True, Atomic("S"), Atomic("S"))
        unfinished_deps = child_parg.unfinished_deps
        if cat_from.res.is_s and cat_from.arg.is_np and cat_to.res.is_nominal and cat_to.arg.is_nominal:
            # reduced relative clause
            if cat_from.slash_is_fwd:
                is_bound = False
                is_unbound = True
            else:
                is_bound = True
                is_unbound = False
            unfinished_deps = {d.change_boundness(is_bound=is_bound, is_unbound=is_unbound) for d in unfinished_deps}
        elif cat_to.equals_featureless(S_fwd_S):
            unfinished_deps = {d.change_boundness(is_bound=True, is_unbound=False) for d in unfinished_deps}

        if cat_to.is_adj_cat:
            new_formula = new_formula.replace_head_term(r.head)
            # unfinished_deps = {d.copy(is_adj=True) for d in unfinished_deps if d.dep_words == new_formula.right.head}
        else:
            new_formula = new_formula.replace_head_term(child_f.find_head_term())
        parg = child_parg.copy(formula=new_formula, unfinished_deps=unfinished_deps)
        return parg

    cdef inline PredArg _process_unary_node(self, Unary node):
        cdef UnaryCombinator c = node.comb
        cdef Category cat = node.cat
        cdef Node child = node.child
        cdef PredArg child_parg = child.predarg
        cdef Category child_cat = child.cat
        cdef Formula child_f = child_parg.formula
        if c.is_type_raise:
            new_var = Var(child_parg.max_var_id()+1)
            func_f = Formula.from_cat((<TypeRaising> c).cat_res).replace_head_term(new_var)
            formula = PAFunctor(new_var,
                                False,
                                False,
                                func_f,
                                PAFunctor(new_var,
                                          False,
                                          False,
                                          func_f,
                                          child_f))
            return child_parg.copy(formula=formula, head_word=None)
        elif c.is_unary_coord:
            return child_parg.copy(state=CoordState(child, None), formula=Formula.from_cat(node.cat))
        elif c.is_XY_to_ZY_change:
            return PredArgAssigner._transform_XY_to_ZY(child_parg, child_cat, cat)
        elif c.is_X_to_XX_change:
            return PredArgAssigner._transform_X_to_XX(child_parg)
        else:
            new_formula = Formula.from_cat(node.cat)
            f = new_formula.replace_head_term(child_f.find_head_term())
            return child_parg.copy(formula=f)

    cdef inline _process_terminal_node(self, Terminal node):
        cdef Formula formula = self._lookup_formula(node.pos, node.word, node.cat)
        cdef Formula x
        cdef list ds = [(x.head, x.is_bound, x.is_unbound) for x in formula.cut_args()[1]]
        cdef Word word = Word(node.pos, node.word)
        cdef list unfinished_deps = []
        cdef bint last_is_adj, is_adj
        cdef int i, deps_count
        cdef Term dw
        cdef UnfinishedDepLink d
        deps_count, last_is_adj = PredArgAssigner._count_deps(node.cat, formula)
        for i, (dw, b_bound, b_unbound) in list(reversed(list(enumerate(ds))))[:deps_count]:
            is_adj = (i==len(ds)-deps_count) and last_is_adj
            unfinished_deps.append(
                UnfinishedDepLink(
                    head_cat=node.cat,
                    slot=i+1,
                    head_words=Words({word}),
                    dep_words=dw,
                    is_bound=b_bound,
                    is_unbound=b_unbound,
                    is_adj=is_adj))
        if node.cat.equals(_special_cat_1):
            unfinished_deps = [d for d in unfinished_deps if d.slot!=1]
        elif is_special_wh_type_2(node.cat, node.word):
            # NP/(S[dcl]\NP)
            w = Words({word})
            v = Var(1)
            formula = PAFunctor(w, False, False,
                                PAAtom(w, False, False),
                                PAFunctor(v, False, False, PAAtom(v, False, False), PAAtom(w, False, True)))
            unfinished_deps = []
        elif is_special_wh_type_1(node.cat):
            # S[wq]/(S\NP) and S[wq]/(S/NP)
            # "(<T S[wq] 0 2> (<L S[wq]/(S[q]/NP) POS POS Who S[wq]/(S[q]/NP)>) (<T S[q]/NP 0 2> (<T S[q]/(S[b]\\NP) 0 2> (<L (S[q]/(S[b]\\NP))/NP POS POS did (S[q]/(S[b]\\NP))/NP>) (<T NP 0 1> (<L N POS POS Gerrard N>) ) ) (<T (S[b]\\NP)/NP 0 2> (<T (S[b]\\NP)/NP 0 2> (<L (S[b]\\NP)/PP POS POS admit (S[b]\\NP)/PP>) (<L PP/NP POS POS to PP/NP>) ) (<L (S\\NP)\\(S\\NP) POS POS punching? (S\\NP)\\(S\\NP)>) ) ) )"
            w = Words({word})
            v = Var(1)
            formula = PAFunctor(v, False, False,
                                PAAtom(v, False, False),
                                PAFunctor(v, False, False, PAAtom(v, False, False), PAAtom(w, False, True)))
            unfinished_deps = []
        elif is_special_wh_type_3(node.cat):
            # (S[wq]/(S[dcl]\NP))/N and (S[wq]/(S[q]/NP))/N and (S[wq]/(S[dcl]\NP))/NP and (S[wq]/(S[dcl]/NP))/NP
            # "(<T S[wq] 0 2> (<T S[wq]/(S[q]/NP) 0 2> (<L (S[wq]/(S[q]/NP))/N POS POS which (S[wq]/(S[q]/NP))/N>) (<L N POS POS group N>) ) (<T S[q]/NP 0 2> (<T S[q]/(S[b]\\NP) 0 2> (<L (S[q]/(S[b]\\NP))/NP POS POS did (S[q]/(S[b]\\NP))/NP>) (<L NP POS POS he NP>) ) (<L (S[b]\\NP)/NP POS POS join? (S[b]\\NP)/NP>) ) ) "
            w = Words({word})
            v1 = Var(1)
            v2 = Var(2)
            formula = \
                PAFunctor(v1, False, False,
                          PAFunctor(v1, False, False,
                                    PAAtom(v1, False, False),
                                    PAFunctor(v1, False, False, PAAtom(v1, False, False), PAAtom(w, False, True))),
                          PAAtom(v2, False, False))
            unfinished_deps = [
                UnfinishedDepLink(
                    head_cat=node.cat,
                    slot=2,
                    head_words=w,
                    dep_words=v2,
                    is_bound=False,
                    is_unbound=False,
                    is_adj=False)]
        elif is_special_wh_type_4(node.cat):
            # (S[wq]/S[q])/(S[adj]\\NP)
            w = Words({word})
            v_q = Var(1)
            v_adj = Var(2)
            formula = PAFunctor(v_q, False, False,
                                PAFunctor(v_q, False, False, PAAtom(v_q, False, False), PAAtom(v_q, False, False)),
                                PAFunctor(v_adj, False, False, PAAtom(v_adj, False, False), PAAtom(w, False, True)))
            unfinished_deps = [
                UnfinishedDepLink(
                    head_cat=node.cat,
                    slot=1,
                    head_words=w,
                    dep_words=v_q,
                    is_bound=False,
                    is_unbound=False,
                    is_adj=True)]
        elif is_special_wh_type_6(node.cat):
            # (S[wq]/(S[q]/(S[adj]\\NP)))/(S[adj]\\NP)
            w = Words({word})
            v_np  = Var(1)
            v_adj = Var(2)
            v_q   = Var(3)
            f_adj_np = PAFunctor(v_adj, False, False, PAAtom(v_adj, False, False), PAAtom(v_np, False, True))
            formula = PAFunctor(v_q, False, False,
                                PAFunctor(v_q, False, False, # (S[wq]/(S[q]/(S[adj]\\NP)))
                                          PAAtom(v_q, False, False), # S[wq]
                                          PAFunctor(v_q, False, False,
                                                    PAAtom(v_q, False, False), # S[q]
                                                    f_adj_np)),
                                f_adj_np)
            unfinished_deps = [
                UnfinishedDepLink(
                    head_cat=node.cat,
                    slot=2,
                    head_words=w,
                    dep_words=v_adj,
                    is_bound=False,
                    is_unbound=False,
                    is_adj=False)]
        # elif is_special_wh_type_5(node.cat):
            # "(<T S[wq] 0 2> (<T S[wq]/S[q] 0 2> (<L (S[wq]/S[q])/(S[wq]/(S[q]/NP)) POS POS On (S[wq]/S[q])/(S[wq]/(S[q]/NP))>) (<T S[wq]/(S[q]/NP) 0 2> (<L (S[wq]/(S[q]/NP))/N POS POS what (S[wq]/(S[q]/NP))/N>) (<L N POS POS day N>) ) ) (<T S[q] 0 2> (<T S[q]/NP 0 2> (<L (S[q]/NP)/NP POS POS was (S[q]/NP)/NP>) (<T NP[nb] 0 2> (<L NP[nb]/N POS POS the NP[nb]/N>) (<L N POS POS girl N>) ) ) (<T NP 0 1> (<L N POS POS found? N>) ) ) )"
            # (S[wq]/S[q])/(S[wq]/(S[q]/NP))
            # handled by the new entry in the lexicon rules
            # pass

        return PredArg(
            formula=formula,
            head_word=None if formula.is_bounded_formula() else word,
            main_word=word,
            state=NormalState(),
            unfinished_deps=set(unfinished_deps),
            new_finished_deps=set())

    @staticmethod
    cdef tuple _count_deps(Category cat, Formula f):
        cdef Functor catf
        cdef PAFunctor ff
        if cat.is_functor:
            catf = cat
            ff = f
            if catf.res.equals(catf.arg) and ff.left.equals(ff.right):
                return 1, True  # stop here if cat is an adjunct
            else:
                dep_count, last_is_adj = PredArgAssigner._count_deps(catf.res, ff.left)
                return 1+dep_count, last_is_adj
        else:
            return 0, False

    cdef inline Formula _lookup_formula(self, int pos, str word, Category cat):
        cdef dict table
        if self.is_English:
            table = _english_predarg_table()
            key = cat
        else:
            table = _chinese_predarg_table()
            key = word, cat
        cdef Formula f = table.get(key, None)
        if f is None:
            f = Formula.from_cat(cat).to_indexed_formula()
        return f.assign_head_word(pos, word)


cdef class PredArg:

    cdef readonly Formula formula
    cdef readonly Word head_word
    cdef readonly Word main_word
    cdef readonly PredArgState state
    cdef readonly set unfinished_deps
    cdef readonly set new_finished_deps
    cdef readonly set vars

    def __cinit__(self,
                 Formula formula,
                 Word head_word,
                 Word main_word,
                 PredArgState state,
                 set unfinished_deps,
                 set new_finished_deps
                 ):
        self.formula = formula
        self.head_word = head_word
        self.main_word = main_word
        self.state = state
        self.unfinished_deps = unfinished_deps
        self.new_finished_deps = new_finished_deps

        self.vars = formula.vars.copy()
        cdef UnfinishedDepLink d
        for d in unfinished_deps:
            if d.dep_words.is_var:
                self.vars.add(d.dep_words)

    cdef inline int max_var_id(self):
        cdef int m = 0
        cdef Var i
        for i in self.vars:
            if i.pd_id>m:
                m=i.pd_id
        return m

    cdef inline PredArg remove_unfinished(self):
        return self.copy(unfinished_deps=set())

    cdef inline PredArg add_finished(self,
                              Words heads,
                              Words deps,
                              int slot,
                              Category cat,
                              bint is_bound,
                              bint is_unbound,
                              bint is_adj,
                              bint is_conj):
        cdef set new_deps = set()
        for h in heads.words:
            for d in deps.words:
                new_deps.add(
                    DepLink(head_cat=cat,
                            head_pos=h.word_pos,
                            dep_pos=d.word_pos,
                            dep_slot=slot,
                            head_word=h.word,
                            dep_word=d.word,
                            is_bound=is_bound,
                            is_unbound=is_unbound,
                            is_adj=is_adj,
                            is_conj=is_conj))
        return self.copy(new_finished_deps=self.new_finished_deps.union(new_deps))

    cdef inline _renew_min_var(self, int mmin):
        replacements = {v: Var(pd_id=v.pd_id + mmin) for i, v in enumerate(self.vars)}
        new_formula = (<Formula> self.formula).replace_vars(replacements)
        cdef set new_unfinished_deps = set()
        cdef UnfinishedDepLink d
        for d in self.unfinished_deps:
            new_unfinished_deps.add(d.replace_vars(replacements))
        return self.copy(formula=new_formula,
                         unfinished_deps=new_unfinished_deps)

    cdef inline PredArg replace_vars(self, dict subs_lookup):
        new_f = (<Formula> self.formula).replace_vars(subs_lookup)
        new_finished = []
        new_unfinished = []
        cdef UnfinishedDepLink d
        for d in self.unfinished_deps:
            d = d.replace_vars(subs_lookup)
            if d.dep_words.is_var:
                new_unfinished.append(d)
            else:
                new_finished.extend(d.to_final_deps())
        return self.copy(formula=new_f, unfinished_deps=set(new_unfinished), new_finished_deps=set(new_finished))

    def active_words(self):
        return self._active_word_as_function() | {self.main_word}

    def _active_simple_heads(self):
        x = self.formula.find_head_term()
        if x.is_words:
            return set(x.words)
        else:
            if self.head_word:
                return {self.head_word}
            else:
                return set()

    # less important
    def _active_words_as_args(self, max_order):
        head, args = self.formula.cut_args()
        vvars= set()
        for x in [head] + args[:max_order]:
            vvars.update(x.vars)
        words = set()
        for d in self.unfinished_deps:
            if d.dep_words in vvars:
                words.update(d.head_words.words)
        return words

    # most important
    def _active_word_as_function(self):
        if self.formula.is_atom:
            return set()
        else:
            res = set()
            right = (<PAFunctor> self.formula).right
            for d in self.unfinished_deps:
                if d.dep_words in right.vars:
                    res.update(set(d.head_words.words))
            return res

    def copy(self,
             Formula formula = None,
             Term head_word = None,
             Term main_word = None,
             PredArgState state = None,
             set unfinished_deps = None,
             set new_finished_deps = None
             ):
        if formula is None:
            formula = self.formula
        if head_word is None:
            head_word = self.head_word
        if main_word is None:
            main_word = self.main_word
        if state is None:
            state = self.state
        if unfinished_deps is None:
            unfinished_deps = self.unfinished_deps
        if new_finished_deps is None:
            new_finished_deps = self.new_finished_deps
        return PredArg(
            formula=formula,
            head_word=head_word,
            main_word=main_word,
            state=state,
            unfinished_deps=unfinished_deps,
            new_finished_deps=new_finished_deps)


######################################################
#                  PredArgState                      #
######################################################


cdef class PredArgState:

    cdef readonly bint is_normal
    cdef readonly bint is_coord
    cdef readonly bint is_weird_coord
    cdef readonly bint is_weird_tc2


cdef class NormalState(PredArgState):

    def __cinit__(self):
        self.is_normal = True
        self.is_coord = False
        self.is_weird_coord = False
        self.is_weird_tc2 = False

    def __hash__(self):
        return 34151262

    def __eq__(self, other):
        return isinstance(other, NormalState)


cdef class CoordState(PredArgState):

    cdef readonly Node right
    cdef readonly Word conj_word

    def __cinit__(self, right, conj_word):
        self.is_normal = False
        self.is_coord = True
        self.is_weird_coord = False
        self.is_weird_tc2 = False

        self.right = right
        self.conj_word = conj_word

    def __hash__(self):
        return 16109834109

    def __eq__(self, other):
        if isinstance(other, CoordState):
            return (<CoordState> other).right == self.right and (<CoordState> other).conj_word == self.conj_word
        else:
            return False


cdef class WeirdCoordState(PredArgState):

    cdef readonly Node right

    def __cinit__(self, right):
        self.is_normal = False
        self.is_coord = False
        self.is_weird_coord = True
        self.is_weird_tc2 = False

        self.right = right

    def __hash__(self):
        return 9101484

    def __eq__(self, other):
        return isinstance(other, WeirdCoordState) and (<WeirdCoordState> other).right == self.right


cdef class WeirdTC2State(PredArgState):

    def __cinit__(self):
        self.is_normal = False
        self.is_coord = False
        self.is_weird_coord = False
        self.is_weird_tc2 = True

    def __hash__(self):
        return 18509185

    def __eq__(self, other):
        return isinstance(other, WeirdTC2State)


######################################################
#                          TERM                      #
######################################################


cdef class Term:

    cdef readonly bint is_var
    cdef readonly bint is_unavailable
    cdef readonly bint is_word
    cdef readonly bint is_words

    cdef bint equals(self, Term other):
        raise NotImplementedError()


cdef class Unavailable(Term):

    def __cinit__(self):
        self.is_var = False
        self.is_unavailable = True
        self.is_word = False
        self.is_words = False

    def __str__(self):
        return ""

    def __hash__(self):
        return 3928984

    cdef bint equals(self, Term other):
        return other.is_unavailable

    def __eq__(self, other):
        return isinstance(other, Unavailable) and self.equals(other)


cdef class Var(Term):

    cdef readonly int pd_id

    def __cinit__(self, int pd_id):
        self.pd_id = pd_id
        self.is_var = True
        self.is_unavailable = False
        self.is_word = False
        self.is_words = False

    def __repr__(self):
        return f"V{self.pd_id}"

    def __hash__(self):
        return self.pd_id

    cdef bint equals(self, Term other):
        return other.is_var and (<Var> other).pd_id == self.pd_id

    def __eq__(self, other):
        return isinstance(other, Var) and self.equals(other)


cdef class Word(Term):

    cdef readonly int word_pos
    cdef readonly str word

    def __cinit__(self, int word_pos, str word):
        super().__init__()
        self.word_pos = word_pos
        self.word = word
        self.is_var = False
        self.is_unavailable = False
        self.is_word = True
        self.is_words = False

    def __repr__(self):
        return "W%d" % self.word_pos

    def __hash__(self):
        return hash((self.word_pos, self.word))

    cdef bint equals(self, Term other):
        if not other.is_word:
            return False
        cdef Word x = other
        return self.word == x.word and self.word_pos == x.word_pos

    def __eq__(self, other):
        return isinstance(other, Word) and self.equals(other)


cdef class Words(Term):

    cdef readonly set words
    cdef int _hash

    def __cinit__(self, set words):
        super().__init__()
        self.words = words
        self.is_var = False
        self.is_unavailable = False
        self.is_word = False
        self.is_words = True
        self._hash = hash(frozenset(words))

    def __repr__(self):
        return "WS:" + ",".join(map(str, self.words))

    def __hash__(self):
        return self._hash

    cdef bint equals(self, Term other):
        if not other.is_words or (<Words> other)._hash != self._hash:
            return False
        return self.words == (<Words> other).words

    def __eq__(self, other):
        return isinstance(other, Words) and self.equals(other)


######################################################
#    BOUNDNESS and DepLink and UnfinishedDepLink     #
######################################################


cdef class UnfinishedDepLink:

    cdef readonly Category head_cat
    cdef readonly int slot
    cdef readonly Term head_words
    cdef readonly Term dep_words
    cdef readonly bint is_bound
    cdef readonly bint is_unbound
    cdef readonly bint is_adj
    cdef int _hash

    def __cinit__(self,
                 Category head_cat,
                 int slot,
                 Term head_words,
                 Term dep_words,
                 bint is_bound,
                 bint is_unbound,
                 bint is_adj):
        self.head_cat = head_cat
        self.slot = slot
        self.head_words = head_words
        self.dep_words = dep_words
        self.is_bound = is_bound
        self.is_unbound = is_unbound
        self.is_adj = is_adj
        self._hash = hash(self.__reduce__())

    def __reduce__(self):
        return self.head_cat, self.slot, self.head_words, self.dep_words, self.is_bound, self.is_unbound, self.is_adj

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        cdef UnfinishedDepLink x
        if isinstance(other, UnfinishedDepLink):
            x = other
            if x._hash != self._hash:
                return False
            else:
                return x.__reduce__() == self.__reduce__()
        else:
            return False

    cdef inline UnfinishedDepLink change_boundness(self, bint is_bound, bint is_unbound):
        return UnfinishedDepLink(
            head_cat = self.head_cat,
            slot = self.slot,
            head_words = self.head_words,
            dep_words = self.dep_words,
            is_bound = is_bound,
            is_unbound = is_unbound,
            is_adj = self.is_adj
        )

    cdef UnfinishedDepLink copy(self,
                                Category head_cat = None,
                                int slot = -1,
                                Term head_words = None,
                                Term dep_words = None,
                                bint is_bound = -1,
                                bint is_unbound = -1,
                                bint is_adj = -1):
        head_cat = head_cat if head_cat is not None else self.head_cat
        slot = slot if slot!=-1 else self.slot
        head_words = head_words if head_words is not None else self.head_words
        dep_words = dep_words if dep_words is not None else self.dep_words
        is_bound = is_bound if is_bound!=-1 else self.is_bound
        is_unbound = is_unbound if is_unbound!=-1 else self.is_unbound
        is_adj = is_adj if is_adj!=-1 else self.is_adj
        return UnfinishedDepLink(
            head_cat,
            slot,
            head_words,
            dep_words,
            is_bound,
            is_unbound,
            is_adj
        )

    def __repr__(self):
        return "%s slot %d var %s" % (self.head_cat, self.slot, self.dep_words)

    cdef UnfinishedDepLink replace_vars(self, dict subs):
        if self.dep_words.is_var and (self.dep_words in subs):
            return UnfinishedDepLink(head_cat=self.head_cat,
                                     slot=self.slot,
                                     head_words=self.head_words,
                                     dep_words=subs[self.dep_words],
                                     is_bound=self.is_bound,
                                     is_unbound=self.is_unbound,
                                     is_adj=self.is_adj)
        else:
            return self

    cdef inline list to_final_deps(self):
        if self.head_words.is_words and self.dep_words.is_words:
            res = []
            head_words = (<Words> self.head_words).words
            dep_words = (<Words> self.dep_words).words
            for h in head_words:
                for d in dep_words:
                    res.append(
                        DepLink(
                            head_cat=self.head_cat,
                            head_pos=h.word_pos,
                            dep_pos=d.word_pos,
                            dep_slot=self.slot,
                            head_word=h.word,
                            dep_word=d.word,
                            is_bound=self.is_bound,
                            is_unbound=self.is_unbound,
                            is_adj=self.is_adj,
                            is_conj=False))
            return res
        else:
            return []


cdef class DepLink:

    def __cinit__(self,
                  Category head_cat,
                  int head_pos,
                  int dep_pos,
                  int dep_slot,
                  str head_word,
                  str dep_word,
                  bint is_bound,
                  bint is_unbound,
                  bint is_adj,
                  bint is_conj):
        self.head_cat = head_cat
        self.head_pos = head_pos
        self.dep_pos = dep_pos
        self.dep_slot = dep_slot
        self.head_word = head_word
        self.dep_word = dep_word
        self.is_bound = is_bound
        self.is_unbound = is_unbound
        self.is_adj = is_adj
        self.is_conj = is_conj
        self._hash = hash((self.head_cat,
                           self.head_pos,
                           self.dep_pos,
                           self.dep_slot,
                           self.head_word,
                           self.dep_word))

    cpdef tuple __reduce__(self):
        return (self.head_cat,
                self.head_pos,
                self.dep_pos,
                self.dep_slot,
                self.head_word,
                self.dep_word,
                self.is_bound,
                self.is_unbound,
                self.is_adj,
                self.is_conj)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, DepLink):
            x = other
            if self._hash != (<DepLink> other)._hash:
                return False
            else:
                return other.__reduce__() == self.__reduce__()
        else:
            return False

    def __repr__(self):
        return self.__str__()

    property label:
        def __get__(self):
            if self.is_bound:
                bs  = ":b"
            elif self.is_unbound:
                bs  = ":u"
            else:
                bs  = ""
            adj = ":a" if self.is_adj else ""
            cnj = ":c" if self.is_conj else ""
            return f"{self.dep_slot}{bs}{adj}{cnj}"

    def __str__(self):
        return "%s:%s   (%d %s) --> (%d %s)" % (self.head_cat,
                                                self.label,
                                                self.head_pos,
                                                self.head_word,
                                                self.dep_pos,
                                                self.dep_word)



######################################################
#                       FORMULA                      #
######################################################

def unify2(Formula lf, Formula rf):
    return unify(lf, rf)

cdef set unify(Formula lf, Formula rf):
    lt = lf.head
    rt = rf.head
    if lt.is_unavailable or rt.is_unavailable:
        sub = set()
    elif lt.is_var:
        sub = {(lt, rt)}
    elif rt.is_var:
        sub = {(rt, lt)}
    else:
        sub = set()

    cdef PAFunctor lff, rff

    if lf.is_functor and rf.is_functor:
        lff = lf
        rff = rf
        return sub | unify(lff.left, rff.left) | unify(lff.right, rff.right)
    elif lf.is_atom and rf.is_atom:
        return sub
    else:
        raise Exception("you shouldn't get here")

cdef class Formula:

    cdef readonly Term head
    cdef readonly bint is_bound
    cdef readonly bint is_unbound
    cdef readonly bint is_atom
    cdef readonly bint is_functor
    cdef readonly set vars

    def is_bounded_formula(self):
        b, u = self.all_non_local_terms()
        return not (b==set() and u==set())

    cdef bint equals(self, Formula f):
        raise NotImplementedError()

    cdef tuple cut_args(self, int args_to_drop = -1):
        if args_to_drop == -1:
            if self.is_atom:
                return self, []
            else:
                l = (<PAFunctor> self).left
                r = (<PAFunctor> self).right
                ll, args = l.cut_args()
                args.append(r)
                return ll, args
        else:
            if args_to_drop==0:
                return self, []
            elif self.is_atom:
                return None
            else:
                l = (<PAFunctor> self).left
                r = (<PAFunctor> self).right
                y = l.cut_args(args_to_drop-1)
                if y is None:
                    return None
                else:
                    ll, args = y
                    args.append(r)
                    return ll, args

    cdef Formula replace_head_term(self, Term h):
        cdef PAFunctor x
        if self.is_functor:
            x = self
            return PAFunctor(x.head, x.is_bound, x.is_unbound, x.left.replace_head_term(h), x.right)
        else:
            return PAAtom(h, self.is_bound, self.is_unbound)

    cdef tuple all_non_local_terms(self):
        cdef set lB, lU, rB, rU
        if self.is_functor:
            lB, lU = (<PAFunctor> self).left.all_non_local_terms()
            rB, rU = (<PAFunctor> self).right.all_non_local_terms()
            if self.is_bound:
                return lB | rB | {self.head}, lU | rU
            elif self.is_unbound:
                return lB | rB, lU | rU | {self.head}
            else:
                return lB | rB, lU | rU
        else:
            if self.is_bound:
                return {self.head}, set()
            elif self.is_unbound:
                return set(), {self.head}
            else:
                return set(), set()

    @staticmethod
    cdef Formula from_cat(Category cat):
        if cat.is_atomic:
            return PAAtom(Unavailable(), False, False)
        elif cat.is_functor:
            l = (<Functor> cat).res
            r = (<Functor> cat).arg
            return PAFunctor(Unavailable(),
                             False,
                             False,
                             Formula.from_cat(l),
                             Formula.from_cat(r))
        else:
            subcat = (<ConjCat> cat).sub_cat
            subf = Formula.from_cat(subcat)
            return PAFunctor(Unavailable(), False, False, subf, subf)

    cdef Formula to_indexed_formula(self, int inx = 1):
        cdef Formula left, right, new_right
        if self.is_atom:
            return self
        else:
            left = (<PAFunctor> self).left
            right = (<PAFunctor> self).right
            if right.is_atom:
                new_right = PAAtom(Var(inx), right.is_bound, right.is_unbound)
            else:
                new_right = PAFunctor(Var(inx), right.is_bound, right.is_unbound,
                                      (<PAFunctor> right).left,
                                      (<PAFunctor> right).right)
            return PAFunctor(self.head, self.is_bound, self.is_unbound, left.to_indexed_formula(inx+1), new_right)

    cdef Term find_head_term(self):
        if self.is_atom:
            return (<PAAtom> self).head
        else:
            return (<PAFunctor> self).left.find_head_term()

    cdef Formula replace_vars(self, dict subs):
        cdef Term h = self.head
        if h in subs:
            h = subs[h]

        cdef Formula left, right
        cdef Formula res
        if self.is_functor:
            left  = (<PAFunctor> self).left
            right = (<PAFunctor> self).right
            res = PAFunctor(h, self.is_bound, self.is_unbound, left.replace_vars(subs), right.replace_vars(subs))
        else:
            res = PAAtom(h, self.is_bound, self.is_unbound)
        return res

    cdef inline Formula assign_head_word(self, int word_pos, str word_str):
        return self.assign_head_words(Words({Word(word_pos, word_str)}))

    cdef Formula assign_head_words(self, Term head):
        if self.is_functor:
            return PAFunctor(self.head,
                             self.is_bound,
                             self.is_unbound,
                             (<PAFunctor> self).left.assign_head_words(head),
                             (<PAFunctor> self).right)
        elif self.head.is_unavailable:
            return PAAtom(head, self.is_bound, self.is_unbound)
        else:
            return self

    @staticmethod
    def from_str(string):
        cdef Category cat = Category.from_str(string)
        return Formula._transform_cat_to_formula(cat)

    @staticmethod
    cdef _transform_cat_to_formula(Category cat):
        cdef bint is_bound = False
        cdef bint is_unbound = False
        if cat.pa_id>=0:
            pa_id = Var(cat.pa_id)
            if cat.pa_b == "":
                is_bound   = False
                is_unbound = False
            elif cat.pa_b == "B":
                is_bound   = True
                is_unbound = False
            elif cat.pa_b == "U":
                is_bound   = False
                is_unbound = True
            else:
                err = f"pa_id {cat.pa_id}"
                err += f"pa_b {cat.pa_b}"
                err += "you shouldn't get here"
                raise Exception(err)
        else:
            pa_id = Unavailable()
            is_bound   = False
            is_unbound = False

        if cat.is_atomic:
            f = PAAtom(pa_id, is_bound, is_unbound)
        else:
            left  = Formula._transform_cat_to_formula((<Functor> cat).res)
            right = Formula._transform_cat_to_formula((<Functor> cat).arg)
            f = PAFunctor(pa_id, is_bound, is_unbound, left, right)

        return f


cdef class PAFunctor(Formula):

    cdef readonly Formula left
    cdef readonly Formula right

    def __cinit__(self,
                 Term head,
                 bint is_bound,
                 bint is_unbound,
                 Formula left,
                 Formula right):
        if head.is_var:
            self.vars = {head}
        else:
            self.vars = set()
        self.vars.update(left.vars)
        self.vars.update(right.vars)
        self.left = left
        self.right = right
        self.head = head
        self.is_bound = is_bound
        self.is_unbound = is_unbound
        self.is_atom = False
        self.is_functor = not self.is_atom

    cdef bint equals(self, Formula f):
        if not f.is_functor:
            return False
        cdef PAFunctor x = f
        return self.head == x.head and \
               self.is_bound==x.is_bound and \
               self.is_unbound==x.is_unbound and \
               self.left.equals(x.left) and \
               self.right.equals(x.right)

    def __str__(self):
        if self.is_bound:
            b = ":B"
        elif self.is_unbound:
            b = ":U"
        else:
            b = ""
        return "( %s | %s )%s%s" % (self.left, self.right, self.head, b)

    def __eq__(self, other):
        if isinstance(other, PAFunctor):
            return self.equals(other)
        else:
            return False

    def __hash__(self):
        return hash((self.head, self.is_bound, self.is_unbound, self.left, self.right))


cdef class PAAtom(Formula):

    def __cinit__(self, Term head, bint is_bound, bint is_unbound):
        if head.is_var:
            self.vars = {head}
        else:
            self.vars = set()
        self.head = head
        self.is_bound = is_bound
        self.is_unbound = is_unbound
        self.is_atom = True
        self.is_functor = not self.is_atom

    def __str__(self):
        if self.is_bound:
            b = ":B"
        elif self.is_unbound:
            b = ":U"
        else:
            b = ""
        return str(self.head) + b

    cdef bint equals(self, Formula f):
        if not f.is_atom:
            return False
        cdef PAAtom x = f
        return self.head == x.head and self.is_bound==x.is_bound and self.is_unbound==x.is_unbound

    def __eq__(self, other):
        cdef PAAtom aother
        if isinstance(other, PAAtom):
            return self.equals(other)
        else:
            return False

    def __hash__(self):
        return hash((self.head, self.is_bound, self.is_unbound))
