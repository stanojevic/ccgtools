# cython: boundscheck=False

from sys import intern

cpdef inline dict matching(Category cat_a, Category cat_b):# -> Optional[Dict[str, Tuple[bool, str]]]:
    """if things don't match returns None
    if things do match returns a substitution dictionary label->(feature_is_from_left, feature)"""
    cdef Functor cat_af, cat_bf
    cdef Atomic cat_aa, cat_ba
    cdef dict res_sub, arg_sub
    if cat_a.is_functor and cat_b.is_functor:
        cat_af = cat_a
        cat_bf = cat_b
        if cat_af.slash_is_fwd != cat_bf.slash_is_fwd:
            return None
        res_sub = matching(cat_af.res, cat_bf.res)
        if res_sub is None:
            return None
        arg_sub = matching(cat_af.arg, cat_bf.arg)
        if arg_sub is None:
            return None
        res_sub.update(arg_sub)
        return res_sub
    elif cat_a.is_atomic and cat_b.is_atomic:
        cat_aa = cat_a
        cat_ba = cat_b
        if cat_aa.label != cat_ba.label:
            return None
        elif cat_aa.feature == cat_ba.feature:
            return {}
        elif cat_aa.has_feature and not cat_ba.has_feature:
            return {cat_aa.label: (True, cat_aa.feature)}
        elif not cat_aa.has_feature and cat_ba.has_feature:
            return {cat_aa.label: (False, cat_ba.feature)}
        else:
            return None

# def filter_substitution(sub: Dict[str, Tuple[bool, str]], from_left: bool):# -> Dict[str, str]:
cdef inline dict filter_substitution(dict sub, bint from_left):# -> Dict[str, str]:
    return {k: v for k, (b, v) in sub.items() if b == from_left}


cdef class Category:

    def __cinit__(self):
        self.pa_id = -1

    def __ne__(self, other):
        return not self.__eq__(other)

    cdef bint equals(self, Category y):
        raise NotImplementedError()

    cdef bint equals_featureless(self, Category y):
        raise NotImplementedError()

    cpdef tuple cut_args(self, int args_to_drop = -1): # -> Optional[Tuple[Category, List[Tuple[Slash, Category]]]]:
        cdef Functor catf
        cdef Category ll
        cdef list args
        cdef tuple y
        if args_to_drop < 0:
            if self.is_functor:
                catf = self
                ll, args = catf.res.cut_args()
                args.append((catf.slash_is_fwd, catf.arg))
                return ll, args
            else:
                return self, []
        else:
            if args_to_drop == 0:
                return self, []
            elif self.is_functor:
                catf = self
                y = catf.res.cut_args(args_to_drop-1)
                if y is None:
                    return None
                else:
                    ll, args = y
                    args.append((catf.slash_is_fwd, catf.arg))
                    return ll, args
            else:
                return None

    cdef Category apply_substitution(self, dict sub):
        cdef Functor catf
        cdef Atomic cata
        if self.is_functor:
            catf = self
            return Functor(
                res=catf.res.apply_substitution(sub),
                arg=catf.arg.apply_substitution(sub),
                slash_is_fwd=catf.slash_is_fwd)
        elif self.is_atomic:
            cata = self
            if not cata.has_feature and cata.label in sub:
                return Atomic(label=cata.label, feature=sub[cata.label])
            else:
                return cata
        else:
            raise Exception("this type of category cannot participate in feature unification")

    @staticmethod
    def from_str(str string):
        # this is a bug in the CCGbank
        if string == "((S[b]\\NP)/NP)/":
            string = "((S[b]\\NP)/NP)"

        # needed for handling EasyCCG output
        string = string.replace("[X]", "")

        # removing a useless "noun bare" feature
        string = string.replace("[nb]", "")

        cdef bint is_conj = string.endswith("[conj]")
        string = string.replace("[conj]", "")

        cdef list tokens = Category._tokenize(string)
        cdef Category cat
        cdef int pos
        cat = Category._process_tokens(tokens)

        if is_conj:
            cat = ConjCat(sub_cat=cat)
        return cat

    @staticmethod
    cdef Category _process_tokens(list tokens):

        cdef list stack_of_cats = []
        cdef list stack_of_slashes = []

        cdef list cats = []
        cdef list slashes = []

        cdef int pos = 0
        cdef int end = len(tokens)

        while pos != end:

            if tokens[pos] == "/":
                slashes.append(True)
                pos += 1
            elif tokens[pos] == "\\":
                slashes.append(False)
                pos += 1

            if tokens[pos] == "(":
                pos += 1
                stack_of_cats.append(cats)
                stack_of_slashes.append(slashes)
                cats = []
                slashes = []
            elif tokens[pos] == ")":
                pos += 1
                cat = Category._combine_cats_and_slashes(cats, slashes)
                cats = stack_of_cats.pop()
                slashes = stack_of_slashes.pop()
                cats.append(cat)
            else:
                if pos + 1 < end and tokens[pos + 1] == "[":
                    label = tokens[pos]
                    feature = tokens[pos + 2]
                    pos += 4
                    cat = Atomic(label=label, feature=feature)
                    cats.append(cat)
                else:
                    label = tokens[pos]
                    pos += 1
                    cat = Atomic(label=label)
                    cats.append(cat)

            if pos != end and tokens[pos] == "_":
                es = tokens[pos+1].split(":")
                cat.pa_id = int(es[0])
                cat.pa_b = es[1] if len(es) == 2 else ""
                pos += 2

        return Category._combine_cats_and_slashes(cats, slashes)

    @staticmethod
    cdef Category _combine_cats_and_slashes(list cats, list slashes):
        cat = cats[0]
        for arg, slash in zip(cats[1:], slashes):
            cat = Functor(slash_is_fwd=slash, res=cat, arg=arg)
        return cat

    @staticmethod
    cdef inline list _tokenize(str string):
        return string \
                     .replace("(", " ( ") \
                     .replace(")", " ) ") \
                     .replace("[", " [ ") \
                     .replace("]", " ] ") \
                     .replace("/", " / ") \
                     .replace("\\", " \\ ") \
                     .replace("_", " _ ") \
                     .replace("  ", " ") \
                     .strip() \
                     .split(" ")

cdef _NP_str = intern("NP")
cdef _PP_str = intern("PP")
cdef _N_str = intern("N")
cdef _S_str = intern("S")
cdef _LRB_str = intern("LRB")
cdef _RRB_str = intern("RRB")
cdef _LQU_str = intern("LQU")
cdef _RQU_str = intern("RQU")
cdef _conj_str = intern("conj")
cdef _comma_str = intern(",")
cdef _column_str = intern(":")
cdef _semicolumn_str = intern(";")
cdef _dot_str = intern(".")


cdef class Atomic(Category):

    def __cinit__(self, str label, str feature=None):

        label = intern(label)
        if feature is not None:
            feature = intern(feature)
        self.label = label
        self.feature = feature

        self.is_atomic  = True
        self.is_functor = False
        self.is_conj    = False

        self.has_feature = feature is not None
        cdef bint is_basic_punc = (label == _comma_str) or (label == _semicolumn_str) or \
                                  (label == _column_str) or (label == _dot_str)
        self.is_conj_atom = (label == _conj_str) or is_basic_punc
        self.is_punc = (label == _LRB_str) or (label == _RRB_str) or \
                       (label == _LQU_str) or (label == _RQU_str) or is_basic_punc
        self.is_left_adj_cat = False
        self.is_right_adj_cat = False
        self.is_adj_cat = False
        self.is_np = label == _NP_str
        self.is_pp = label == _PP_str
        self.is_n = label == _N_str
        self.is_nominal = self.is_np or self.is_n
        self.is_s = label == _S_str
        self.is_verbal = False

    def __reduce__(self):
        return Atomic, (self.label, self.feature)

    def __eq__(self, other):
        if isinstance(other, Atomic):
            return (self.label == other.label) and (self.feature == other.feature)
        else:
            return False

    def __repr__(self):
        if self.has_feature:
            return "%s[%s]" % (self.label, self.feature)
        else:
            return self.label

    cpdef Atomic strip_features(self):
        return Atomic(self.label)

    def __hash__(self):
        return hash((self.label, self.feature))

    cdef bint equals(self, Category y):
        if not y.is_atomic:
            return False
        cdef Atomic z = y
        return (z.label == self.label) and (z.feature == self.feature)

    cdef bint equals_featureless(self, Category y):
        if not y.is_atomic:
            return False
        cdef Atomic z = y
        return z.label == self.label

cdef class ConjCat(Category):

    def __reduce__(self):
        return ConjCat, (self.sub_cat,)

    def __cinit__(self, Category sub_cat):

        assert not sub_cat.is_conj
        self.sub_cat = sub_cat

        self.is_atomic  = False
        self.is_functor = False
        self.is_conj    = True

        self.is_punc = False
        self.is_left_adj_cat = False
        self.is_right_adj_cat = False
        self.is_adj_cat = False
        self.is_np = False
        self.is_pp = False
        self.is_n = False
        self.is_nominal = False
        self.is_s = False
        self.is_verbal = False

    def __eq__(self, other):
        if isinstance(other, ConjCat):
            return self.sub_cat == other.sub_cat
        else:
            return False

    def __repr__(self):
        return "%s[conj]" % self.sub_cat

    cpdef ConjCat strip_features(self):
        return ConjCat(self.sub_cat.strip_features())

    def __hash__(self):
        return hash(self.sub_cat)

    cdef bint equals(self, Category y):
        if not y.is_conj:
            return False
        cdef ConjCat z = y
        return z.sub_cat.equals(self.sub_cat)

    cdef bint equals_featureless(self, Category y):
        if not y.is_conj:
            return False
        cdef ConjCat z = y
        return z.sub_cat.equals_featureless(self.sub_cat)


cdef class Functor(Category):

    def __reduce__(self):
        return Functor, (self.slash_is_fwd, self.res, self.arg)

    def __cinit__(self, bint slash_is_fwd, Category res, Category arg):

        self.res = res
        self.arg = arg
        self.slash_is_fwd = slash_is_fwd

        self.is_atomic  = False
        self.is_functor = True
        self.is_conj    = False

        self.is_punc = False
        self.is_left_adj_cat = (res == arg and slash_is_fwd)
        self.is_right_adj_cat = (res == arg and not slash_is_fwd)
        self.is_adj_cat = self.is_left_adj_cat or self.is_right_adj_cat
        self.is_np = False
        self.is_pp = False
        self.is_n = False
        self.is_nominal = False
        self.is_s = False
        self.is_verbal = (self.res.is_atomic and self.res.is_s) or self.res.is_verbal

    def __eq__(self, other):
        if isinstance(other, Functor):
            return (self.slash_is_fwd == other.slash_is_fwd) and (self.res == other.res) and (self.arg == other.arg)
        else:
            return False

    def __repr__(self):
        arg_str = str(self.arg)
        if self.arg.is_functor:
            arg_str = "(" + arg_str + ")"
        res_str = str(self.res)
        if self.res.is_functor:
            res_str = "(" + res_str + ")"
        slash_str = "/" if self.slash_is_fwd else "\\"
        return "%s%s%s" % (res_str, slash_str, arg_str)

    cpdef Functor strip_features(self):
        return Functor(self.slash_is_fwd, self.res.strip_features(), self.arg.strip_features())

    def __hash__(self):
        return hash((self.slash_is_fwd, self.res, self.arg))

    cdef bint equals(self, Category y):
        if not y.is_functor:
            return False
        cdef Functor z = y
        return z.res.equals(self.res) and z.arg.equals(self.arg)

    cdef bint equals_featureless(self, Category y):
        if not y.is_functor:
            return False
        cdef Functor z = y
        return z.res.equals_featureless(self.res) and z.arg.equals_featureless(self.arg)
