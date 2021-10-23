from typing import Set, Dict, List
import lark
from .combinators import *
from .derivation import *
from .category import *
from .visualization import escape_latex_text


def bwrap(s):
    return "(" + s + ")"


def bwrap_tex(s):
    return "\\left(" + s + "\\right)"


def is_simple(t):
    return isinstance(t, (Skolem, Var, Constant, UQuant, EQuant)) or \
           (isinstance(t, Neg) and is_simple(t.pred))


def is_complex(t):
    return not is_simple(t)


_alphabet = [chr(ord('a') + x) for x in range(26)]
var_names = _alphabet + [a + b for a in _alphabet for b in _alphabet]

# this is a simple greek alphabet with some characters excluded because they have a special meaning (lambda)
time_names_latex = ["\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta", "\\eta", "\\theta", "\\iota",
     "\\kappa", "\\mu", "\\nu", "\\xi", "\\rho", "\\sigma", "\\tau", "\\upsilon", "\\phi", "\\chi", "\\psi", "\\omega"]
time_names_str = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'μ', 'ν', 'ξ', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω']


def merge_var_lists(xs, ys):
    xs = list(xs)
    for y in ys:
        if y not in xs:
            xs.append(y)
    return xs


def b0_logic(a, b, time):
    [a, b] = alpha_transform_expressions([a, b])
    res = Apply(a, b)
    return res.beta_normal_form().refresh_env(time)


def b1_logic(a, b, time):
    x = Var(0)
    [a, b, x] = alpha_transform_expressions([a, b, x])
    res = Lambda(x, Apply(a, Apply(b, x)))
    return res.beta_normal_form().refresh_env(time)


def b2_logic(a, b, time):
    x = Var(0)
    y = Var(0)
    [a, b, x, y] = alpha_transform_expressions([a, b, x, y])
    res = Lambda(x, Lambda(y, Apply(a, Apply(Apply(b, x), y))))
    return res.beta_normal_form().refresh_env(time)


def strip_lambda_args(term):
    args = []
    core = term
    while isinstance(core, Lambda):
        args.append(core.var)
        core = core.formula
    return args, core


def coord_logic(a, is_conj: bool, time):
    p = Var(0)
    [a, p] = alpha_transform_expressions([a, p])

    args = []
    coreLeft = a
    while isinstance(coreLeft, Lambda):
        args.append(coreLeft.var)
        coreLeft = coreLeft.formula

    coreRight = p
    for v in reversed(args):
        coreRight = Apply(coreRight, v)

    if is_conj:
        core = Conj(coreRight, coreLeft)
    else:
        core = Disj(coreRight, coreLeft)

    for v in reversed(args):
        core = Lambda(v, core)
    core = Lambda(p, core)

    return core.beta_normal_form().refresh_env(time)


def tr_logic(a, time):
    x = Var(0)
    [a, x] = alpha_transform_expressions([a, x])
    res = Lambda(x, Apply(x, a))
    return res.beta_normal_form().refresh_env(time)


def alpha_transform_expressions(exps):
    ee = exps[0]
    curr_max = max([x.var_id for x in ee.all_vars()], default=0)
    new_exps = [ee]
    for e in exps[1:]:
        ee = e.shift_var_start_by(1 + curr_max)
        new_exps.append(ee)
        curr_max = max([x.var_id for x in ee.all_vars()], default=curr_max)
    return new_exps


class Term:

    def __init__(self):
        pass

    def assign_index(self, index: int):
        raise NotImplementedError()

    def all_vars(self):
        raise NotImplementedError()

    def all_times(self):
        raise NotImplementedError()

    def all_skolems(self):
        raise NotImplementedError()

    def beta_normal_form(self):
        raise NotImplementedError()

    def shift_var_start_by(self, shift: int):
        raise NotImplementedError()

    def filter_time(self, time_filter):
        raise NotImplementedError()

    def replace_var(self, var, term):
        raise NotImplementedError()

    def replace_constant_label(self, label_from, label_to):
        raise NotImplementedError()

    def refresh_env(self, time, env=None):
        raise NotImplementedError()

    def term_to_str(self, var_name_mapping, time_name_mapping):
        raise NotImplementedError()

    def __str__(self):
        var_name_mapping = dict(zip(self.all_vars(), var_names))
        times = list(self.all_times())
        assert len(times) <= len(time_names_str)
        time_name_mapping = dict(zip(times, time_names_str))
        return self.term_to_str(var_name_mapping, time_name_mapping)

    def term_to_tex(self, var_name_mapping, time_name_mapping):
        raise NotImplementedError()

    # TODO skolem_to_existential doesn't work -- zipper needed or deriv tree
    def skolem_to_existential(self):
        curr_max = max([x.var_id for x in self.all_vars()], default=0)
        skolems = self.all_skolems()
        myself = self
        for sk in skolems:
            curr_max += 1
            myself = myself._skolem_to_existential_rec(sk, Var(curr_max), False)
        return myself

    def _skolem_to_existential_rec(self, sk, free_var, replace_skolem_bool):
        raise NotImplementedError()

    def to_latex(self):
        var_name_mapping = dict(zip(self.all_vars(), var_names))
        times = list(self.all_times())
        assert len(times) <= len(time_names_latex)
        time_name_mapping = dict(zip(times, time_names_latex))
        return self.term_to_tex(var_name_mapping, time_name_mapping)

    def to_latex_standalone(self):
        res = ""
        res += "\\documentclass{standalone}\n"
        res += "\\usepackage{amsmath}\n"
        res += "\\usepackage{xcolor}  % this is for coloring logical formulas\n"
        res += "\\usepackage{mathptmx}  % this is for nicer looking lambdas\n"
        res += "\\begin{document}\n"
        res += "$" + self.to_latex() + "$\n"
        res += "\\end{document}\n"
        return res

    def display(self, title: str = None, warning: str = None):
        from IPython.display import Image, display, HTML
        from .universal_tree_visualization import create_temp_file
        tmp_fn = create_temp_file("", "png")
        self.save(tmp_fn)
        display(HTML("<br/>"))
        if title:
            display(HTML(f"<b><font size=\"3\" color=\"red\">{title}</font></b>"))
        if warning:
            display(HTML(f"<font color=\"red\">WARNING: {warning}</font>"))
        if isinstance(tmp_fn, str):
            display(Image(tmp_fn))
        else:
            display(tmp_fn)

    def _repr_png_(self):
        from .universal_tree_visualization import create_temp_file
        tmp_fn = create_temp_file("", "png")
        self.save(tmp_fn)
        with open(tmp_fn, "rb") as fh:
            x = fh.read()
        return x

    def save(self, fn: str):
        from .universal_tree_visualization import run_latex
        run_latex(self.to_latex_standalone(), out_file=fn)

    def unpack_all_readings(self):
        skolems = self.all_skolems()

        unique_index_times = dict()
        for sk in skolems:
            if sk.index not in unique_index_times:
                unique_index_times[sk.index] = set()
            for e in sk.environments:
                unique_index_times[sk.index].add(e.time)
        unique_index_times = {x : sorted(list(y), reverse=True) for x, y in unique_index_times.items()}
        index_times = unique_index_times.items()
        indices = [x[0] for x in index_times]
        times_lists = [x[1] for x in index_times]
        readings = []

        from itertools import product
        for time_choice in product(*times_lists):
            time_filter = dict(zip(indices, time_choice))
            filtered_skolem = self.filter_time(time_filter)
            # TODO skolem_to_existential doesn't work -- zipper needed or deriv tree
            # reading = filtered_skolem.skolem_to_existential()
            reading = filtered_skolem
            readings.append(reading)
        return readings


class Constant(Term):

    def __init__(self, label: str, args=None):
        super().__init__()
        if args is None:
            args = []
        self.args = args
        self.label = label

    def assign_index(self, index: int):
        if self.args:
            return Constant(self.label, [a.assign_index(index) for a in self.args])
        else:
            return self

    def replace_var(self, var, term):
        if self.args:
            return Constant(self.label, [a.replace_var(var, term) for a in self.args])
        else:
            return self

    def replace_constant_label(self, label_from, label_to):
        if self.label == label_from:
            new_label = label_to
        else:
            new_label = self.label
        return Constant(new_label, [a.replace_constant_label(label_from, label_to) for a in self.args])

    def beta_normal_form(self):
        if self.args:
            return Constant(self.label, [a.beta_normal_form() for a in self.args])
        else:
            return self

    def refresh_env(self, time, env=None):
        if self.args:
            return Constant(self.label, [a.refresh_env(time, env) for a in self.args])
        else:
            return self

    def _skolem_to_existential_rec(self, sk, free_var, replace_skolem_bool):
        return Constant(self.label, [a._skolem_to_existential_rec(sk, free_var, replace_skolem_bool) for a in self.args])

    def term_to_str(self, var_name_mapping, time_name_mapping):
        label = self.label  # TODO escape characters in label for terminal
        if self.args:
            return "%s'(%s)" % (label, ", ".join([a.term_to_str(var_name_mapping, time_name_mapping) for a in self.args]))
        else:
            return "%s'" % label

    def term_to_tex(self, var_name_mapping, time_name_mapping):
        label = escape_latex_text(self.label)
        if self.args:
            return "\\textit{%s}'\\left(%s\\right)" % (label, ", ".join([a.term_to_tex(var_name_mapping, time_name_mapping) for a in self.args]))
        else:
            return "\\textit{%s}'" % label

    def all_vars(self):
        out = []
        for a in self.args:
            out = merge_var_lists(out, a.all_vars())
        return out

    def all_times(self):
        res = set()
        for a in self.args:
            res.update(a.all_times())
        return res

    def all_skolems(self):
        res = []
        for a in self.args:
            res.extend(a.all_skolems())
        return res

    def shift_var_start_by(self, shift: int):
        if self.args:
            return Constant(self.label, [a.shift_var_start_by(shift) for a in self.args])
        else:
            return self

    def filter_time(self, time_filter):
        if self.args:
            return Constant(self.label, [a.filter_time(time_filter) for a in self.args])
        else:
            return self


class Var(Term):

    def __init__(self, var_id: int):
        super().__init__()
        self.var_id = var_id

    def assign_index(self, index: int):
        return self

    def replace_var(self, var, term):
        return term if self == var else self

    def replace_constant_label(self, label_from, label_to):
        return self

    def _skolem_to_existential_rec(self, sk, free_var, replace_skolem_bool):
        return self

    def beta_normal_form(self):
        return self

    def refresh_env(self, time, env=None):
        return self

    def all_vars(self):
        return [self]

    def all_times(self):
        return set()

    def all_skolems(self):
        return []

    def shift_var_start_by(self, shift: int):
        return Var(var_id=self.var_id + shift)

    def filter_time(self, time_filter):
        return self

    def __hash__(self):
        return self.var_id

    def __eq__(self, other):
        if isinstance(other, Var):
            return other.var_id == self.var_id
        return False

    def __lt__(self, other):
        if isinstance(other, Var):
            return self.var_id < other.var_id
        return False

    def term_to_str(self, var_name_mapping, time_name_mapping):
        return var_name_mapping[self]

    def term_to_tex(self, var_name_mapping, time_name_mapping):
        return var_name_mapping[self]


class Apply(Term):

    def __init__(self, func: Term, arg: Term):
        super().__init__()
        assert isinstance(func, (Lambda, Constant, Var, Apply))
        self.func = func
        self.arg = arg

    def assign_index(self, index: int):
        return Apply(self.func.assign_index(index), self.arg.assign_index(index))

    def beta_normal_form(self):
        if isinstance(self.func, Lambda):
            return self.func.formula.replace_var(self.func.var, self.arg).beta_normal_form()
        elif isinstance(self.func, Constant):
            return Constant(self.func.label, self.func.args + [self.arg]).beta_normal_form()
        else:
            new_func = self.func.beta_normal_form()
            new_arg  = self.arg.beta_normal_form()
            new_self = Apply(new_func, new_arg)
            if isinstance(new_func, Lambda):
                return new_self.beta_normal_form()
            else:
                return new_self

    def refresh_env(self, time, env=None):
        return Apply(self.func.refresh_env(time, env), self.arg.refresh_env(time, env))

    def replace_var(self, var, term):
        return Apply(self.func.replace_var(var, term), self.arg.replace_var(var, term))

    def _skolem_to_existential_rec(self, sk, free_var, replace_skolem_bool):
        return Apply(self.func._skolem_to_existential_rec(sk, free_var, replace_skolem_bool), self.arg)

    def replace_constant_label(self, label_from, label_to):
        return Apply(self.func.replace_constant_label(label_from, label_to),
                     self.arg.replace_constant_label(label_from, label_to))

    def all_vars(self) -> List[Var]:
        return merge_var_lists(self.func.all_vars(), self.arg.all_vars())

    def all_times(self):
        return self.func.all_times() | self.arg.all_times()

    def all_skolems(self):
        return self.func.all_skolems() + self.arg.all_skolems()

    def shift_var_start_by(self, shift: int) -> Term:
        return Apply(self.func.shift_var_start_by(shift), self.arg.shift_var_start_by(shift))

    def filter_time(self, time_filter):
        return Apply(self.func.filter_time(time_filter), self.arg.filter_time(time_filter))

    def term_to_str(self, var_name_mapping: Dict[Var, str], time_name_mapping) -> str:
        lstr = self.func.term_to_str(var_name_mapping, time_name_mapping)
        if is_complex(self.func):  # and not isinstance(self.func, Apply):
            lstr = bwrap(lstr)
        rstr = self.arg.term_to_str(var_name_mapping, time_name_mapping)
        if is_complex(self.arg):
            rstr = bwrap(rstr)
        return "%s @ %s" % (lstr, rstr)

    def term_to_tex(self, var_name_mapping, time_name_mapping):
        lstr = self.func.term_to_tex(var_name_mapping, time_name_mapping)
        if is_complex(self.func):  # and not isinstance(self.func, Apply):
            lstr = bwrap_tex(lstr)
        rstr = self.arg.term_to_tex(var_name_mapping, time_name_mapping)
        if is_complex(self.arg):
            rstr = bwrap_tex(rstr)
        return "%s\\ %s" % (lstr, rstr)


class Lambda(Term):

    def __init__(self, var: Var, formula: Term):
        super().__init__()
        self.var = var
        self.formula = formula

    def assign_index(self, index: int):
        return Lambda(self.var, self.formula.assign_index(index))

    def replace_var(self, var, term):
        return Lambda(self.var.replace_var(var, term), self.formula.replace_var(var, term))

    def replace_constant_label(self, label_from, label_to):
        return Lambda(self.var.replace_constant_label(label_from, label_to),
                      self.formula.replace_constant_label(label_from, label_to))

    def beta_normal_form(self):
        return Lambda(self.var, self.formula.beta_normal_form())

    def refresh_env(self, time, env=None):
        return Lambda(self.var, self.formula.refresh_env(time, env))

    def _skolem_to_existential_rec(self, sk, free_var, replace_skolem_bool):
        return Lambda(self.var, self.formula._skolem_to_existential_rec(sk, free_var, replace_skolem_bool))

    def all_vars(self) -> List[Var]:
        return merge_var_lists([self.var], self.formula.all_vars())

    def all_times(self):
        return self.formula.all_times()

    def all_skolems(self):
        return self.formula.all_skolems()

    def shift_var_start_by(self, shift: int) -> Term:
        return Lambda(self.var.shift_var_start_by(shift), self.formula.shift_var_start_by(shift))

    def filter_time(self, time_filter):
        return Lambda(self.var, self.formula.filter_time(time_filter))

    def term_to_str(self, var_name_mapping: Dict[Var, str], time_name_mapping) -> str:
        return "λ%s %s" % (self.var.term_to_str(var_name_mapping, time_name_mapping), self.formula.term_to_str(var_name_mapping, time_name_mapping))

    def term_to_tex(self, var_name_mapping, time_name_mapping):
        return "\\lambda{}%s.\\ %s" % (self.var.term_to_tex(var_name_mapping, time_name_mapping), self.formula.term_to_tex(var_name_mapping, time_name_mapping))
        # return "\\lambda{}%s\\ %s" % (self.var.term_to_tex(var_name_mapping, time_name_mapping), self.formula.term_to_tex(var_name_mapping, time_name_mapping))


class Neg(Term):

    def __init__(self, pred: Term):
        super().__init__()
        self.pred = pred

    def assign_index(self, index: int):
        return Neg(self.pred.assign_index(index))

    def replace_var(self, var, term):
        return Neg(self.pred.replace_var(var, term))

    def replace_constant_label(self, label_from, label_to):
        return Neg(self.pred.replace_constant_label(label_from, label_to))

    def _skolem_to_existential_rec(self, sk, free_var, replace_skolem_bool):
        return Neg(self.pred._skolem_to_existential_rec(sk, free_var, replace_skolem_bool))

    def all_vars(self) -> List[Var]:
        return self.pred.all_vars()

    def all_times(self):
        return self.pred.all_times()

    def all_skolems(self):
        return self.pred.all_skolems()

    def beta_normal_form(self):
        return Neg(self.pred.beta_normal_form())

    def refresh_env(self, time, env=None):
        return Neg(self.pred.refresh_env(time, env))

    def shift_var_start_by(self, shift: int) -> Term:
        return Neg(self.pred.shift_var_start_by(shift))

    def filter_time(self, time_filter):
        return Neg(self.pred.filter_time(time_filter))

    def term_to_str(self, var_name_mapping: Dict[Var, str], time_name_mapping) -> str:
        s = self.pred.term_to_str(var_name_mapping, time_name_mapping)
        if is_complex(self.pred):
            s = bwrap(s)
            return "¬ " + s
        else:
            return "¬" + s

    def term_to_tex(self, var_name_mapping, time_name_mapping):
        s = self.pred.term_to_tex(var_name_mapping, time_name_mapping)
        if is_complex(self.pred):
            s = bwrap_tex(s)
            return "\\neg\\ " + s
        else:
            return "\\neg" + s


class Impl(Term):

    def __init__(self, left: Term, right: Term):
        super().__init__()
        self.left = left
        self.right = right

    def assign_index(self, index: int):
        return Impl(self.left.assign_index(index), self.right.assign_index(index))

    def replace_var(self, var, term):
        return Impl(self.left.replace_var(var, term), self.right.replace_var(var, term))

    def replace_constant_label(self, label_from, label_to):
        return Impl(self.left.replace_constant_label(label_from, label_to),
                    self.right.replace_constant_label(label_from, label_to))

    def _skolem_to_existential_rec(self, sk, free_var, replace_skolem_bool):
        return Impl(self.left._skolem_to_existential_rec(sk, free_var, replace_skolem_bool),
                    self.right._skolem_to_existential_rec(sk, free_var, replace_skolem_bool))

    def beta_normal_form(self):
        return Impl(self.left.beta_normal_form(), self.right.beta_normal_form())

    def refresh_env(self, time, env=None):
        return Impl(self.left.refresh_env(time, env), self.right.refresh_env(time, env))

    def all_vars(self) -> List[Var]:
        return merge_var_lists(self.left.all_vars(), self.right.all_vars())

    def all_times(self):
        return self.left.all_times() | self.right.all_times()

    def all_skolems(self):
        return self.left.all_skolems() + self.right.all_skolems()

    def shift_var_start_by(self, shift: int) -> Term:
        return Impl(self.left.shift_var_start_by(shift), self.right.shift_var_start_by(shift))

    def filter_time(self, time_filter):
        return Impl(self.left.filter_time(time_filter), self.right.filter_time(time_filter))

    def term_to_str(self, var_name_mapping: Dict[Var, str], time_name_mapping) -> str:
        lstr = self.left.term_to_str(var_name_mapping, time_name_mapping)
        if is_complex(self.left):
            lstr = bwrap(lstr)
        rstr = self.right.term_to_str(var_name_mapping, time_name_mapping)
        if is_complex(self.right):
            rstr = bwrap(rstr)
        return "%s ⇒ %s" % (lstr, rstr)

    def term_to_tex(self, var_name_mapping, time_name_mapping):
        lstr = self.left.term_to_tex(var_name_mapping, time_name_mapping)
        if is_complex(self.left):
            lstr = bwrap_tex(lstr)
        rstr = self.right.term_to_tex(var_name_mapping, time_name_mapping)
        if is_complex(self.right):
            rstr = bwrap_tex(rstr)
        return "%s \\Rightarrow %s" % (lstr, rstr)


class Disj(Term):

    def __init__(self, left: Term, right: Term):
        super().__init__()
        self.left = left
        self.right = right

    def assign_index(self, index: int):
        return Disj(self.left.assign_index(index), self.right.assign_index(index))

    def replace_var(self, var, term):
        return Disj(self.left.replace_var(var, term), self.right.replace_var(var, term))

    def replace_constant_label(self, label_from, label_to):
        return Disj(self.left.replace_constant_label(label_from, label_to),
                    self.right.replace_constant_label(label_from, label_to))

    def _skolem_to_existential_rec(self, sk, free_var, replace_skolem_bool):
        return Disj(self.left._skolem_to_existential_rec(sk, free_var, replace_skolem_bool),
                    self.right._skolem_to_existential_rec(sk, free_var, replace_skolem_bool))

    def beta_normal_form(self):
        return Disj(self.left.beta_normal_form(), self.right.beta_normal_form())

    def refresh_env(self, time, env=None):
        return Disj(self.left.refresh_env(time, env), self.right.refresh_env(time, env))

    def all_vars(self) -> List[Var]:
        return merge_var_lists(self.left.all_vars(), self.right.all_vars())

    def all_times(self):
        return self.left.all_times() | self.right.all_times()

    def all_skolems(self):
        return self.left.all_skolems() + self.right.all_skolems()

    def shift_var_start_by(self, shift: int) -> Term:
        return Disj(self.left.shift_var_start_by(shift), self.right.shift_var_start_by(shift))

    def filter_time(self, time_filter):
        return Disj(self.left.filter_time(time_filter), self.right.filter_time(time_filter))

    def term_to_str(self, var_name_mapping: Dict[Var, str], time_name_mapping) -> str:
        lstr = self.left.term_to_str(var_name_mapping, time_name_mapping)
        if is_complex(self.left) and not isinstance(self.left, Disj):
            lstr = bwrap(lstr)
        rstr = self.right.term_to_str(var_name_mapping, time_name_mapping)
        if is_complex(self.right) and not isinstance(self.right, Disj):
            rstr = bwrap(rstr)
        return "%s ∨ %s" % (lstr, rstr)

    def term_to_tex(self, var_name_mapping, time_name_mapping):
        lstr = self.left.term_to_tex(var_name_mapping, time_name_mapping)
        if is_complex(self.left) and not isinstance(self.left, Disj):
            lstr = bwrap_tex(lstr)
        rstr = self.right.term_to_tex(var_name_mapping, time_name_mapping)
        if is_complex(self.right) and not isinstance(self.right, Disj):
            rstr = bwrap_tex(rstr)
        return "%s \\lor %s" % (lstr, rstr)


class Conj(Term):

    def __init__(self, left: Term, right: Term):
        super().__init__()
        self.left = left
        self.right = right

    def assign_index(self, index: int):
        return Conj(self.left.assign_index(index), self.right.assign_index(index))

    def replace_var(self, var, term):
        return Conj(self.left.replace_var(var, term), self.right.replace_var(var, term))

    def replace_constant_label(self, label_from, label_to):
        return Conj(self.left.replace_constant_label(label_from, label_to),
                    self.right.replace_constant_label(label_from, label_to))

    def _skolem_to_existential_rec(self, sk, free_var, replace_skolem_bool):
        return Conj(self.left._skolem_to_existential_rec(sk, free_var, replace_skolem_bool),
                    self.right._skolem_to_existential_rec(sk, free_var, replace_skolem_bool))

    def beta_normal_form(self):
        return Conj(self.left.beta_normal_form(), self.right.beta_normal_form())

    def refresh_env(self, time, env=None):
        return Conj(self.left.refresh_env(time, env), self.right.refresh_env(time, env))

    def all_vars(self) -> List[Var]:
        return merge_var_lists(self.left.all_vars(), self.right.all_vars())

    def all_times(self):
        return self.left.all_times() | self.right.all_times()

    def all_skolems(self):
        return self.left.all_skolems() + self.right.all_skolems()

    def shift_var_start_by(self, shift: int) -> Term:
        return Conj(self.left.shift_var_start_by(shift), self.right.shift_var_start_by(shift))

    def filter_time(self, time_filter):
        return Conj(self.left.filter_time(time_filter), self.right.filter_time(time_filter))

    def term_to_str(self, var_name_mapping: Dict[Var, str], time_name_mapping) -> str:
        lstr = self.left.term_to_str(var_name_mapping, time_name_mapping)
        if is_complex(self.left) and not isinstance(self.left, Conj):
            lstr = bwrap(lstr)
        rstr = self.right.term_to_str(var_name_mapping, time_name_mapping)
        if is_complex(self.right) and not isinstance(self.right, Conj):
            rstr = bwrap(rstr)
        return "%s Λ %s" % (lstr, rstr)

    def term_to_tex(self, var_name_mapping, time_name_mapping):
        lstr = self.left.term_to_tex(var_name_mapping, time_name_mapping)
        if is_complex(self.left) and not isinstance(self.left, Conj):
            lstr = bwrap_tex(lstr)
        rstr = self.right.term_to_tex(var_name_mapping, time_name_mapping)
        if is_complex(self.right) and not isinstance(self.right, Conj):
            rstr = bwrap_tex(rstr)
        return "%s \\land %s" % (lstr, rstr)


class UQuant(Term):

    def __init__(self, var: Var, formula: Term):
        super().__init__()
        self.var = var
        self.formula = formula

    def assign_index(self, index: int):
        return UQuant(self.var, self.formula.assign_index(index))

    def replace_var(self, var, term):
        return UQuant(self.var.replace_var(var, term), self.formula.replace_var(var, term))

    def replace_constant_label(self, label_from, label_to):
        return UQuant(self.var,
                      self.formula.replace_constant_label(label_from, label_to))

    def beta_normal_form(self):
        return UQuant(self.var.beta_normal_form(), self.formula.beta_normal_form())

    def refresh_env(self, time, env=None):
        if env is None:
            env = Environment({self.var}, time)
        else:
            env = env.add_var(self.var)
        return UQuant(self.var, self.formula.refresh_env(time, env))

    def _skolem_to_existential_rec(self, sk, free_var, replace_skolem_bool):
        if self.var in sk.environments[0].vars and not replace_skolem_bool:
            return EQuant(free_var,
                          Conj(
                              Apply(sk.predicate, free_var).beta_normal_form(),
                              UQuant(self.var, self.formula._skolem_to_existential_rec(sk, free_var, True))))
        else:
            return UQuant(self.var, self.formula._skolem_to_existential_rec(sk, free_var, replace_skolem_bool))

    def all_vars(self) -> List[Var]:
        return merge_var_lists([self.var], self.formula.all_vars())

    def all_times(self):
        return self.formula.all_times()

    def all_skolems(self):
        return self.formula.all_skolems()

    def shift_var_start_by(self, shift: int) -> Term:
        return UQuant(self.var.shift_var_start_by(shift), self.formula.shift_var_start_by(shift))

    def filter_time(self, time_filter):
        return UQuant(self.var, self.formula.filter_time(time_filter))

    def term_to_str(self, var_name_mapping: Dict[Var, str], time_name_mapping) -> str:
        return "∀%s[%s]" % (self.var.term_to_str(var_name_mapping, time_name_mapping), self.formula.term_to_str(var_name_mapping, time_name_mapping))

    def term_to_tex(self, var_name_mapping, time_name_mapping):
        return "\\forall{}%s\\left[%s\\right]" % (self.var.term_to_tex(var_name_mapping, time_name_mapping), self.formula.term_to_tex(var_name_mapping, time_name_mapping))


class EQuant(Term):

    def __init__(self, var: Var, formula: Term):
        super().__init__()
        self.var = var
        self.formula = formula

    def assign_index(self, index: int):
        return EQuant(self.var, self.formula.assign_index(index))

    def replace_var(self, var, term):
        return EQuant(self.var.replace_var(var, term), self.formula.replace_var(var, term))

    def replace_constant_label(self, label_from, label_to):
        return EQuant(self.var,
                      self.formula.replace_constant_label(label_from, label_to))

    def beta_normal_form(self):
        return EQuant(self.var.beta_normal_form(), self.formula.beta_normal_form())

    def refresh_env(self, time, env=None):
        if env is None:
            env = Environment({self.var}, time)
        else:
            env = env.add_var(self.var)
        return EQuant(self.var, self.formula.refresh_env(time, env))

    def _skolem_to_existential_rec(self, sk, free_var, replace_skolem_bool):
        return EQuant(self.var, self.formula._skolem_to_existential_rec(sk, free_var, replace_skolem_bool))

    def all_vars(self) -> List[Var]:
        return merge_var_lists([self.var], self.formula.all_vars())

    def all_times(self):
        return self.formula.all_times()

    def all_skolems(self):
        return self.formula.all_skolems()

    def shift_var_start_by(self, shift: int) -> Term:
        return EQuant(self.var.shift_var_start_by(shift), self.formula.shift_var_start_by(shift))

    def filter_time(self, time_filter):
        return EQuant(self.var, self.formula.filter_time(time_filter))

    def term_to_str(self, var_name_mapping: Dict[Var, str], time_name_mapping) -> str:
        return "∃%s[%s]" % (self.var.term_to_str(var_name_mapping, time_name_mapping), self.formula.term_to_str(var_name_mapping, time_name_mapping))

    def term_to_tex(self, var_name_mapping, time_name_mapping):
        return "\\exists{}%s%s" % (self.var.term_to_tex(var_name_mapping, time_name_mapping), self.formula.term_to_tex(var_name_mapping, time_name_mapping))


class Environment:

    def __init__(self, vars: Set[Var], time):
        self.vars = vars
        self.time = time

    def add_var(self, v: Var):
        return Environment(self.vars | {v}, self.time)

    def shift_var_start_by(self, shift: int):
        return Environment({v.shift_var_start_by(shift) for v in self.vars}, self.time)

    def all_vars(self) -> List[Var]:
        return list(self.vars)

    def term_to_str(self, var_name_mapping: Dict[Var, str], time_name_mapping) -> str:
        return "{%s}%s" % (" ".join([x.term_to_str(var_name_mapping, time_name_mapping) for x in sorted(list(self.vars))]), self.time)

    def term_to_tex(self, var_name_mapping, time_name_mapping):
        return "\\{%s\\}^{%s}" % (
            "\\ ".join([x.term_to_tex(var_name_mapping, time_name_mapping) for x in sorted(list(self.vars))]),
            self.time)
        # return "\\{%s\\}^{%s}" % (
        #     "\\ ".join([x.term_to_tex(var_name_mapping, time_name_mapping) for x in sorted(list(self.vars))]),
        #     time_name_mapping[self.time])

    def __lt__(self, other):
        if self.time == other.time:
            return self.vars < other.vars
        else:
            return self.time < other.time

    def __eq__(self, other):
        return self.vars == other.vars  # and self.time == other.time

    def __hash__(self):
        return hash(self.vars)


class Skolem(Term):

    def __init__(self, predicate: Term, index: int, environments: List[Environment]):
        super().__init__()
        self.predicate = predicate
        self.index = index
        self.environments = environments

    def assign_index(self, index: int):
        return Skolem(self.predicate, index, self.environments)

    def replace_var(self, var, term):
        return Skolem(self.predicate.replace_var(var, term), self.index, self.environments)

    def replace_constant_label(self, label_from, label_to):
        return Skolem(self.predicate.replace_constant_label(label_from, label_to),
                      self.index,
                      self.environments)

    def beta_normal_form(self):
        return Skolem(self.predicate.beta_normal_form(), self.index, self.environments)

    def refresh_env(self, time, env=None):
        if env is None:
            env = Environment(vars=set(), time=time)
        if env in self.environments:
            return self
        else:
            return Skolem(self.predicate.refresh_env(time, env), self.index, self.environments + [env])

    def _skolem_to_existential_rec(self, sk, free_var, replace_skolem_bool):
        if sk.index == self.index and replace_skolem_bool:
            return free_var
        else:
            return self

    def all_vars(self) -> List[Var]:
        res = self.predicate.all_vars()
        for e in self.environments:
            res = merge_var_lists(res, e.all_vars())
        return res

    def all_times(self):
        return {e.time for e in self.environments}

    def all_skolems(self):
        return [self]+self.predicate.all_skolems()

    def shift_var_start_by(self, shift: int) -> Term:
        return Skolem(predicate=self.predicate.shift_var_start_by(shift),
                      index=self.index,
                      environments=[e.shift_var_start_by(shift) for e in self.environments])

    def filter_time(self, time_filter):
        my_time = time_filter[self.index]
        return Skolem(self.predicate.filter_time(time_filter), self.index, [[e for e in self.environments if e.time >= my_time][-1]])

    def term_to_str(self, var_name_mapping: Dict[Var, str], time_name_mapping) -> str:
        pred_str = self.predicate.term_to_str(var_name_mapping, time_name_mapping)
        index_str = str(self.index) if type(self.index) is int else ""
        env_str = " ".join([e.term_to_str(var_name_mapping, time_name_mapping) for e in self.environments])
        if index_str == "" and env_str == "":
            return "skolem(%s)" % pred_str
        else:
            return "skolem(%s)" % " ; ".join([pred_str, index_str, env_str])

    def term_to_tex(self, var_name_mapping, time_name_mapping):
        pred_str = self.predicate.term_to_tex(var_name_mapping, time_name_mapping)
        index_str = str(self.index) if type(self.index) is int else ""
        env_str = "\\ ".join([e.term_to_tex(var_name_mapping, time_name_mapping) for e in self.environments])
        return "\\operatorname{sk}_{%s\\ :\\  %s}^{%s}" % (index_str, pred_str, env_str)


################################### LAMBDA PARSER ###################################

_lark = lark.Lark('''
            start: _term
            %ignore " " // Disregard spaces in text
            _term: lambda
                 | equant
                 | uquant
                 | impl
                 | disj
                 | conj
                 | apply
                 | skolem
                 | constant
                 | "(" _term ")"
                 | "[" _term "]"
                 | neg
                 | VAR
            _non_apply_term: lambda     // needed for making apply rule left associative
                           | equant
                           | uquant
                           | impl
                           | disj
                           | conj
                           | skolem
                           | constant
                           | "(" _term ")"
                           | neg
                           | VAR
            lambda : "λ" VAR "."? _term
            uquant : "∀" VAR "."? _term
            equant : "∃" VAR "."? _term
            impl : _term "⇒" _term
            disj : _term "∨" _term
            conj : _term "Λ" _term
            neg: "¬" _term
            apply : _term "@"? _non_apply_term
            _CHAR: "a".."z" | "A".."Z" | "_" | "-" | "0" .. "9"
            env : "{" VAR* "}"
            skolem.5  : "skolem" "(" _term [";" INDEX ";" env+] ")"
            constant: LITERAL ["(" _term ("," _term)* ")"]
            INDEX: ("0".."9")+
            LITERAL: _CHAR+ "'"
            VAR: ("a".."z")+
         ''')


def _parse_convert(ast, symbol_table) -> Term:
    if isinstance(ast, lark.Token):
        if ast.type == "VAR":
            if ast.value not in symbol_table:
                symbol_table[ast.value] = Var(len(symbol_table))
            return symbol_table[ast.value]
        else:
            raise Exception("something is wrong with this parse")
    else:
        if ast.data == "constant":
            label = ast.children[0].value[:-1]
            args = [_parse_convert(c, symbol_table) for c in ast.children[1:]]
            return Constant(label, args)
        elif ast.data == "env":
            return {_parse_convert(c, symbol_table) for c in ast.children}
        elif ast.data == "skolem":
            pred = _parse_convert(ast.children[0], symbol_table)
            if len(ast.children) > 1:
                index = int(ast.children[1])
                envs = [_parse_convert(c, symbol_table) for c in ast.children[2:]]
            else:
                index = None
                envs = []
            return Skolem(pred, index, envs)
        new_children = [_parse_convert(c, symbol_table) for c in ast.children]
        if ast.data == "start":
            return new_children[0]
        elif ast.data == "disj":
            return Disj(new_children[0], new_children[1])
        elif ast.data == "conj":
            return Conj(new_children[0], new_children[1])
        elif ast.data == "impl":
            return Impl(new_children[0], new_children[1])
        elif ast.data == "apply":
            return Apply(new_children[0], new_children[1])
        elif ast.data == "neg":
            return Neg(new_children[0])
        elif ast.data == "lambda":
            return Lambda(new_children[0], new_children[1])
        elif ast.data == "uquant":
            return UQuant(new_children[0], new_children[1])
        elif ast.data == "equant":
            return EQuant(new_children[0], new_children[1])
        else:
            raise Exception("something is wrong with this parse")


def parse_lambda(s) -> Term:
    ast = _lark.parse(s)
    return _parse_convert(ast, dict())


####################################### TRANSFORMS #######################################

def reattach_relatives_to_N(node: Node):
    if node.is_unary:
        return node.comb(reattach_relatives_to_N(node.child))
    elif node.is_binary:
        new_left = reattach_relatives_to_N(node.left)
        new_right = reattach_relatives_to_N(node.right)
        if node.comb.is_B_bck and new_left.cat.is_np and new_right.cat.is_right_adj_cat and _is_relative_clause(node.right):
            relativizer = _replace_NP_with_N_in_relative(new_right)
            return _reattach_relativizer(new_left, relativizer)
        else:
            return node.comb(new_left, new_right)
    else:
        return Terminal(node.cat, node.word, node.pos)


def _reattach_relativizer(left_node, relativizer):
    N = Category.from_str("N")
    NP = Category.from_str("NP")
    if left_node.cat == N:
        return B(is_forward=False, order=0, is_crossed=False)(left_node, relativizer)
    elif left_node.cat==NP:
        if left_node.is_unary:
            return left_node.comb(_reattach_relativizer(left_node.child, relativizer))
        elif left_node.is_binary:
            if left_node.comb.is_punc_right:
                return _reattach_relativizer(left_node.left, Punc(True)(left_node.right, relativizer))
            else:
                return left_node.comb(left_node.left, _reattach_relativizer(left_node.right, relativizer))
        else:
            raise Exception("i don't know what to do here")
    else:
        raise Exception("I don't know how to find N")


def _replace_NP_with_N_in_relative(node):
    if node.is_binary:
        if node.comb.is_punc_left:
            return node.comb(node.left, _replace_NP_with_N_in_relative(node.right))
        elif node.comb.is_punc_right:
            return node.comb(_replace_NP_with_N_in_relative(node.left), node.right)
        else:
            return node.comb(_replace_NP_with_N_in_relative(node.left), node.right)
    elif node.is_unary:
        N_N = Category.from_str("N\\N")
        comb = TypeChanging1(node.child.cat, N_N)
        return comb(node.child)
    else:
        RCATL_NP = Category.from_str("S\\NP/(S/NP)")
        RCATR_NP = Category.from_str("S\\NP/(S\\NP)")
        RCATR_NP2 = Category.from_str("NP\\NP/(S\\NP)")
        RCATL_N = Category.from_str("S\\N/(S/NP)")
        RCATR_N = Category.from_str("S\\N/(S\\NP)")
        RCATR_N2 = Category.from_str("N\\N/(S\\NP)")
        if node.cat.strip_features() == RCATL_NP:
            return Terminal(RCATL_N, node.word, node.pos)
        elif node.cat.strip_features() == RCATR_NP:
            return Terminal(RCATR_N, node.word, node.pos)
        elif node.cat.strip_features() == RCATR_NP2:
            return Terminal(RCATR_N2, node.word, node.pos)
        else:
            raise Exception("unknown relativizer category %s"%node.cat)


def _is_relative_clause(node: Node):
    if node.is_binary:
        if node.comb.is_punc_left:
            return _is_relative_clause(node.right)
        elif node.comb.is_punc_right:
            return _is_relative_clause(node.left)
        else:
            lcat = node.left.cat
            RCATL = Category.from_str("S\\NP/(S/NP)")
            RCATR = Category.from_str("S\\NP/(S\\NP)")
            RCATR2 = Category.from_str("NP\\NP/(S\\NP)")
            NP_NP = Category.from_str("NP\\NP")
            if node.cat == NP_NP and node.left.is_term and (lcat.strip_features() in [RCATL, RCATR, RCATR2]):
                return True
            else:
                return False
    elif node.is_unary:
        NP_NP = Category.from_str("NP\\NP")
        if node.comb.is_type_change and node.cat == NP_NP:
            return True
        else:
            return False
    else:
        return False

################################### SEMANTICS ASSIGNER ###################################


_MEMO_ENGLISH_LEXICON = None


def _english_lexicon_table():
    global _MEMO_ENGLISH_LEXICON
    if _MEMO_ENGLISH_LEXICON is None:
        from os.path import realpath, dirname, join
        from os import getcwd
        res = dict()
        file_loc = join(realpath(join(getcwd(), dirname(__file__))), "semantics_mapping_english.txt")
        with open(file_loc) as fh:
            for line in fh:
                line = line.rstrip()
                if line.startswith("#") or line.strip()=='':
                    continue
                fields = [entry for entry in line.split("\t") if entry!='']
                cat = Category.from_str(fields[0])
                sem = parse_lambda(fields[-1])
                if len(fields) == 2:
                    res[cat] = sem
                elif len(fields) == 3:
                    words = fields[1].split(",")
                    for word in words:
                        res[(cat, word)] = sem
                else:
                    raise Exception("lexicon format is wrong for line: " + line)
        _MEMO_ENGLISH_LEXICON = res

    return _MEMO_ENGLISH_LEXICON


class SemanticsAssigner:

    def __init__(self, lang: str):
        self.lang = lang
        self.is_English = lang.lower().startswith("english")
        if self.is_English:
            self.lexicon = _english_lexicon_table()
            self.unary_det = self.lexicon[(Category.from_str("NP/N"), "a")]
            self.unary_that = self.lexicon[(Category.from_str("N\\N/(S/NP)"), "that")]
        else:
            raise Exception("only English is supported at the moment")

    def construct_formula_from_cat(self, cat: Category) -> Term:
        main, args = cat.cut_args()
        varss = [Var(var_id) for var_id in range(len(args))]
        if varss:
            lam = Constant("word", varss)
        else:
            lam = Constant("word", None)
        for v in varss:
            lam = Lambda(v, lam)
        return lam

    def _push_raising_downstream(self, node: Node, cat_from: Category, cat_raised: Category):
        if node.is_binary:
            if node.left.cat.is_adj_cat:
                self._push_raising_downstream(node.right, cat_from, cat_raised)
            elif node.right.cat.is_adj_cat:
                self._push_raising_downstream(node.left, cat_from, cat_raised)
            elif node.comb.is_B0_fwd:
                self._push_raising_downstream(node.left, cat_from, cat_raised)
            elif node.comb.is_B0_bck:
                self._push_raising_downstream(node.right, cat_from, cat_raised)
            elif node.comb.is_conj_top:
                self._push_raising_downstream(node.left, cat_from, cat_raised)
                self._push_raising_downstream(node.right, cat_from, cat_raised)
            elif node.comb.is_conj_bottom:
                self._push_raising_downstream(node.right, cat_from, cat_raised)
            elif node.comb.is_punc_left:
                self._push_raising_downstream(node.right, cat_from, cat_raised)
            elif node.comb.is_punc_right:
                self._push_raising_downstream(node.left, cat_from, cat_raised)
            else:
                raise Exception("I didn't count on this")
        elif node.is_unary:
            if node.cat == cat_from:
                node.raised_to = cat_raised
            else:
                raise Exception("I didn't count on this")
        else:
            node.raised_to = self._sub_cat_raise(node.cat, cat_from, cat_raised)

    # noinspection PyArgumentList
    def _sub_cat_raise(self, cat: Category, cat_from: Category, cat_raised: Category):
        if cat_from == cat:
            return cat_raised
        elif cat.is_functor:
            return Functor(cat.slash_is_fwd, self._sub_cat_raise(cat.res, cat_from, cat_raised), cat.arg)
        else:
            raise Exception("I didn't count on this")

    # noinspection PyArgumentList
    def _assign_raised_over(self, node: Node):
        if node.is_binary:
            lcat = node.left.cat
            rcat = node.right.cat
            pcat = node.cat
            comb = node.comb
            if (lcat.is_pp or lcat.is_np) and rcat.is_verbal and comb.is_B0_bck:
                prop_node, cat_from, cat_raised = node.left, lcat, Functor(True, pcat, rcat)
                # self._push_raising_downstream(node.left, cat_from, cat_raised)
                self._push_raising_downstream(prop_node, cat_from.strip_features(), cat_raised.strip_features())
            elif (rcat.is_pp or rcat.is_np) and lcat.is_verbal and comb.is_B0_fwd:
                prop_node, cat_from, cat_raised = node.right, rcat, Functor(False, pcat, lcat)
                # cat_from, cat_raised = cat_from.strip_features(), cat_raised.strip_features()
                # self._push_raising_downstream(node.right, cat_from, cat_raised)
                self._push_raising_downstream(prop_node, cat_from.strip_features(), cat_raised.strip_features())
        elif node.is_unary and node.comb.is_type_raise and node.comb.is_order_preserving:
            prop_node, cat_from, cat_raised = node.child, node.child.cat, node.cat
            # cat_from, cat_raised = cat_from.strip_features(), cat_raised.strip_features()
            # self._push_raising_downstream(node.child, cat_from, cat_raised)
            self._push_raising_downstream(prop_node, cat_from.strip_features(), cat_raised.strip_features())


        for c in node.children:
            self._assign_raised_over(c)

    def _is_and_conj(self, node: Node):
        """ whether it is 'and' or 'or' """
        if node.is_term:
            return node.word=="and" or node.word==","
        elif node.is_binary:
            if node.comb.is_punc_left:
                self._is_and_conj(node.right)
            elif node.comb.is_punc_right:
                self._is_and_conj(node.left)
            else:
                raise Exception("I didn't count on this")
        else:
            raise Exception("I didn't count on this")

    def _find_noun_index(self, node: Node):
        if node.is_term:
            return node.pos
        elif node.is_binary:
            if node.left.cat.is_adj_cat:
                return self._find_noun_index(node.right)
            elif node.right.cat.is_adj_cat:
                return self._find_noun_index(node.left)
            elif node.comb.is_punc_right:
                return self._find_noun_index(node.left)
            elif node.comb.is_punc_left:
                return self._find_noun_index(node.right)
            else:
                raise Exception("i didn't count on this")
        else:
            raise Exception("i didn't count on this")

    def _assign_semantics_rec(self, node: Node):
        for c in node.children:
            self._assign_semantics_rec(c)
        time = node.gorn
        if node.is_term:
            w = node.word.lower()
            c = node.cat
            if hasattr(node, 'raised_to') and (node.raised_to, w) in self.lexicon:
                f = self.lexicon[(node.raised_to, w)]
            elif (c, w) in self.lexicon:
                f = self.lexicon[(c, w)]
            elif c in self.lexicon:
                f = self.lexicon[c]
            else:
                f = self.construct_formula_from_cat(c)
            if f is not None:
                f = f.assign_index(node.pos).replace_constant_label("word", w).refresh_env(time)
            semantics = f
        elif node.is_unary:
            if node.child.semantics is None:
                semantics = None
            elif node.comb.is_type_raise:
                semantics = node.child.semantics  # do nothing because NP and PP sem is already raised
            elif node.comb.is_type_change:
                cat_from = node.comb.cat_from
                cat_to = node.comb.cat_to
                S_NP = Category.from_str("S\\NP")
                Sto_NP = Category.from_str("S[to]\\NP")
                Sdcl_NP = Category.from_str("S[dcl]/NP")
                NP_NP = Category.from_str("NP\\NP")
                N_N = Category.from_str("N\\N")
                if cat_from.is_n and cat_to.is_np:
                    det_sem = self.unary_det.assign_index(self._find_noun_index(node.child))
                    semantics = b0_logic(det_sem, node.child.semantics, time)
                elif (cat_from.strip_features() == S_NP or cat_from == Sdcl_NP) and cat_to == N_N:
                    that_sem = self.unary_that
                    semantics = b0_logic(that_sem, node.child.semantics, time)
                # elif (cat_from.strip_features()==S_NP or cat_from==Sdcl_NP) and cat_to==NP_NP:
                #     semantics = None
                #     # semantics = node.child.semantics
                # elif cat_from == Sto_NP and cat_to == N_N:
                #     that_sem = self.unary_that
                #     semantics = b0_logic(that_sem, node.child.semantics, time)
                else:
                    semantics = None
                    # raise NotImplementedError()  # TODO
            elif node.comb.is_unary_coord:
                semantics = coord_logic(node.child.semantics, True, time)
            elif node.comb.is_XY_to_ZY_change:
                raise NotImplementedError()  # TODO
            elif node.comb.is_X_to_XX_change:
                raise NotImplementedError()  # TODO
            else:
                raise Exception("you should not get here")
        else:
            lsem = node.left.semantics
            rsem = node.right.semantics
            lcat = node.left.cat
            rcat = node.right.cat
            if not node.comb.is_punc and (lsem is None or rsem is None):
                semantics = None
            elif node.comb.is_glue:
                semantics = None
            elif node.comb.is_B_bck and lcat.is_np and rcat.is_right_adj_cat:
                #  N N\N --> N
                semantics = None
                # [lsem, rsem] = alpha_transform_expressions([lsem, rsem])
                # llambs, lcore = strip_lambda_args(lsem)
                # core = Conj(lcore, Apply(rsem, lcore))
                # for lamb in reversed(llambs):
                #     core = Lambda(lamb, core)
                # semantics = core.beta_normal_form()
            elif node.comb.is_B_bck:
                if node.comb.order == 0:
                    if lcat.is_np or lcat.is_pp:
                        semantics = b0_logic(lsem, rsem, time)
                    else:
                        semantics = b0_logic(rsem, lsem, time)
                elif node.comb.order == 1:
                    semantics = b1_logic(rsem, lsem, time)
                elif node.comb.order == 2:
                    semantics = b2_logic(rsem, lsem, time)
                else:
                    raise Exception("I didn't expect this")
            elif node.comb.is_B_fwd:
                if node.comb.order == 0:
                    if rcat.is_np or rcat.is_pp:
                        semantics = b0_logic(rsem, lsem, time)
                    else:
                        semantics = b0_logic(lsem, rsem, time)
                elif node.comb.order == 1:
                    semantics = b1_logic(lsem, rsem, time)
                elif node.comb.order == 2:
                    semantics = b2_logic(lsem, rsem, time)
                else:
                    raise Exception("I didn't expect this")
            elif node.comb.is_punc:
                if node.comb.punc_is_left:
                    semantics = node.right.semantics
                else:
                    semantics = node.left.semantics
            elif node.comb.is_conj_top:
                semantics = b0_logic(rsem, lsem, time)
            elif node.comb.is_conj_bottom:
                semantics = coord_logic(node.right.semantics, self._is_and_conj(node.left), time)
            elif node.comb.is_tc_X_Y_to_Xconj:
                semantics = coord_logic(node.left.semantics, True, time)
            elif node.comb.is_tc_X_Y_to_Yconj:
                semantics = coord_logic(node.right.semantics, True, time)
            elif node.comb.is_tc_A_B_to_A:
                semantics = node.left.semantics  # TODO glue sems
            elif node.comb.is_tc_A_B_to_B:
                semantics = node.right.semantics  # TODO glue sems
            elif node.comb.is_tc_A_XY_to_ZY:
                raise NotImplementedError()  # TODO
            elif node.comb.is_tc_XY_A_to_ZY:
                raise NotImplementedError()  # TODO
            elif node.comb.is_type_change_binary:
                raise NotImplementedError()  # TODO
            else:
                raise Exception("you should not get here")
        node.semantics = semantics
        return node.semantics

    def assign_semantics(self, node: Node):
        if hasattr(node, 'semantics'):
            return node
        else:
            new_node = reattach_relatives_to_N(node)
            self._assign_raised_over(new_node)
            self._assign_gorn(new_node, 't')
            self._assign_semantics_rec(new_node)
            return new_node

    def _assign_gorn(self, node: Node, gorn: str):
        node.gorn = gorn
        if node.is_term:
            return
        elif node.is_unary:
            self._assign_gorn(node.child, gorn+'d')
        else:
            self._assign_gorn(node.left , gorn+'l')
            self._assign_gorn(node.right, gorn+'r')


from .combinators import Conj as CConj

tr1 = TypeRaising(cat_res=Category.from_str("S"), cat_arg=Category.from_str("NP"), is_forward=True)
tr2 = TypeRaising(cat_res=Category.from_str("S\\NP"), cat_arg=Category.from_str("NP"), is_forward=False)
tc_np_n = TypeChanging1(Category.from_str("N"), Category.from_str("NP"))
bx1f = B(is_forward=True, is_crossed=True, order=1)
bx1b = B(is_forward=False, is_crossed=True, order=1)
b1f = B(is_forward=True, is_crossed=False, order=1)
b0f = B(is_forward=True, is_crossed=False, order=0)
b0b = B(is_forward=False, is_crossed=False, order=0)
pr = Punc(punc_is_left=False)
conj_top = CConj(is_bottom=False)
conj_bottom = CConj(is_bottom=True)
sem_assigner = SemanticsAssigner("English")


def test_semantics_1():
    w1 = Terminal(Category.from_str("NP/N"), "Every")
    w2 = Terminal(Category.from_str("N"), "man")
    w3 = Terminal(Category.from_str("conj"), "and")
    w4 = Terminal(Category.from_str("NP/N"), "every")
    w5 = Terminal(Category.from_str("N"), "woman")
    w6 = Terminal(Category.from_str("S\\NP/NP"), "loves")
    w7 = Terminal(Category.from_str("NP/N"), "some")
    w8 = Terminal(Category.from_str("N"), "animal")

    tree = b0b(conj_top(b0f(w1, w2), conj_bottom(w3, b0f(w4, w5))), b0f(w6, b0f(w7, w8)))
    tree.assign_word_positions()
    tree = sem_assigner.assign_semantics(tree)

    for sem in tree.semantics.unpack_all_readings():
        print(sem)

    print(tree.semantics)


def test_relativization_rebranching():
    w1 = Terminal(Category.from_str("NP/N"), "Every")
    w2 = Terminal(Category.from_str("N"), "man")
    w3 = Terminal(Category.from_str("S\\NP/NP"), "loves")
    w4 = Terminal(Category.from_str("NP/N"), "a")
    w5 = Terminal(Category.from_str("N"), "woman")
    w6 = Terminal(Category.from_str("(NP\\NP)/(S[dcl]\\NP)"), "who")
    w7 = Terminal(Category.from_str("S\\NP/NP"), "read")
    w8 = Terminal(Category.from_str("NP/N"), "a")
    w9 = Terminal(Category.from_str("N"), "book")

    tree = b0b(b0f(w1, w2), b0f(w3, b0b(b0f(w4, w5), b0f(w6, b0f(w7, b0f(w8, w9))))))

    tree.assign_word_positions()
    tree2 = sem_assigner.assign_semantics(tree)
    tree2.visualize_proof()
    for r in tree2.semantics.unpack_all_readings():
        print(r)


def print_most_frequent_cats():
    tags = dict()
    weird = Category.from_str("NP\\NP/NP")
    for i, tree in enumerate(DerivationLoader.iter_from_file("/home/milos/PycharmProjects/ccg_toolz/data/ccgbank_split/train.ccg")):
        for term in tree.iter_terminals():
            tags[term.cat] = tags.get(term.cat, 0)+1
            if term.cat == weird:
                print("pause")
        print(i)
    for tag, count in sorted(tags.items(), key=lambda x: x[1]):
        print("%s %d" % (tag, count))

# print_most_frequent_cats()
# test_semantics_1()
# test_relativization_rebranching()
