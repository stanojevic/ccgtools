
from .category import Category
from .visualization import DepsDesc
from .predarg import PredArgAssigner

_MEMO_CANDC_RULES = None
_MEMO_CANDC_WORD_GROUPS = None


class _Rule:

    def __init__(self, label, ccg_slot, fields, constraint):
        self.label = label
        self.ccg_slot = ccg_slot
        self.fields = fields
        self.constraint = constraint

    def __call__(self, orig_dep, orig_deps_map):
        if self.label == "ignore":
            return []
        if isinstance(self.constraint, str) and orig_dep.head_word not in _MEMO_CANDC_WORD_GROUPS[self.constraint]:
            return None
        if isinstance(self.constraint, Category):
            xs = orig_deps_map[orig_dep.dep_pos]
            if xs and xs[0].head_cat != self.constraint:
                return None

        fields = []
        for x in self.fields:
            if x == "%l":
                fields.append(orig_dep.head_pos)
            elif x == "%f":
                fields.append(orig_dep.dep_pos)
            else:
                fields.append(x)
        res = [fields]
        for special_label in ["%1", "%2", "%3", "%c", "%k"]:
            old_res = res
            res = []
            for d in old_res:
                if special_label in d:
                    slot = [i for i, x in enumerate(d) if x==special_label][0]
                    if special_label in ["%1", "%2", "%3"]:
                        i = int(special_label[1:])
                        for dep_pos in [x.dep_pos for x in orig_deps_map[orig_dep.head_pos] if x.dep_slot==i]:
                            d2 = d.copy()
                            d2[slot] = dep_pos
                            res.append(d2)
                    else:
                        for dep_pos in [x.dep_pos for x in orig_deps_map[orig_dep.dep_pos]]:
                            d2 = d.copy()
                            d2[slot] = dep_pos
                            res.append(d2)
                else:
                    res.append(d)
        return [[self.label]+x for x in res]


def _get_candc_rules():
    global _MEMO_CANDC_RULES
    global _MEMO_CANDC_WORD_GROUPS
    if _MEMO_CANDC_RULES is None:
        _MEMO_CANDC_RULES = dict()
        _MEMO_CANDC_WORD_GROUPS = dict()
        from os.path import realpath, dirname, join
        from os import getcwd
        file_loc = join(realpath(join(getcwd(), dirname(__file__))), "gr_candc_rules.txt")
        curr_cat = None
        curr_rules = []
        with open(file_loc) as fh:
            for line in fh:
                line = line.split("#")[0]  # removing comments
                line = line.rstrip()
                if not line or "{" in line:
                    continue
                elif line.startswith("="):
                    fields = line.split(" ")
                    group = fields[0][1:]
                    words = set(fields[1:])
                    if group not in _MEMO_CANDC_WORD_GROUPS:
                        _MEMO_CANDC_WORD_GROUPS[group] = set()
                    _MEMO_CANDC_WORD_GROUPS[group].update(words)
                elif line.startswith("  "):
                    # NEW RULE
                    fields = line[2:].split()
                    ccg_slot = int(fields.pop(0))
                    gr_label = fields.pop(0)
                    if gr_label.startswith("("):
                        gr_label = gr_label[1:]
                    if fields and fields[-1].startswith("="):
                        const = fields.pop()
                        const = const[1:]
                        if const not in _MEMO_CANDC_WORD_GROUPS:
                            const = Category.from_str(const)
                    else:
                        const = None
                    curr_rules.append(_Rule(gr_label, ccg_slot, fields, const))
                else:
                    # NEW CAT
                    if curr_rules:
                        for rule in curr_rules:
                            key = (curr_cat, rule.ccg_slot)
                            if key not in _MEMO_CANDC_RULES:
                                _MEMO_CANDC_RULES[key] = []
                            _MEMO_CANDC_RULES[key].append(rule)
                        curr_rules = []
                    curr_cat = Category.from_str(line)
    return _MEMO_CANDC_RULES


def convert_ccg_to_gr_deps(deps):
    if not deps:
        return []
    all_rules = _get_candc_rules()
    word_count = max([max(x.dep_pos, x.head_pos) for x in deps])+1
    deps_map = []
    for i in range(word_count):
        deps_map.append([])
    for dep in deps:
        deps_map[dep.head_pos].append(dep)
    res = []
    for dep in deps:
        rules = all_rules.get((dep.head_cat, dep.dep_slot), [])
        for rule in rules:
            dep_new = rule(dep, deps_map)
            if dep_new is not None:
                res.extend(dep_new)
                break
    return {tuple(d) for d in res}


def convert_gr_to_DepsDesc(gr_deps, words):
    deps = []
    for d in gr_deps:
        if d[0] == "aux":
            # aux %% %%
            deps.append((d[1], d[2], "aux", "solid"))
        elif d[0] == "ccomp":
            if d[1] == "_":
                # ccomp _ %% %%
                deps.append((d[2], d[3], "ccomp", "solid"))
            else:
                # ccomp %% %% %%
                deps.append((d[2], d[1], "ccomp-a", "solid"))
                deps.append((d[1], d[3], "ccomp-b", "solid"))
        elif d[0] == "cmod":
            if d[1] == "_":
                # cmod _ %% %%
                deps.append((d[2], d[3], "cmod", "solid"))
            else:
                # cmod %% %% %%
                deps.append((d[2], d[1], "cmod-a", "solid"))
                deps.append((d[1], d[3], "cmod-b", "solid"))
        elif d[0] == "csubj":
            # csubj %% %% _
            deps.append((d[1], d[2], "csubj", "solid"))
        elif d[0] == "det":
            # det %% %%
            deps.append((d[1], d[2], "det", "solid"))
        elif d[0] == "dobj":
            if d[1] == "_":
                # dobj _ %% %%
                deps.append((d[2], d[3], "dobj_", "solid"))
            else:
                # dobj %% %%
                deps.append((d[1], d[2], "dobj", "solid"))
        elif d[0] == "iobj":
            if d[1] == "_":
                # iobj _ %% %%
                deps.append((d[2], d[3], "iobj_", "solid"))
            elif len(d) == 3:
                # iobj %% %%
                deps.append((d[1], d[2], "iobj", "solid"))
            else:
                # iobj %% %% %%
                deps.append((d[2], d[1], "iobj-a", "solid"))
                deps.append((d[1], d[3], "iobj-b", "solid"))
        elif d[0] == "ncmod":
            x = "_" if d[1] == "_" else "-"+d[1]
            deps.append((d[2], d[3], "ncmod"+x, "solid"))
        elif d[0] == "ncsubj":
            if d[1] == "_":
                # ncsubj _ %% %%
                deps.append((d[2], d[3], "ncsubj_", "solid"))
            elif len(d) == 3:
                # ncsubj %% %%
                deps.append((d[1], d[2], "ncsubj", "solid"))
            else:
                # ncsubj %% %% _
                # ncsubj %% %% inv
                # ncsubj %% %% obj
                deps.append((d[1], d[2], "ncsubj-"+d[3], "solid"))
        elif d[0] == "obj2":
            # obj2 %% %%
            deps.append((d[1], d[2], "obj2", "solid"))
        elif d[0] == "poss":
            # poss %% %%
            deps.append((d[1], d[2], "poss", "solid"))
        elif d[0] == "xcomp":
            if d[1] == "_":
                # xcomp _ %% %%
                deps.append((d[2], d[3], "xcomp", "solid"))
            else:
                # xcomp %% %% %%
                deps.append((d[2], d[1], "xcomp-a", "solid"))
                deps.append((d[1], d[3], "xcomp-b", "solid"))
        elif d[0] == "xmod":
            if d[1] == "_":
                # xmod _ %% %%
                deps.append((d[2], d[3], "xmod", "solid"))
            else:
                # xmod %% %% %%
                deps.append((d[2], d[1], "xmod-a", "solid"))
                deps.append((d[1], d[3], "xmod-b", "solid"))
        elif d[0] == "xsubj":
            if d[3] == "_":
                # xsubj %% %% _
                deps.append((d[1], d[2], "xmod", "solid"))
            else:
                # xsubj %% %% inv
                deps.append((d[1], d[2], "xmod-inv", "solid"))
        else:
            raise Exception("unknown "+str(d))

    return DepsDesc(words, deps, starting_position=0)


def simplify_DepsDesc(depsdesc):
    return DepsDesc(depsdesc.words,
                    [(a, b, c.replace("_", "").rstrip("-"), d) for a, b, c, d in depsdesc.deps],
                    starting_position=depsdesc.starting_position)


def tree_to_gr_deps(tree, lang="English"):
    pa = PredArgAssigner(lang=lang, include_conj_term=False)
    orig_deps = pa.all_deps(tree)
    return convert_ccg_to_gr_deps(orig_deps)


def tree_to_DepsDesc(tree, lang="English"):
    return convert_gr_to_DepsDesc(tree_to_gr_deps(tree, lang), tree.words())


def tree_to_simple_DepsDesc(tree, lang="English"):
    return simplify_DepsDesc(tree_to_DepsDesc(tree, lang))


def candc_eval(gold_trees, pred_trees, lang="English"):
    gold_grs = [tree_to_gr_deps(x) for x in gold_trees]
    pred_grs = [tree_to_gr_deps(x) for x in pred_trees]

    # TODO

    gold_gr = set(gold_gr)
    pred_gr = set(pred_gr)
    # TODO
    raise NotImplemented()