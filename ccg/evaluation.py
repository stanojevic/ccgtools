from .predarg import PredArgAssigner, DepLink
from collections import Counter


def _safe_divide(x: float, y: float):
    if x == 0.0 and y == 0.0:
        return 1.0
    elif x != 0.0 and y == 0.0:
        return 0.0
    else:
        return x/y


def f_score(p: float, r: float):
    if p == 0.0 and r == 0.0:
        return 0.0
    else:
        return 2*p*r/(p+r)


def prf_score_from_stats(overlap: int, gold_count: int, pred_count: int):
    p = _safe_divide(overlap, pred_count)*100
    r = _safe_divide(overlap, gold_count)*100
    f = f_score(p, r)
    return p, r, f


def overlap(xs, ys):
    return sum((Counter(x) & Counter(y)).values())


def _to_stag(d : DepLink):
    return d.head_cat

def _to_directed_labelled_dep(d : DepLink):
    return d.head_cat, d.dep_slot, d.head_pos, d.dep_pos


def _to_undirected_unlabelled_dep(d : DepLink):
    return min(d.head_pos, d.dep_pos), max(d.head_pos, d.dep_pos)


def sufficient_stats(gold_tree, pred_tree, language):
    stats = dict()

    predarg_assigner = PredArgAssigner(language, include_conj_term=False)

    gold_deps = predarg_assigner.all_deps(gold_tree)
    pred_deps = predarg_assigner.all_deps(pred_tree)

    for name, fun in METRICS:
        gold = [fun(d) for d in gold_deps]
        pred = [fun(d) for d in pred_deps]
        stats[f"{name}_overlap"] = overlap(gold, pred)
        stats[f"{name}_gold_count"] = len(gold)
        stats[f"{name}_pred_count"] = len(pred)

    return stats


METRICS = {
    "labelled_dep" : _to_directed_labelled_dep,
    "undirected_unlabelled_dep": _to_undirected_unlabelled_dep,
    "stag": _to_stag
}


def combine_stats(stats):
    all_scores = dict()
    for name, fun in METRICS:
        overlap = sum(x[f"{name}_overlap"] for x in stats)
        gold_count = sum(x[f"{name}_gold_count"] for x in stats)
        pred_count = sum(x[f"{name}_pred_count"] for x in stats)
        p, r, f = prf_score_from_stats(overlap, gold_count, pred_count)
        all_scores[f"{name}_P"] = p
        all_scores[f"{name}_R"] = r
        all_scores[f"{name}_F"] = f
    return all_scores
