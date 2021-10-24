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
    return sum((Counter(xs) & Counter(ys)).values())


def sufficient_stats_from_deps(gold_deps, pred_deps):
    stats = dict()
    for name, fun in METRICS:
        gold = [fun(d) for d in gold_deps]
        pred = [fun(d) for d in pred_deps]
        stats[f"{name}_overlap"] = overlap(gold, pred)
        stats[f"{name}_gold_count"] = len(gold)
        stats[f"{name}_pred_count"] = len(pred)
    return stats


def sufficient_stats(gold_tree, pred_tree, language="English"):
    return sufficient_stats_from_deps(
        gold_deps=gold_tree.deps(lang=language),
        pred_deps=pred_tree.deps(lang=language)
    )


METRICS = {
    ("labeled_dep" , lambda d: (d.head_cat, d.dep_slot, d.head_pos, d.dep_pos)),
    ("undirected_unlabeled_dep", lambda d: (min(d.head_pos, d.dep_pos), max(d.head_pos, d.dep_pos))),
    ("unlabeled_dep", lambda d: (d.head_pos, d.dep_pos)),
    ("stag", lambda d: d.head_cat)
}


def combine_stats(stats):
    stats = list(stats)
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


def evaluate(gold_trees, pred_trees, language):
    gold_deps = [x.deps(lang=language) for x in gold_trees]
    pred_deps = [x.deps(lang=language) for x in pred_trees]
    assert len(gold_deps) == len(pred_deps), "the number of trees differ"
    return combine_stats(sufficient_stats_from_deps(g, p) for g, p in zip(gold_deps, pred_deps) if g and p)


def _main_evaluate():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--lang", default="English")
    parser.add_argument("--include-generate", dest='include_generate', default=False, action='store_true')
    args = parser.parse_args()
    import ccg

    gold_trees = list(ccg.open(args.gold))
    pred_trees = list(ccg.open(args.pred))

    result = evaluate(gold_trees, pred_trees, args.lang)

    def num_to_str(res, name, metric):
        decimals = 1
        return str(round(res[f"{name}_{metric}"], decimals))

    table = []
    for orig_name, new_name in [('labeled_dep', "deps labeled"),
                                ('unlabeled_dep', "deps unlabeled"),
                                ('undirected_unlabeled_dep', "deps unlabeled undirected"),
                                ('stag', "super-tagging")]:
        table.append({
            'P' : num_to_str(result, orig_name, "P"),
            'R' : num_to_str(result, orig_name, "R"),
            'F' : num_to_str(result, orig_name, "F"),
            'eval' : new_name
        })
        if orig_name == "stag":
            table[-1]["F"] = ""
            table[-1]["R"] = ""

    if args.include_generate:
        from ccg.generate_gr_deps import evaluate as evaluate_generate
        gen_result = evaluate_generate(gold_trees, pred_trees)
        for orig_name, new_name in [('labeled_dep', "generate labeled"),
                                    ('unlabeled_dep', "generate unlabeled")]:
            table.append({
                'P': num_to_str(gen_result, orig_name, "P"),
                'R': num_to_str(gen_result, orig_name, "R"),
                'F': num_to_str(gen_result, orig_name, "F"),
                'eval' : new_name
            })

    headers = ["eval", "P", "R", "F"]
    table = [[entry[h] for h in headers] for entry in table]

    import tabulate
    print(tabulate.tabulate(table, headers=headers, tablefmt="fancy_grid", disable_numparse=True))
