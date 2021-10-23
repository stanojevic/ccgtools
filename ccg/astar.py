import ccg
import numpy as np
import jpype as jp
# import jpype.imports
from jpype.types import *
from os.path import realpath, dirname, join
from os import getcwd

if not jp.isJVMStarted():

    EASYCCG_JAR = join(realpath(join(getcwd(), dirname(__file__))), "astar.jar")
    jp.startJVM(convertStrings=True, classpath=[EASYCCG_JAR])

MainSearch = jp.JPackage("astar").MainSearch
ArrayList = jp.JPackage("java").util.ArrayList


class AStarSearch:

    def __init__(self,
                 ordered_cats,
                 max_steps : int = 1_000_000,
                 prune_beta : float = 0.0001,
                 prune_top_k_tags : int = 50,
                 unary_prob : float = 0.9,
                 puncconj_prob : float = 0.99,
                 num_cpus : int = None,
                 use_normal_form : bool = True):
        assert ordered_cats != []
        assert issubclass(type(ordered_cats[0]), ccg.categories.Category)
        self.prune_beta = prune_beta
        self.prune_top_k_tags = prune_top_k_tags
        if num_cpus is None:
            num_cpus = 0
        self.astar = MainSearch(
            ArrayList([str(x) for x in ordered_cats]),
            max_steps,
            num_cpus,
            unary_prob,
            puncconj_prob,
            use_normal_form
        )

    def parse_batch(self, sents, batch_tag_logprobs, batch_span_logprobs=None):
        """
        :param List[List[str]] sents: words of sentence
        :param numpy.ndarray batch_tag_logprobs: matrix of shape (batch, words, tags) and type np.float
        :param numpy.ndarray batch_span_logprobs: matrix of shape (batch, words, words) and type np.float
        """

        k = min(batch_tag_logprobs.shape[-1], self.prune_top_k_tags)
        batch_tag_indices = np.argpartition(batch_tag_logprobs, -k, axis=-1)[:, :, -k:]
        batch_tag_logprobs = np.take_along_axis(batch_tag_logprobs, batch_tag_indices, axis=-1)
        batch_thresholds = np.max(batch_tag_logprobs, axis=-1) + np.log(self.prune_beta)

        b_words = ArrayList([ArrayList(x) for x in sents])
        b_tag_indices = JArray.of(batch_tag_indices)
        b_tag_logprobs = JArray.of(batch_tag_logprobs.astype(np.float64))
        b_thresholds = JArray.of(batch_thresholds.astype(np.float64))

        if batch_span_logprobs is None:
            b_span_logprobs = MainSearch.emptySpans(len(sents))
        else:
            b_span_logprobs = JArray.of(batch_span_logprobs.astype(np.float64))

        future = self.astar.searchBatchFuture(b_words,
                                              b_tag_indices,
                                              b_tag_logprobs,
                                              b_thresholds,
                                              b_span_logprobs)

        return LazyFuture(future)


class LazyFuture:

    def __init__(self, future):
        self.future = future

    def __iter__(self):
        for tree_str in list(self.future.get()):
            yield ccg.derivation(tree_str)


if __name__ == "__main__":
    import numpy as np
    print("start")
    tags = [ccg.category(x) for x in ["NP", r"S\NP/NP", "NP/N", "N"]]
    search = AStarSearch(tags)
    sents = ["John loves the apples".split()]
    b = len(sents)
    l = max(len(x) for x in sents)
    t = len(tags)
    batch_tag_logprobs = np.log(np.random.random((b, l, 1)))
    # batch_tag_indices = np.ones((b, l, 1), dtype=int)
    # batch_tag_indices = np.array([[[0], [1], [2], [3]]], dtype=int)
    batch_tag_indices = np.arange(4).reshape(1, -1, 1)
    batch_thresholds = np.full((b, l), -np.inf)
    batch_span_logprobs = None
    res = search.parse_batch(sents, batch_tag_logprobs, batch_span_logprobs)
    res = list(res)
    res[0].visualize()
    print(search)
    print("done")
