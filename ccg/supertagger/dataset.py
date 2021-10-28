from torch.utils.data import Dataset, IterableDataset, ConcatDataset, Sampler
import random
from sys import stderr

import ccg
from ccg.combinators import Punc
from .any2int import Any2Int


class TreeTrainingBatchSampler(Sampler):

    def __init__(self, data_lengths, batch_size_in_words, out_of_order_noise: float = 10_000):
        super().__init__(None)
        self.batch_size = batch_size_in_words//25
        self.batch_size_in_words = batch_size_in_words
        self.out_of_order_noise = out_of_order_noise
        self.lens = data_lengths
        self.indices = list(range(len(data_lengths)))

    def __iter__(self):
        noisy_lens = [x + random.uniform(-self.out_of_order_noise, self.out_of_order_noise) for x in self.lens]
        new_indices = sorted(self.indices, key=lambda x: noisy_lens[x])
        batches = []
        curr_batch_size_in_sents = 0
        curr_batch_max_sent_len = 0
        curr_batch = []
        for index in new_indices:
            curr_batch_max_sent_len = max([self.lens[index], curr_batch_max_sent_len])
            curr_batch_size_in_sents += 1
            curr_batch.append(index)
            if curr_batch_size_in_sents*curr_batch_max_sent_len > self.batch_size_in_words:
                batches.append(curr_batch)
                curr_batch = []
                curr_batch_size_in_sents = 0
                curr_batch_max_sent_len = 0
        if curr_batch != []:
            batches.append(curr_batch)
        random.shuffle(batches)
        return iter(batches)


def reattach_punc_top_left(node):
    if node.is_term:
        return node
    elif node.is_unary:
        child = reattach_punc_top_left(node.child)
        if child.is_binary and child.comb.is_punc:
            if child.comb.is_punc_left:
                return Punc(True)(child.left, node.comb(child.right))
            else:
                return Punc(False)(node.comb(child.left), child.right)
        else:
            return node.comb(child)
    else:
        left = reattach_punc_top_left(node.left)
        right = reattach_punc_top_left(node.right)
        if right.is_binary and right.comb.is_punc:
            if right.comb.is_punc_left:
                return node.comb(Punc(False)(left, right.left), right.right)
            else:
                return Punc(False)(node.comb(left, right.left), right.right)
        else:
            return node.comb(left, right)


class CCGTreeDataset(Dataset):

    def __init__(self, fn: str, max_sent_len: int, tree_transform=None):
        self.trees = [tree for tree in ccg.open(fn) if len(tree.words()) < max_sent_len]
        if tree_transform is not None:
            self.trees = [tree_transform(x) for x in self.trees]

    def __getitem__(self, i):
        tree = self.trees[i]
        return {"words": tree.words(),
                "stags": tree.stags(),
                "tree": tree}

    def __len__(self):
        return len(self.trees)


class CCGTagDataset(Dataset):

    def __init__(self, fn: str, max_sent_len: int):
        self.data = list(CCGTagDataset.iter_supertagging_file(fn, max_sent_len))

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def iter_supertagging_file(fn: str, max_sent_len: int):
        with open(fn) as fh:
            for line in fh:
                tokens = [x.split("|") for x in line.strip().split(" ")]
                if len(tokens) > max_sent_len:
                    continue
                words = [x[0] for x in tokens]
                stags = [ccg.category(x[2]) for x in tokens]
                yield words, stags, None


def combined_CCG_dataset(tree_file, stag_file, max_sent_len : int, tree_transform=None):
    assert tree_file is not None or stag_file is not None

    if stag_file is not None:
        tag_dataset = CCGTagDataset(stag_file, max_sent_len=max_sent_len)

    if tree_file is not None:
        tree_dataset = CCGTreeDataset(tree_file, max_sent_len=max_sent_len, tree_transform=tree_transform)

    if stag_file is None:
        return tree_dataset
    elif tree_file is None:
        return tag_dataset
    else:
        return ConcatDataset([tag_dataset, tree_dataset])


def estimate_w2i(train_dataset):
    w2i = Any2Int(min_count=2, include_UNK=True, include_PAD=True)
    stag2i = Any2Int(min_count=1, include_UNK=False, include_PAD=False)
    for instance in train_dataset:
        for w in instance['words']:
            w2i.add_to_counts(w)
        for s in instance['stags']:
            stag2i.add_to_counts(s)
    w2i.freeze()
    stag2i.freeze()
    return w2i, stag2i


def prepare_data(train_trees_fn, train_stags_fn, dev_trees_fn, max_sent_len : int = 70):
    tree_transform = reattach_punc_top_left
    print("loading training data", file=stderr)
    train_dataset = combined_CCG_dataset(train_trees_fn, train_stags_fn, max_sent_len=max_sent_len, tree_transform=tree_transform)
    print("loading validation data", file=stderr)
    dev_dataset   = CCGTreeDataset(dev_trees_fn, max_sent_len=max_sent_len, tree_transform=tree_transform)
    print("loading done", file=stderr)

    w2i, stag2i = estimate_w2i(train_dataset)

    return {
        'train_dataset' : train_dataset,
        'dev_dataset' : dev_dataset,
        'w2i' : w2i,
        'stag2i' : stag2i,
    }
