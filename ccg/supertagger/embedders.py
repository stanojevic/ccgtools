from sys import stderr
import os
from os import getcwd
from os.path import realpath, dirname, join
import torch
from torch import nn
import numpy as np
from .any2int import Any2Int


def construct_embedder(trial, initialized: bool, w2i, prefix: str = ""):
    embedder_type = trial.suggest_categorical(prefix + "embedder type",
                                              ["normal", "fixed-fasttext"] +
                                              TransformersEmbedder.BERT_TYPES +
                                              ["fixed-" + x for x in TransformersEmbedder.BERT_TYPES])
    if embedder_type == "normal":
        embedder_dim = trial.suggest_categorical(prefix + "embedder dim", [128, 256, 300, 512])
        fasttext_language = trial.suggest_categorical(prefix + "embedder fasttext lang", ['none', 'en', 'zh'])
        embedder = NormalEmbedder(w2i, embedder_dim, fasttext_language, initialized)
    elif embedder_type == "fixed-fasttext":
        fasttext_language = trial.suggest_categorical(prefix + "embedder fasttext lang", ['en', 'zh'])
        embedder = FasttextEmbedder(fasttext_language)
    elif embedder_type.startswith("fixed-"):
        embedder = TransformersEmbedderFixed(embedder_type)
    else:
        embedder = TransformersEmbedder(embedder_type, initialized)
    return embedder


def get_fasttext_model(lang_id, dim: int = 300):
    # This function is better than fasttext.util.download_model() because by
    # default all embeddings will be downloaded in $FASTTEXT
    # instead of the local execution directory
    # and it doesn't depend on the fasttext version
    import fasttext as ft
    import gzip
    from pathlib import Path

    if lang_id.lower().startswith("english"):
        lang_id = "en"
    elif lang_id.lower().startswith("chinese"):
        lang_id = "zh"
    elif len(lang_id) != 2:
        raise Exception(f"unknown fasttext language {lang_id}")

    if 'FASTTEXT' in os.environ:
        storage_dir = os.environ['FASTTEXT']
    else:
        storage_dir = os.path.join(str(Path.home()), ".cache", "fasttext")

    if not os.path.exists(storage_dir):
        os.mkdir(storage_dir)
    raw_fn = f"cc.{lang_id}.300.bin.gz"
    fn_bin_gz = os.path.join(storage_dir, raw_fn)
    fn_bin = os.path.join(storage_dir, f"cc.{lang_id}.300.bin")
    if not os.path.exists(fn_bin):
        link = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/' + raw_fn
        print(f"downloading fasttext embeddings from {link}", file=stderr)
        import gdown
        gdown.download(link, fn_bin_gz)
        print(file=stderr)
        print("unzipping fasttext embeddings", file=stderr)
        with gzip.open(fn_bin_gz, 'rb') as fh_in, \
                open(fn_bin, "wb") as fh_out:
            fh_out.writelines(fh_in)
        os.remove(fn_bin_gz)

    from contextlib import redirect_stderr
    from os import devnull
    with open(devnull, 'w') as fnull, redirect_stderr(fnull) as err:
        ft_model = ft.load_model(fn_bin)

    if dim != 300:
        from fasttext.util import reduce_model
        reduce_model(ft_model, dim)
    return ft_model


class FasttextEmbedder(nn.Module):

    def __init__(self, fasttext_language: str):
        super(FasttextEmbedder, self).__init__()
        self.model_name = "fixed-fasttext"
        self.ft_model = get_fasttext_model(fasttext_language)
        self.dim = self.ft_model.get_dimension()

    def tokenize_and_s2i(self, words):
        word_ids = [self.ft_model.get_word_id(w) for w in words]
        word2tok_map = list(range(len(word_ids)))
        return word_ids, word2tok_map

    def embed_token_ids_and_remap(self, word_ids, word_mask, word2tok_map):
        device = word_ids.device
        word_ids = word_ids.cpu().numpy()
        b, l = word_ids.shape
        embs = np.empty((b, l, self.ft_model.get_dimension()), dtype=np.float32)
        for i in range(b):
            for j in range(l):
                embs[i, j] = self.ft_model.get_input_vector(word_ids[i][j])
        return torch.from_numpy(embs).to(device)


class NormalEmbedder(nn.Module):

    def __init__(self, w2i: Any2Int, dim: int, fasttext_language: str, initialized: bool):
        super(NormalEmbedder, self).__init__()
        self.model_name = "normal"
        self.w2i = w2i
        self.dim = dim
        self.emb = nn.Embedding(w2i.voc_size, dim)
        if not initialized and fasttext_language != 'none':
            ft_model = get_fasttext_model(fasttext_language, dim)
            vecs = np.empty((w2i.voc_size, dim), dtype=np.float32)
            for i, w in w2i.iter_item():
                vecs[i] = ft_model.get_word_vector(w)
            self.emb.weight.data.copy_(torch.from_numpy(vecs))

    def tokenize_and_s2i(self, words):
        UNK_i = self.w2i.UNK_i if self.w2i.UNK_i >= 0 else self.w2i.voc_size - 1
        word_ids = [self.w2i.get_s2i(w, UNK_i) for w in words]
        word2tok_map = list(range(len(word_ids)))
        return word_ids, word2tok_map

    def embed_token_ids_and_remap(self, word_ids, word_mask, word2tok_map):
        return self.emb(word_ids)


class TransformersEmbedder(nn.Module):
    BERT_TYPES = ["bert-base-uncased", "roberta-base", "xlm-roberta-base", "hfl/chinese-roberta-wwm-ext"]

    def __init__(self, model_name: str, initialized: bool):
        super(TransformersEmbedder, self).__init__()
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        self.model_name = "transformers-" + model_name
        kwargs = dict()
        if model_name.startswith("roberta"):
            kwargs['add_prefix_space'] = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        if initialized:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_config(config)
        else:
            self.model = AutoModel.from_pretrained(model_name)
        o = torch.zeros(1, 1, dtype=torch.int64)
        self.dim = self.embed_token_ids_and_remap(o, o, o, with_norm=False).shape[-1]
        self.layer_norm = nn.LayerNorm(self.dim)

    def tokenize_and_s2i(self, words):
        max_len = 512
        inputt = self.tokenizer \
            .batch_encode_plus([words], padding=True, is_split_into_words=True, return_tensors='pt')
        token_ids = inputt['input_ids'][0][:max_len].tolist()

        word2tok_map = [None] * len(words)
        for p, w in enumerate(inputt.word_ids(0)):
            if w is not None and word2tok_map[w] is None:
                if p >= max_len:
                    word2tok_map[w] = max_len - 1
                else:
                    word2tok_map[w] = p

        return token_ids, word2tok_map

    def embed_token_ids_and_remap(self, tok_ids, tok_mask, word2tok_map, with_norm=True):
        embs = self.model(input_ids=tok_ids, attention_mask=tok_mask)[0]
        word2tok_map = word2tok_map.unsqueeze(2).expand(-1, -1, embs.shape[-1])
        embs = embs.gather(-2, word2tok_map)
        if with_norm:
            embs = self.layer_norm(embs)
        return embs


class TransformersEmbedderFixed(nn.Module):

    def __init__(self, model_name: str):
        super(TransformersEmbedderFixed, self).__init__()
        self.embedder = [TransformersEmbedder(model_name[len("fixed-"):], initialized=False)]
        self.model_name = "fixed-" + self.embedder[0].model_name
        self.embedder[0].eval()
        self.dim = self.embedder[0].dim

    def tokenize_and_s2i(self, words):
        return self.embedder[0].tokenize_and_s2i(words)

    def embed_token_ids_and_remap(self, tok_ids, tok_mask, word2tok_map):
        return self.embedder[0].embed_token_ids_and_remap(tok_ids, tok_mask, word2tok_map).detach()
