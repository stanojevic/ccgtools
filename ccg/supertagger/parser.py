from ccg.astar import AStarSearch
import ccg
import torch
import os
from pathlib import Path

PRETRAINED_MODELS_DIR = os.path.join(str(Path.home()), ".cache", "ccgtools", "models")


class Parser:

    def __init__(self,
                 model_file : str,
                 words_per_batch : int = 25*100,
                 do_tokenization : bool = False,
                 max_steps : int = 1_000_000,
                 prune_beta : float = 0.0001,
                 prune_top_k_tags : int = 50,
                 num_cpus : int = None):
        self.words_per_batch = words_per_batch
        self.do_tokenization = do_tokenization

        if type(model_file) == str:
            from ccg.supertagger.model import Model
            if model_file.startswith("pretrained:"):
                model_name = model_file[len("pretrained:"):]
                model_file = os.path.join(PRETRAINED_MODELS_DIR, model_name)
                if not os.path.isfile(model_file):
                    import gdown
                    link_to_list = r"https://raw.githubusercontent.com/stanojevic/ccgtools/main/ccg/supertagger/configs/pretrained_models_locations.tsv"
                    os.makedirs(PRETRAINED_MODELS_DIR, exist_ok=True)
                    list_file = os.path.join(PRETRAINED_MODELS_DIR, "available_models_links.tsv")
                    if os.path.isfile(list_file):
                        os.unlink(list_file)
                    gdown.download(link_to_list, list_file)
                    with open(list_file) as fh:
                        pretrained_links = dict(x.split("\t") for x in fh.readlines())
                    if model_name in pretrained_links:
                        gdown.download(pretrained_links[model_name], model_file)
                    else:
                        raise Exception(f"Model {model_name} not found in the list of available models")
            self.model = Model.load_from_checkpoint(checkpoint_path=model_file)
        else:
            self.model = model_file

        self.model.eval()

        ordered_cats = [tag for _, tag in self.model.stag2i.iter_item()]
        if num_cpus is None:
            num_cpus = 0

        self.search = AStarSearch(
            ordered_cats,
            max_steps=max_steps,
            prune_beta=prune_beta,
            prune_top_k_tags=prune_top_k_tags,
            num_cpus=1,
            use_normal_form=True
        )

    def to(self, device):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif type(device) == int:
            if device >= 0:
                device = f"cuda:{device}"
            else:
                device = 'cpu'
        self.model.to(device)

    def parse_batch(self, sents):
        batch = [{'words': x} for x in sents]
        batch_prepared = self.model.collate(batch)
        tag_logprobs, span_logprobs = self.model(batch_prepared)
        return self.search.parse_batch(sents, tag_logprobs.detach().cpu().numpy(), span_logprobs.detach().cpu().numpy())

    def parse_iter(self, sent_iter, return_as_str: bool = False):
        future = []
        for sent_batch in self._sent_iter_to_batch_iter(sent_iter):
            bck = future
            future = self.parse_batch(sent_batch)
            for tree in bck:
                tree.language = self.model.language
                yield tree
        for tree in future:
            tree.language = self.model.language
            yield tree

    def parse_sent(self, sent: str):
        assert type(sent) == str, "input sentence must be a raw string"
        return list(self.parse_iter([sent]))[0]

    def _sent_iter_to_batch_iter(self, sent_iter):
        curr_batch = []
        curr_batch_max_sent_len = 0
        for sent in sent_iter:
            if type(sent) == str:
                sent = x.strip().split()
            if self.do_tokenization:
                sent = ccg.tokenize(sent, self.model.language)
            curr_batch.append(sent)
            curr_batch_max_sent_len = max(curr_batch_max_sent_len, len(sent))
            if curr_batch_max_sent_len*len(curr_batch) > self.words_per_batch:
                yield curr_batch
                curr_batch = []
                curr_batch_max_sent_len = 0
        if curr_batch != []:
            yield curr_batch
