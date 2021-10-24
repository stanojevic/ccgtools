from typing import Any, Tuple, List
import os
from os.path import join
from sys import stderr
import time

from torch.utils.data import DataLoader
from torch import nn
import torch
from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from optuna import Trial
from optuna.integration import PyTorchLightningPruningCallback

from .neural_utils import optimizer_class_by_name, BiAffine, LSTMSmoother
from .embedders import construct_embedder
from .dataset import TreeTrainingBatchSampler
from .parser import Parser
from ccg.evaluation import sufficient_stats, combine_stats


class Model(pl.LightningModule):

    def __init__(self, trial: Trial, w2i, stag2i, batch_size, language="English", **kwargs):
        super().__init__()
        initialized = 'initialized' in kwargs
        kwargs['initialized'] = True

        self.language = language
        self.w2i = w2i
        self.stag2i = stag2i
        self.batch_size = batch_size
        self.splitting_length = 80
        self.optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "Adagrad", "RMSProp", "SGD"])
        self.lr = trial.suggest_loguniform("lr", 1e-5, 0.01)

        self.embedder = construct_embedder(trial, initialized, self.w2i)
        mlp_in_dim = self.embedder.dim
        mlp_hid_dim = trial.suggest_categorical("mlp hid dim", [32, 128, 256, 512])

        lstm_layers = trial.suggest_int("lstm layers", 0, 2)
        lstm_dropout = trial.suggest_float("lstm dropout", 0.0, 0.4, step=0.1)

        self.smoother = LSTMSmoother(self.embedder.dim, lstm_layers, dropout=lstm_dropout)

        use_span_logprobs = trial.suggest_categorical("use span logprobs", [True, False])
        self.biaffine_span = None
        if use_span_logprobs:
            self.biaffine_span = BiAffine(in_dim=mlp_in_dim, mid_dim=mlp_hid_dim, classes=1)

        self.mlp = nn.Sequential(nn.Linear(mlp_in_dim, mlp_hid_dim),
                                 nn.ReLU(),
                                 nn.Linear(mlp_hid_dim, self.stag2i.voc_size),
                                 nn.LogSoftmax(dim=-1))

        print("MODEL_HYPERPARAMETERS = ", file=stderr)
        pprint(trial.params, stream=stderr)
        self.save_hyperparameters()

    def on_validation_start(self) -> None:
        if hasattr(self, 'is_parsing_ready') and self.is_parsing_ready:
            self.parser = Parser(self, words_per_batch=25 * self.batch_size)

    def forward(self, batch):
        tok_ids = batch['token_ids'].to(self.device)
        tok_mask = batch['token_mask'].to(self.device)
        word2tok_map = batch['word2tok_maps'].to(self.device)
        word_mask = batch['word_mask'].to(self.device)

        embs = self.embedder.embed_token_ids_and_remap(tok_ids, tok_mask, word2tok_map)
        embs2 = self.smoother(embs)

        tag_logprobs = self.mlp(embs2)

        if self.biaffine_span is not None:
            span_logprobs = torch.sigmoid(self.biaffine_span(embs2, word_mask).squeeze(1)).log()
        else:
            span_logprobs = None

        return tag_logprobs, span_logprobs

    def training_step(self, batch, batch_idx):
        tag_logprobs, span_logprobs = self.forward(batch)
        stag_ids = batch['stag_ids'].to(self.device)
        stag_mask = batch['stag_mask'].to(self.device)

        gold_logprobs = tag_logprobs.gather(-1, stag_ids.unsqueeze(-1)).squeeze(-1)
        tag_loss = -gold_logprobs.masked_fill(~stag_mask, 0.).sum(-1)
        loss = tag_loss.clone()

        if span_logprobs is not None:
            criterion = nn.BCELoss(reduction='none')
            b = span_logprobs.shape[0]
            n = span_logprobs.shape[-1]
            span_matrix = batch['span_matrix'].to(self.device)
            importance = torch.ones(span_matrix.shape, device=self.device).triu(1) + span_matrix * (n - 1)
            span_loss = criterion(span_logprobs.exp(), batch['span_matrix']).triu(1) * importance
            span_loss = span_loss.sum(-1).sum(-1) * 2 / n
            span_loss *= batch['tree_sent_mask'].to(self.device)
            loss += span_loss

        loss *= batch['weights'].to(self.device)
        return loss.mean()

    def validation_step(self, batch, batch_idx):
        res = dict()
        tic = time.time()
        tag_logprobs, span_logprobs = self.forward(batch)
        predictions = tag_logprobs.argmax(-1)  # (b, l)
        res['correct_tags'] = (predictions.detach() == batch['stag_ids']).masked_fill(~batch['word_mask'], False).sum().item()
        res['period'] = time.time() - tic  # needs to be here because of PyTorch asynchrnous handling of ops
        res['tag_count'] = batch['word_mask'].sum().item()
        res['sent_count'] = batch['word_mask'].shape[0]
        if hasattr(self, 'is_parsing_ready') and self.is_parsing_ready:
            tic = time.time()
            pred_trees = list(self.parser.parse_iter(batch['words']))
            res['period'] = time.time() - tic
            gold_trees = batch['trees']
            res['metric_stats'] = [sufficient_stats(gold_tree, pred_tree, language=self.language) for gold_tree, pred_tree in zip(gold_trees, pred_trees)]
            res['incomplete'] = sum(x.is_binary and x.comb.is_glue for x in pred_trees)
        return res

    def validation_epoch_end(self, outputs):
        correct_tags = sum(x['correct_tags'] for x in outputs)
        total_tags = sum(x['tag_count'] for x in outputs)
        accuracy = 100 * correct_tags / total_tags
        self.log("accuracy", accuracy, prog_bar=True)

        sents_count = sum(x['sent_count'] for x in outputs)
        total_time = sum(x['period'] for x in outputs)
        sents_per_second = sents_count / total_time
        self.log("sent/s", int(sents_per_second), prog_bar=True)

        if hasattr(self, 'is_parsing_ready') and self.is_parsing_ready:
            delattr(self, 'parser')
            metric_stats = [x for xs in outputs for x in xs['metric_stats']]
            metrics = combine_stats(metric_stats)
            self.log("lf", metrics["labeled_dep_F"], prog_bar=True)
            self.log("uf", metrics["unlabeled_dep_F"], prog_bar=True)

            is_incomplete = sum(x['incomplete'] for x in outputs)
            sents_count = len(metric_stats)
            self.log("incomplete", 100 * is_incomplete / sents_count, prog_bar=True)

        if accuracy > 91:
            self.is_parsing_ready = True

    def configure_optimizers(self):
        optimizer = optimizer_class_by_name(self.optimizer)(self.parameters(), lr=self.lr)

        from ccg.supertagger.embedders import TransformersEmbedder
        if type(self.embedder) == TransformersEmbedder:
            max_epochs = 20
            steps_per_epoch = 2 * 40_000 // self.batch_size
            # very nice desc of schedulers https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr/10, max_lr=self.lr,
                                                          step_size_up=2*steps_per_epoch,
                                                          step_size_down=7*steps_per_epoch,
                                                          mode="exp_range", gamma=0.85, cycle_momentum=False)
            return {"optimizer": optimizer, "lr_scheduler" : {"scheduler" : scheduler, "interval" : "step"}}
        else:
            return optimizer

    def collate(self, batch_raw):
        tok_name = self.embedder.model_name
        tok_func = self.embedder.tokenize_and_s2i

        for instance in batch_raw:
            if not hasattr(instance, 'tokens'):
                instance['tokens'] = dict()
            if tok_name not in instance['tokens']:
                token_ids, word2tok_map = tok_func(instance['words'])
                instance['tokens'][tok_name] = dict()
                instance['tokens'][tok_name]['token_ids'] = token_ids
                instance['tokens'][tok_name]['word2tok_map'] = word2tok_map
            if 'stags' in instance and 'stag_ids' not in instance:
                instance['stag_ids'] = [self.stag2i.get_s2i(x, 0) for x in instance['stags']]

        batch = dict()
        batch['words'] = [x['words'] for x in batch_raw]
        batch['trees'] = [x.get('tree') for x in batch_raw]
        batch['word_mask'] = _pad_and_tensorize([[True for _ in s] for s in batch['words']])
        if 'stags' in batch_raw[0]:
            batch['stags'] = [x['stags'] for x in batch_raw]
            batch['stag_ids'] = _pad_and_tensorize([x['stag_ids'] for x in batch_raw])
            batch['stag_mask'] = _pad_and_tensorize([[self.stag2i.contains(y) for y in x['stags']] for x in batch_raw])
            batch['weights'] = torch.tensor([x.get('weight', 1.) for x in batch_raw])
            batch_size, max_word_len = batch['word_mask'].shape
            batch['span_matrix'] = torch.zeros((batch_size, max_word_len, max_word_len), dtype=torch.float32)
            batch['tree_sent_mask'] = torch.zeros(batch_size, dtype=torch.bool)
            for i, x in enumerate(batch_raw):
                if 'tree' in x:
                    batch['tree_sent_mask'][i] = True
                    for node in x['tree']:
                        start, end = node.span
                        if not node.is_term:
                            batch['span_matrix'][i, start, end - 1] = 1
        batch['token_ids'] = _pad_and_tensorize([x['tokens'][tok_name]['token_ids'] for x in batch_raw])
        batch['token_mask'] = _pad_and_tensorize(
            [[True for _ in x['tokens'][tok_name]['token_ids']] for x in batch_raw])
        batch['word2tok_maps'] = _pad_and_tensorize([x['tokens'][tok_name]['word2tok_map'] for x in batch_raw])

        return batch


def _pad_and_tensorize(list_batch):
    is_bool = type(list_batch[0][0]) == bool
    PAD = False if is_bool else 0
    max_len = max(len(x) for x in list_batch)
    batch2 = [x + [PAD] * (max_len - len(x)) for x in list_batch]
    if is_bool:
        return torch.tensor(batch2, dtype=torch.bool)
    else:
        return torch.tensor(batch2, dtype=torch.long)


class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback needed for Optuna."""

    def __init__(self, metric_to_optimize: str):
        super().__init__()
        self.metric_to_optimize = metric_to_optimize
        self.metrics = []
        self.epoch = 1

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)
        print("Epoch %d accuracy = %f" % (self.epoch, self.metrics[-1][self.metric_to_optimize]), file=stderr)
        self.epoch += 1


def objective(trial: Trial,
              save_dir,
              train_dataset,
              dev_dataset,
              w2i,
              stag2i,
              gpus,
              min_epochs,
              max_epochs,
              language,
              seed=None,
              use_tensorboard=False,
              optuna=False):
    print("PROCESS ID: {}".format(os.getpid()), file=stderr)
    print("TIME START : " + time.strftime('%l:%M%p on %d %b %Y'), file=stderr)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    if seed is not None:
        pl.seed_everything(seed)
    if gpus == -1:
        gpus = None

    batch_size = trial.suggest_int("batch size", 1, 1024)
    model_name = "model"
    metric_to_monitor = "accuracy"

    model = Model(trial, w2i, stag2i, batch_size=batch_size, language=language)

    train_sent_lens = [len(x['words']) for x in train_dataset]
    dev_sent_lens = [len(x['words']) for x in dev_dataset]

    loader_kwargs = {'collate_fn': model.collate, 'num_workers': 0}  # os.cpu_count()
    if (type(gpus) == int and gpus > 1) or (type(gpus) == list and len(gpus) > 1):
        train_data = DataLoader(train_dataset, batch_size=batch_size, **loader_kwargs)
        dev_data = DataLoader(dev_dataset, batch_size=batch_size, **loader_kwargs)
    else:
        train_sampler = TreeTrainingBatchSampler(train_sent_lens, batch_size_in_words=25 * batch_size,
                                                 out_of_order_noise=5000)
        train_data = DataLoader(train_dataset, batch_sampler=train_sampler, **loader_kwargs)
        dev_sampler = TreeTrainingBatchSampler(dev_sent_lens, batch_size_in_words=25 * batch_size, out_of_order_noise=0)
        dev_data = DataLoader(dev_dataset, batch_sampler=dev_sampler, **loader_kwargs)

    if optuna:
        logger = False
        early_stop_callback = PyTorchLightningPruningCallback(trial, monitor=metric_to_monitor)
        save_top_k = 0
    else:
        if use_tensorboard:
            from pytorch_lightning.loggers import TensorBoardLogger
            logger = TensorBoardLogger(
                save_dir=save_dir,
                name=model_name
            )
        else:
            logger = True
        early_stop_callback = EarlyStopping(
            monitor=metric_to_monitor,
            min_delta=0.00,
            patience=10,
            verbose=False,
            mode='max')
        save_top_k = 1
    checkpoint_callback_accuracy = ModelCheckpoint(
        dirpath=save_dir,
        filename="best_model",
        save_top_k=save_top_k,
        verbose=True,
        monitor=metric_to_monitor,
        mode='max')
    metrics_callback = MetricsCallback(metric_to_monitor)
    gradient_clip_val = 1. if trial.suggest_categorical("gradient clip", [True, False]) else 0.
    trainer = pl.Trainer(logger=logger,
                         min_epochs=min_epochs,
                         max_epochs=max_epochs,
                         num_sanity_val_steps=0,
                         gradient_clip_val=gradient_clip_val,
                         default_root_dir=join(save_dir, model_name),
                         gpus=gpus,
                         callbacks=[checkpoint_callback_accuracy, early_stop_callback, metrics_callback])
    trainer.fit(model, train_data, dev_data)
    print("TIME END : " + time.strftime('%l:%M%p on %d %b %Y'), file=stderr)
    return metrics_callback.metrics[-1][metric_to_monitor]
