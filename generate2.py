import argparse
import random

import numpy as np
import torch

from .model_pytorch import LMModel, load_openai_pretrained_model
from .text_utils import TextEncoder

class GenerateText:
    def __init__(self, **kwargs):
        self.args = argparse.Namespace(
            desc = kwargs.get('desc', "Description"),
            dataset = kwargs.get('dataset'),
            log_dir = kwargs.get('log_dir', 'log/'),
            save_dir = kwargs.get('save_dir', 'save/'),
            data_dir = kwargs.get('data_dir', 'data/'),
            submission_dir = kwargs.get('submission_dir', 'submission/'),
            path_model = kwargs.get('path_model', './model/'),
            path_names = kwargs.get('path_names', './'),
            submit = kwargs.get('submit', True),
            analysis = kwargs.get('analysis', True),
            seed = kwargs.get('seed', 42),
            n_iter = kwargs.get('n_iter', 3),
            n_batch = kwargs.get('n_batch', 8),
            max_grad_norm = kwargs.get('max_grad_norm', 1),
            lr = kwargs.get('lr', 6.25e-5),
            lr_warmup = kwargs.get('lr_warmup', 0.002),
            n_ctx = kwargs.get('n_ctx', 512),
            n_embd = kwargs.get('n_embd', 768),
            n_head = kwargs.get('n_head', 12),
            n_layer = kwargs.get('n_layer', 12),
            embd_pdrop = kwargs.get('embd_pdrop', 0.1),
            attn_pdrop = kwargs.get('attn_pdrop', 0.1),
            resid_pdrop = kwargs.get('resid_pdrop', 0.1),
            clf_pdrop = kwargs.get('clf_pdrop', 0.1),
            l2 = kwargs.get('l2', 0.01),
            vector_l2 = kwargs.get('vector_l2', True),
            opt = kwargs.get('opt', 'adam'),
            afn = kwargs.get('afn', 'gelu'),
            lr_schedule = kwargs.get('lr_schedule', 'warmup_linear'),
            encoder_path = kwargs.get('encoder_path', 'model/encoder_bpe_40000.json'),
            bpe_path = kwargs.get('bpe_path', 'model/vocab_40000.bpe'),
            n_transfer = kwargs.get('n_transfer', 12),
            lm_coef = kwargs.get('lm_coef', 0.5),
            b1 = kwargs.get('b1', 0.9),
            b2 = kwargs.get('b2', 0.999),
            e = kwargs.get('e', 1e-8),
            n_valid = kwargs.get('n_valid', 374),
            gen_len = kwargs.get('gen_len', 20),
            topk = kwargs.get('topk', 10),
        )

        seed = self.args.seed
        n_ctx = self.args.n_ctx

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_encoder = TextEncoder(self.args.encoder_path, self.args.bpe_path)
        self.n_vocab = len(self.text_encoder.encoder)

        self.n_special = 0   # XD: useless for language modeling task
        vocab = self.n_vocab + self.n_special + n_ctx

        self.lm_model = LMModel(self.args, vocab, n_ctx, return_probs=True)
        load_openai_pretrained_model(
            self.lm_model.transformer,
            n_ctx=n_ctx,
            n_special=self.n_special,
            path=self.args.path_model,
            path_names=self.args.path_names
        )
        self.lm_model.to(self.device)

    def make_batch(self, X):
        X = np.array(X)
        assert X.ndim in [1, 2]
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        pos_enc = np.arange(self.n_vocab + self.n_special, self.n_vocab + self.n_special + X.shape[-1])
        pos_enc = np.expand_dims(pos_enc, axis=0)
        batch = np.stack([X, pos_enc], axis=-1)
        batch_tensor = torch.tensor(batch, dtype=torch.long).to(self.device)
        return batch_tensor

    def append_batch(self, X, next_idx):
        next_pos = X[:, -1:, 1] + 1
        next_x = torch.cat((next_idx, next_pos), -1).unsqueeze(1)
        return torch.cat((X, next_x), 1)

    def generate(self, text: str):
        self.lm_model.eval()

        generated = []
        X = self.text_encoder.encode([text,])
        XMB = self.make_batch(X)

        for _ in range(self.args.gen_len):
            lm_probs = self.lm_model(XMB)
            if self.args.topk == 0:
                next_idx = torch.multinomial(lm_probs[:, -1, :], 1)
            else:
                values, indices = lm_probs[:, -1, :].topk(self.args.topk)
                next_idx = indices.gather(-1, torch.multinomial(values, 1))
            next_token = self.text_encoder.decoder[next_idx.item()].replace('</w>', '')
            generated.append(next_token)
            XMB = self.append_batch(XMB, next_idx)

        return ' '.join(generated)
