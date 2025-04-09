import argparse
import random

import numpy as np
import torch

from model_pytorch import LMModel, load_openai_pretrained_model
from text_utils import TextEncoder


def make_batch(X, n_vocab: int, n_special: int, device: torch.device):
    X = np.array(X)
    assert X.ndim in [1, 2]
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
    pos_enc = np.arange(n_vocab + n_special, n_vocab + n_special + X.shape[-1])
    pos_enc = np.expand_dims(pos_enc, axis=0)
    batch = np.stack([X, pos_enc], axis=-1)
    batch_tensor = torch.tensor(batch, dtype=torch.long).to(device)
    return batch_tensor

def append_batch(X, next_idx):
    next_pos = X[:, -1:, 1] + 1
    next_x = torch.cat((next_idx, next_pos), -1).unsqueeze(1)
    return torch.cat((X, next_x), 1)

class GenerateText:
    def __init__(self, **kwargs):
        self.args = argparse.Namespace(
            desc = kwargs.get('desc', "Description"),
            dataset = kwargs.get('dataset'),
            log_dir = kwargs.get('log_dir', 'log/'),
            save_dir = kwargs.get('save_dir', 'save/'),
            data_dir = kwargs.get('data_dir', 'data/'),
            submission_dir = kwargs.get('submission_dir', 'submission/'),
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
        load_openai_pretrained_model(self.lm_model.transformer, n_ctx=n_ctx, n_special=self.n_special)
        self.lm_model.to(self.device)

    def generate(self, text: str):
        self.lm_model.eval()

        generated = []
        X = self.text_encoder.encode([text,])
        XMB = make_batch(X, self.n_vocab, self.n_special, self.device)

        for _ in range(self.args.gen_len):
            lm_probs = self.lm_model(XMB)
            if self.args.topk == 0:
                next_idx = torch.multinomial(lm_probs[:, -1, :], 1)
            else:
                values, indices = lm_probs[:, -1, :].topk(self.args.topk)
                next_idx = indices.gather(-1, torch.multinomial(values, 1))
            next_token = self.text_encoder.decoder[next_idx.item()].replace('</w>', '')
            generated.append(next_token)
            XMB = append_batch(XMB, next_idx)

        return ' '.join(generated)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, help="Description")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument('--gen_len', type=int, default=20)
    parser.add_argument('--topk', type=int, default=10)

    args = parser.parse_args()
    # print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Constants
    # submit = args.submit
    # dataset = args.dataset
    n_ctx = args.n_ctx
    # save_dir = args.save_dir
    # desc = args.desc
    # data_dir = args.data_dir
    # log_dir = args.log_dir
    # submission_dir = args.submission_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    n_special = 0   # XD: useless for language modeling task
    vocab = n_vocab + n_special + n_ctx

    lm_model = LMModel(args, vocab, n_ctx, return_probs=True)
    load_openai_pretrained_model(lm_model.transformer, n_ctx=n_ctx, n_special=n_special)
    lm_model.to(device)

    lm_model.eval()

    text = input('Input some beginning words: ')
    while text != 'q':
        X = text_encoder.encode([text,])
        XMB = make_batch(X, n_vocab, n_special, device)

        for _ in range(args.gen_len):
            lm_probs = lm_model(XMB)
            if args.topk == 0:
                next_idx = torch.multinomial(lm_probs[:, -1, :], 1)
            else:
                values, indices = lm_probs[:, -1, :].topk(args.topk)
                next_idx = indices.gather(-1, torch.multinomial(values, 1))
            next_token = text_encoder.decoder[next_idx.item()].replace('</w>', '')
            print(next_token, end=' ')
            XMB = append_batch(XMB, next_idx)

        print()
        text = input('Input some beginning words: ')
