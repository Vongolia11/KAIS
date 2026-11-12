"""KAIS (DiFA+) fixed-parameter detector for the five-corpus main table.

The reconstruction architecture and batching are adapted from DiFA
(Vongolia11/DiFA, 979206760d085f8230f1a5ac674ac192642f88fb; see LICENSE).
Only the teacher is trained. All score references come from normal FIT/CAL.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter, defaultdict

import numpy as np
from sklearn.covariance import LedoitWolf

COMPACT_SHAPE = (0, 1, 2, 3, 4, 7)
_LEX = re.compile(r"^[^A-Za-z0-9]*([A-Za-z]+(?:['-][A-Za-z]+)*)[^A-Za-z0-9]*$")
_CHARS = string.ascii_lowercase + "'-" + '$'


def documents(flat, lengths):
    x = np.asarray(flat, dtype=np.float64)
    raw = np.asarray(lengths)
    if (raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all()
            or np.any(raw <= 0) or np.any(raw != np.floor(raw))):
        raise ValueError('Document lengths must be positive integers')
    n = raw.astype(np.int64)
    if x.ndim != 1 or n.sum() != len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid flat score/length alignment')
    edges = np.r_[0, n.cumsum()]
    return [x[edges[i]:edges[i+1]] for i in range(len(n))]

def spike(docs):
    # log(mean(exp(4*s)))/4, evaluated stably.
    out = []
    for x in docs:
        m = x.max()
        out.append(m + np.log(np.exp(4. * (x-m)).mean()) / 4.)
    return np.asarray(out, dtype=np.float64)

def tail(x, reference):
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    x = np.asarray(x, dtype=np.float64)
    if (ref.ndim != 1 or not len(ref) or not np.isfinite(ref).all()
            or x.ndim != 1 or not np.isfinite(x).all()):
        raise ValueError('Invalid normal reference or scores')
    return -np.log((len(ref)-np.searchsorted(ref, x, side='left')+1.) / (len(ref)+1.))

def _lexical_unit(token):
    m = _LEX.fullmatch(str(token))
    return m.group(1).lower() if m else None

class _CharacterTrigram:
    """Frozen normal-only Char3 with Jeffreys smoothing, as in the paired experiment."""
    def __init__(self, fit_tokens, cal_tokens):
        self.counts, self.ctx = Counter(), Counter()
        self.cache = {}
        for doc in fit_tokens:
            for token in doc:
                word = _lexical_unit(token)
                if word is None:
                    continue
                text = '^^' + word + '$'
                for i in range(2, len(text)):
                    pair = text[i-2:i]
                    self.counts[(pair, text[i])] += 1
                    self.ctx[pair] += 1
        bins = defaultdict(list)
        for doc in cal_tokens:
            for token in doc:
                word = _lexical_unit(token)
                if word is not None:
                    bins[self.key(word)].append(self._nll(word))
        self.references = {k: np.sort(v) for k, v in bins.items()}

    @staticmethod
    def key(word):
        return int(math.floor(math.log2(len(word))))

    def _nll(self, word):
        if word in self.cache:
            return self.cache[word]
        s, text = 0., '^^' + word + '$'
        for i in range(2, len(text)):
            ctx = text[i-2:i]
            p = (self.counts[(ctx, text[i])] + .5) / (self.ctx[ctx] + .5 * len(_CHARS))
            s -= math.log(p)
        score = s / (len(word) + 1)
        self.cache[word] = score
        return score

    def score_docs(self, token_docs):
        results = []
        for doc in token_docs:
            values = []
            for token in doc:
                word = _lexical_unit(token)
                if word is None:
                    values.append(0.)
                    continue
                ref = self.references.get(self.key(word))
                values.append(float(tail([self._nll(word)], ref)[0]) if ref is not None else 0.)
            results.append(np.asarray(values, np.float32))
        return results

def routing(lengths):
    n=np.asarray(lengths,float)
    if n.ndim!=1 or not len(n) or np.any(n<=0):raise ValueError('Invalid normal routing lengths')
    med=float(np.median(n));q25,q75=np.quantile(n,[.25,.75])
    r=np.clip(1.-(q75-q25)/max(med,1e-8),0.,1.)
    d=np.clip((med*(1.-.90)-1.)/2.,0.,1.)
    struct=float(r+(1.-r)*.5*d)
    return np.array([1.-struct,struct])

def structural(flat, lengths, cal):
    mu,sd = float(np.mean(cal)),float(np.std(cal)+1e-8)
    threshold = (float(np.quantile(cal,.9))-mu)/sd
    scores=[]
    for doc in documents(flat,lengths):
        z = np.maximum((doc-mu)/sd-threshold,0.)
        k=min(3,len(z)); scores.append(np.partition(z,-k)[-k:].mean())
    return np.asarray(scores)

def saturated(diff,cap):
    return cap*np.arcsinh(np.maximum(diff,0.)/cap)


def shape_features(docs):
    rows = []
    for doc in docs:
        for token in doc:
            n = max(len(token), 1)
            run = longest = int(bool(token))
            for i in range(1, len(token)):
                run = run + 1 if token[i] == token[i - 1] else 1
                longest = max(longest, run)
            row = np.zeros(16, np.float32)
            row[list(COMPACT_SHAPE)] = [len(token), sum(c.isdigit() for c in token) / n,
                sum(c.isalpha() for c in token) / n, sum(c.isupper() for c in token) / n,
                sum(c in string.punctuation for c in token) / n, longest / n]
            rows.append(row)
    # The indexed six-column array retains the numerical layout of the published fit.
    return np.asarray(rows, np.float64)[:, COMPACT_SHAPE]


class ShapeGeometry:
    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-6)
        z = (x - self.mean) / self.scale
        cov = LedoitWolf(assume_centered=True).fit(z).covariance_
        ev, v = np.linalg.eigh(cov)
        floor = 1e-6 * max(float(ev.max()), 1.)
        self.inverse = np.linalg.inv((v * np.maximum(ev, floor)) @ v.T)
        return self

    def score(self, x):
        z = (x - self.mean) / self.scale
        return np.sqrt(np.maximum(np.einsum('ni,ij,nj->n', z, self.inverse, z), 0.))


def calibrated_scores(cal_sem, cal_char, cal_shape, cal_lengths,
                      sem, char, shape, lengths, weights):
    """Map test evidence to token and document scores with normal references."""
    gc, gt = tail(cal_shape, cal_shape), tail(shape, cal_shape)
    bc, bt = .5 * (cal_sem + gc), .5 * (sem + gt)
    fc, ft = .5 * (cal_sem + np.maximum(gc, cal_char)), .5 * (sem + np.maximum(gt, char))
    rc, rt = .5 * (cal_sem + cal_char), .5 * (sem + char)
    ac = .5 * tail(bc, bc) + .5 * tail(fc, fc)
    at = .5 * tail(bt, bc) + .5 * tail(ft, fc)
    hc, ht = tail(rc, rc), tail(rt, rc)
    cap = max(float(np.quantile(np.maximum(hc - ac, 0.), .95)), .25)
    token = at + .6 * saturated(ht - at, cap)

    pc, pt = spike(documents(bc, cal_lengths)), spike(documents(bt, lengths))
    sc = structural(cal_shape, cal_lengths, cal_shape)
    st = structural(shape, lengths, cal_shape)
    ws, wt = weights
    basec = ws * tail(pc, pc) + wt * tail(sc, sc)
    baset = ws * tail(pt, pc) + wt * tail(st, sc)
    hc, ht = spike(documents(rc, cal_lengths)), spike(documents(rt, lengths))
    hc, ht = tail(hc, hc), tail(ht, hc)
    cap = max(float(np.quantile(np.maximum(hc - basec, 0.), .95)), .25)
    return token, baset + saturated(ht - baset, cap)


def _teacher(d_model, device):
    from torch import nn

    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = self.block(d_model)
            self.teacher = self.block(d_model)

        @staticmethod
        def block(width):
            layer = nn.TransformerEncoderLayer(width, 8, dim_feedforward=1024,
                dropout=.1, activation='gelu', batch_first=True)
            return nn.TransformerEncoder(layer, num_layers=2)

        def forward(self, x, pad):
            latent = self.encoder(x, src_key_padding_mask=pad)
            return self.teacher(latent, src_key_padding_mask=pad)

    net = Teacher()
    # Match the CPU RNG after upstream DiFA construction, so the seeded
    # training shuffle starts at the same state. These layers are discarded;
    # they are never trained or used by KAIS.
    nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
    for n, h, out in [(6, 32, 2), (9, 48, 4)]:
        nn.Sequential(nn.Linear(n, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, out))
    return net.to(device)


def _loader(features, shuffle):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader

    def collate(batch):
        padded = pad_sequence(batch, batch_first=True)
        lengths = torch.tensor([len(x) for x in batch], dtype=torch.long)
        pad = torch.arange(padded.size(1)).expand(len(batch), -1) >= lengths.unsqueeze(1)
        return padded, pad, lengths

    return DataLoader([torch.tensor(x, dtype=torch.float32) for x in features],
                      batch_size=32, shuffle=shuffle, collate_fn=collate)


class KAIS:
    def __init__(self, device='cuda'):
        self.device = device

    @staticmethod
    def _loss(reconstructed, target):
        from torch import nn
        return (nn.functional.mse_loss(reconstructed, target, reduction='none').mean(dim=-1)
                + .5 * (1. - nn.functional.cosine_similarity(reconstructed, target, dim=-1)))

    def _reconstruction(self, features):
        import torch
        self.net.eval()
        result = []
        with torch.inference_mode():
            for x, pad, lengths in _loader(features, shuffle=False):
                x, pad = x.to(self.device), pad.to(self.device)
                loss = self._loss(self.net(x, pad), x).cpu().numpy()
                result.extend(loss[i, :n].astype(np.float32) for i, n in enumerate(lengths.tolist()))
        return np.concatenate(result).astype(np.float64)

    def fit(self, fit_features, fit_docs, cal_features, cal_docs):
        import torch
        self.net = _teacher(fit_features[0].shape[1], self.device)
        optimizer = torch.optim.AdamW(list(self.net.encoder.parameters()) + list(self.net.teacher.parameters()),
                                      lr=1e-4, weight_decay=1e-4)
        loader = _loader(fit_features, shuffle=True)
        self.net.train()
        for epoch in range(30):
            losses = []
            for x, pad, _ in loader:
                x, pad = x.to(self.device), pad.to(self.device)
                valid = (~pad).float()
                optimizer.zero_grad(set_to_none=True)
                loss = (self._loss(self.net(x, pad), x) * valid).sum() / valid.sum().clamp_min(1.)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            print(f'  Teacher {epoch + 1}/30: loss={np.mean(losses):.5f}', flush=True)
        self.geometry = ShapeGeometry().fit(shape_features(fit_docs))
        self.char = _CharacterTrigram(fit_docs, cal_docs)
        self.cal_rec = self._reconstruction(cal_features)
        self.cal_sem = tail(self.cal_rec, self.cal_rec).astype(np.float32).astype(np.float64)
        self.cal_char = np.concatenate(self.char.score_docs(cal_docs)).astype(np.float64)
        self.cal_shape = self.geometry.score(shape_features(cal_docs))
        self.cal_lengths = list(map(len, cal_docs))
        self.weights = routing(list(map(len, fit_docs)) + self.cal_lengths)
        return self

    def score(self, features, docs):
        semantic = tail(self._reconstruction(features), self.cal_rec).astype(np.float32).astype(np.float64)
        character = np.concatenate(self.char.score_docs(docs)).astype(np.float64)
        shape = self.geometry.score(shape_features(docs))
        return calibrated_scores(self.cal_sem, self.cal_char, self.cal_shape, self.cal_lengths,
                                 semantic, character, shape, list(map(len, docs)), self.weights)
