"""Fit KAIS and write its main-table row for each of the five corpora."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import random

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from kais import KAIS
from prepare_data import TOKENCORE, SOURCES, ensure_tokenad, ensure_controlled

ROOT = Path(__file__).resolve().parent
MODEL = 'bert-base-uncased'
REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'
CONDITIONS = ('typo_1', 'typo_3', 'gibberish_1', 'gibberish_3')
METRICS = ('token_auroc', 'token_ap', 'doc_auroc', 'doc_ap')


def seed():
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def load_records(name, data_dir):
    if name in SOURCES:
        path = data_dir / 'prepared' / f'{name}_1000_200_500_20261008_271828.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        return data['fit'], data['cal'], data['test']
    # Called only after verifying the pinned upstream file's SHA-256.
    data = np.load(data_dir / f'{name}.npy', allow_pickle=True).item()
    tokens, labels = data['tokens'], data['labels']
    rng = random.Random(42)
    normal = [i for i, y in enumerate(labels) if not any(y)]
    anomaly = [i for i, y in enumerate(labels) if any(y)]
    rng.shuffle(normal)
    n = len(normal) // 2
    reference, test = normal[:n], normal[n:] + anomaly
    rng.shuffle(test)
    groups = {}
    for i, idx in enumerate(reference):
        groups.setdefault(tuple(tokens[idx]), []).append(i)
    groups = list(groups.values())
    cal, fit = [], []
    target = max(1, round(.2 * len(reference)))
    for i in np.random.default_rng(953).permutation(len(groups)):
        (cal if len(cal) < target else fit).extend(groups[i])

    def rows(ids):
        return [dict(tokens=list(map(str, tokens[i])), labels=list(map(int, labels[i]))) for i in ids]

    return rows([reference[i] for i in sorted(fit)]), rows([reference[i] for i in sorted(cal)]), rows(test)


def encode(splits, device, cache_dir, offline):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    kwargs = dict(revision=REVISION, cache_dir=cache_dir, local_files_only=offline)
    config = AutoConfig.from_pretrained(MODEL, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, **kwargs)
    model = AutoModel.from_pretrained(MODEL, **kwargs).to(device).eval()
    limit = min(int(tokenizer.model_max_length), int(config.max_position_embeddings))
    output = []
    with torch.inference_mode():
        for docs in splits:
            features = []
            for words in docs:
                encoded = tokenizer(words, is_split_into_words=True, return_tensors='pt',
                                    truncation=False, padding=False)
                ids = encoded.word_ids(batch_index=0)
                kept = [i for i in ids if i is not None]
                if len(ids) > limit or len(ids) != encoded['input_ids'].shape[-1] or (
                        sorted(set(kept)) != list(range(len(words))) or kept != sorted(kept)):
                    raise ValueError('BERT could not preserve the original word sequence without truncation')
                h = model(input_ids=encoded['input_ids'].to(device),
                          attention_mask=encoded['attention_mask'].to(device)).last_hidden_state[0].float().cpu().numpy()
                features.append(np.stack([np.max(h[[j for j, w in enumerate(ids) if w == i]], axis=0)
                                          for i in range(len(words))]).astype(np.float32))
            output.append(features)
    return output


def evaluate(token, doc, test, controlled):
    lengths = [len(r['tokens']) for r in test]
    y_token = np.asarray([y for r in test for y in r['labels']], dtype=int)
    y_doc = np.asarray([any(r['labels']) for r in test], dtype=int)
    if token.shape != y_token.shape or doc.shape != y_doc.shape or sum(lengths) != len(y_token):
        raise ValueError('Score/label alignment mismatch')

    def metrics(mask):
        words = np.repeat(mask, lengths)
        return [roc_auc_score(y_token[words], token[words]),
                average_precision_score(y_token[words], token[words]),
                roc_auc_score(y_doc[mask], doc[mask]),
                average_precision_score(y_doc[mask], doc[mask])]

    if not controlled:
        return metrics(np.ones(len(test), dtype=bool))
    conditions = np.asarray([r['condition'] for r in test])
    return np.mean([metrics(np.isin(conditions, ['clean', name])) for name in CONDITIONS], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT.parent / 'data')
    parser.add_argument('--model-cache-dir', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--offline', action='store_true', help='Require local data and BERT weights')
    parser.add_argument('--out', type=Path, default=Path('kais_main_table.csv'))
    args = parser.parse_args()

    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use a CUDA machine or explicitly pass --device cpu')
    ensure_tokenad(args.data_dir, offline=args.offline)
    ensure_controlled(args.data_dir, offline=args.offline)

    rows = []
    for name in (*TOKENCORE, *SOURCES):
        fit, cal, test = load_records(name, args.data_dir)
        if any(any(r['labels']) for r in fit + cal):
            raise ValueError('FIT and CAL must contain normal documents only')
        documents = [[r['tokens'] for r in split] for split in (fit, cal, test)]
        print(f'[{name}] encoding FIT={len(fit)}, CAL={len(cal)}, TEST={len(test)}', flush=True)
        seed()
        xf, xc, xt = encode(documents, args.device, args.model_cache_dir, args.offline)
        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
        seed()
        model = KAIS(args.device).fit(xf, documents[0], xc, documents[1])
        token, doc = model.score(xt, documents[2])
        values = evaluate(token, doc, test, name in SOURCES)
        rows.append(dict(dataset=name, **dict(zip(METRICS, map(float, values)))))
        print('  ' + ', '.join(f'{k}={v:.6f}' for k, v in zip(METRICS, values)), flush=True)
        del model, xf, xc, xt

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=('dataset', *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
