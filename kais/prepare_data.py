"""Recreate the five fixed evaluation corpora from pinned upstream sources.
Downloads ORIGINAL three TokenCore word-labeled .npy files from pinned GitHub commit,
and constructs AG News/DBpedia using the EXACT historic controlled-v1 procedure.
This script never calls the previous experiment directory.
Requires: numpy (for verification), pyarrow (only for controlled corpora).
"""
from __future__ import annotations

import argparse
import numpy as np

TOKENCORE_COMMIT = 'a5887b9c7b26805465b4f0c4a582865b00c43344'
TOKENCORE = {
    'sms_spam': '912b081f401cd5b07aae235c138337c80b63cc9d943d0bc5a00da93449ba2806',
    'restaurant_review': '04b366d402276c58f863928df2e333ce19abc9a489e023267ec700253cef7e53',
    'grammar': '92c89efe7f698608c672d8eac6ba73fd6a7bcf8d899a773552c64e0b20726904',
}
CONTROLLED_SOURCE_SHA256 = {
    'ag_news': '7b01eef3b8ab677e084c77308dbe86885049b887091d48929d70dd4d2169e2ef',
    'dbpedia': '96c6d219c4cbce3cd7abf282a34a3fe78ffddb14f7f72f00cd5376c1b5b93078',
}
CONTROLLED_LEXICON_SHA256 = '3ed0c94610d8bcf7c11bbb49c56aa49c7234d32b66824df91f554169e572da48'
CONTROLLED_MANIFEST_SHA256 = {
    'ag_news': '7fc666343aba6d4a9c51adffa694b2952b36ad6ac72ac113aa4a5391cf8fa80f',
    'dbpedia': '71ddd20e3b844176d1dfd5600bcff8b6e9989a2068fa0edc9dd1f3e86bf9c1a3',
}

import collections
import hashlib
import html
import json
import os
import random
import re
import string
import tempfile
import urllib.request
from pathlib import Path

VERSION = 'controlled-v1'
SOURCES = {
    'ag_news': dict(repo='fancyzhx/ag_news', revision='eb185aade064a813bc0b7f42de02595523103ca4', prefix='data', classes=4),
    'dbpedia': dict(repo='fancyzhx/dbpedia_14', revision='9abd46cf7fc8b4c64290f26993c540b92aa145ac', prefix='dbpedia_14', classes=14),
}
LEXICON_URL = 'https://raw.githubusercontent.com/dwyl/english-words/8179fe68775df3f553ef19520db065228e65d1d3/words_alpha.txt'
WORD = re.compile(r'^([^A-Za-z]*)([A-Za-z]{4,20})([^A-Za-z]*)$')
EDITED_WORD = re.compile(r'^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.flush(); os.fsync(f.fileno())
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(path)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def download(url, path):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    print('Downloading', path.name, flush=True)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'KAIS-controlled-bench/1.0'})
        with urllib.request.urlopen(req, timeout=60) as src, tmp.open('wb') as dst:
            for b in iter(lambda: src.read(1024 * 1024), b''):
                dst.write(b)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def lexicon(data_dir):
    path = download(LEXICON_URL, data_dir / 'lexicon.txt')
    if file_hash(path) != CONTROLLED_LEXICON_SHA256:
        raise ValueError(f'{path}: pinned lexicon checksum mismatch')
    return set(path.read_text(encoding='utf-8').lower().split())


def normalize(text):
    # Same fixed formatting cleanup for every source split and method.
    text = re.sub(r'(?<!\w)#(\d{2,5});', lambda m: '&#' + m[1] + ';', text)
    text = html.unescape(html.unescape(text))
    text = text.replace('\\n', ' ').replace('\\', ' ')
    return ' '.join(text.split())


def eligible(tokens, words):
    return [i for i, t in enumerate(tokens) if (m := WORD.fullmatch(t)) and m[2].lower() in words
            and (m[2].islower() or m[2].isupper() or m[2].istitle())]


def source_rows(name, split, data_dir):
    """Stream a pinned public parquet release; no Hugging Face account needed."""
    import pyarrow.parquet as pq
    s = SOURCES[name]
    filename = f'{s["prefix"]}/{split}-00000-of-00001.parquet'
    url = f'https://huggingface.co/datasets/{s["repo"]}/resolve/{s["revision"]}/{filename}'
    path = download(url, data_dir / 'raw' / f'{name}_{split}.parquet')
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            text = row['text'] if name == 'ag_news' else row['title'] + ' ' + row['content']
            yield text, int(row['label'])


def make_source_pool(name, data_dir, words, normal_count=4000, test_count=2000):
    """Preselect balanced small pools from official train/test, before any scoring."""
    nc = SOURCES[name]['classes']
    selected, counters, seen = {}, {}, set()
    for split, limit in [('test', test_count), ('train', normal_count)]:
        buckets = collections.defaultdict(list)
        counts = collections.Counter()
        for index, (raw, topic) in enumerate(source_rows(name, split, data_dir)):
            counts['source_rows'] += 1
            tokens = normalize(raw).split()
            if not 12 <= len(tokens) <= 96 or len(eligible(tokens, words)) < 3:
                counts['length_or_eligibility_rejected'] += 1
                continue
            key = ' '.join(tokens).casefold()
            if key in seen:
                counts['duplicate_rejected'] += 1
                continue
            seen.add(key)  # Remove train duplicates of all retained test candidates.
            rid = f'{name}:{split}:{index}'
            r = dict(id=rid, source_split=split, topic=topic, tokens=tokens)
            buckets[topic].append((digest(dict(id=rid, selection_seed=20261008)), r))
        rows = []
        for topic in range(nc):
            quota = limit // nc + int(topic < limit % nc)
            ranked = sorted(buckets[topic], key=lambda x: x[0])
            if len(ranked) < quota:
                raise ValueError(f'{name}/{split}: insufficient topic {topic} candidates')
            rows.extend(r for _, r in ranked[:quota])
        selected['normal_pool' if split == 'train' else 'test_pool'] = rows
        counters[split] = dict(counts)
    result = dict(dataset=name, version=VERSION, source=SOURCES[name], filter='12..96 whitespace words; >=3 lexicon-matched editable words',
                  selection_seed=20261008, counters=counters, **selected)
    atomic_json(data_dir / 'sources' / f'{name}.json', result)
    return result


def load_pool(name, args, words):
    path = args.data_dir / 'sources' / f'{name}.json'
    if not path.exists():
        make_source_pool(name, args.data_dir, words)
    if file_hash(path) != CONTROLLED_SOURCE_SHA256[name]:
        raise ValueError(f'{path}: pinned source pool checksum mismatch')
    pool = json.loads(path.read_text(encoding='utf-8'))
    if pool['version'] != VERSION or pool['source'] != SOURCES[name]:
        raise ValueError('Source protocol mismatch')
    return pool


def balanced(rows, count, seed):
    if count > len(rows):
        raise ValueError(f'Requested {count} rows, source pool has only {len(rows)}')
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[r['topic']].append(r)
    for topic, values in buckets.items():
        values.sort(key=lambda r: digest(dict(id=r['id'], seed=seed)))
    # Round-robin prevents dependence on the class-ordered source file.
    out, i = [], 0
    while len(out) < count:
        for topic in sorted(buckets):
            if i < len(buckets[topic]) and len(out) < count:
                out.append(buckets[topic][i])
        i += 1
    return out


def typo_candidates(word):
    w = word.lower()
    out = {w[:i] + w[i+1:] for i in range(len(w))}
    out |= {w[:i] + w[i] + w[i:] for i in range(len(w))}
    out |= {w[:i] + w[i+1] + w[i] + w[i+2:] for i in range(len(w)-1)}
    out |= {w[:i] + c + w[i+1:] for i in range(len(w)) for c in string.ascii_lowercase if c != w[i]}
    out.discard(w)
    return sorted(out)


def corrupt(token, mode, words, rng):
    m = WORD.fullmatch(token)
    prefix, word, suffix = m.groups()
    if mode == 'typo':
        candidates = [c for c in typo_candidates(word) if c not in words]
        if not candidates:
            raise ValueError('No lexicon-excluded one-edit candidate')
        replacement = rng.choice(candidates)
    else:
        # Equal character count and preserved punctuation/case; no length-only cue.
        for _ in range(10000):
            replacement = ''.join(rng.choice(string.ascii_lowercase) for _ in word)
            if replacement not in words and replacement != word.lower():
                break
        else:
            raise ValueError('Failed to generate lexicon-excluded gibberish')
    if word.isupper(): replacement = replacement.upper()
    elif word.istitle(): replacement = replacement.capitalize()
    return prefix + replacement + suffix


def record(source, condition='clean', tokens=None, labels=None, edits=None):
    return dict(id=source['id'] + '/' + condition, group=source['id'], topic=source['topic'],
                condition=condition, tokens=list(source['tokens'] if tokens is None else tokens),
                labels=[0]*len(source['tokens']) if labels is None else labels, edits=edits or [])


def stats(rows):
    return dict(documents=len(rows), tokens=sum(len(r['tokens']) for r in rows),
                anomaly_documents=sum(any(r['labels']) for r in rows),
                anomaly_tokens=sum(sum(r['labels']) for r in rows))


def validate(payload, words):
    splits = [{r['group'] for r in payload[k]} for k in ['fit','cal','test']]
    if any(splits[a] & splits[b] for a,b in [(0,1),(0,2),(1,2)]):
        raise ValueError('Source group leakage')
    texts = [{tuple(t.casefold() for t in r['tokens']) for r in payload[k]} for k in ['fit','cal','test']]
    if any(texts[a] & texts[b] for a,b in [(0,1),(0,2),(1,2)]):
        raise ValueError('Exact normalized text leakage')
    if len({r['id'] for k in ['fit','cal','test'] for r in payload[k]}) != sum(len(payload[k]) for k in ['fit','cal','test']):
        raise ValueError('Duplicate record ID')
    clean = {r['group']: r for r in payload['test'] if r['condition'] == 'clean'}
    for split in ['fit','cal','test']:
        for r in payload[split]:
            if not r['tokens'] or len(r['tokens']) != len(r['labels']):
                raise ValueError('Token/label alignment error')
            if any(not isinstance(t, str) or not t.strip() for t in r['tokens']):
                raise ValueError('Empty token')
            if split != 'test' and any(r['labels']):
                raise ValueError('Non-normal training/calibration record')
            if r['condition'] == 'clean':
                assert not any(r['labels'])
                continue
            original = clean[r['group']]['tokens']
            changed = [int(a != b) for a,b in zip(original, r['tokens'])]
            assert len(original) == len(r['tokens']) and changed == r['labels']
            assert sum(changed) == int(r['condition'].rsplit('_',1)[1])
            for e in r['edits']:
                assert e['before'] == original[e['index']] and e['after'] == r['tokens'][e['index']]
                new = EDITED_WORD.fullmatch(e['after'])[2].lower()
                assert new not in words
                if r['condition'].startswith('typo'):
                    assert new in typo_candidates(WORD.fullmatch(e['before'])[2])
                else:
                    assert len(e['before']) == len(e['after'])


def prepare(name, args):
    words = lexicon(args.data_dir)
    pool = load_pool(name, args, words)
    normal = balanced(pool['normal_pool'], args.normal_docs, args.data_seed)
    test = balanced(pool['test_pool'], args.test_sources, args.data_seed + 1)
    # Stratified, source-disjoint calibration. All models use the same fit subset.
    calibration = balanced(normal, args.cal_docs, args.data_seed + 2)
    cal_ids = {r['id'] for r in calibration}
    fit = [record(r) for r in normal if r['id'] not in cal_ids]
    cal = [record(r) for r in calibration]
    tests = [record(r) for r in test]
    for mode in ['typo','gibberish']:
        for source in test:
            positions = eligible(source['tokens'], words)
            rng = random.Random(int(digest(dict(id=source['id'], seed=args.corruption_seed, purpose='positions')),16))
            rng.shuffle(positions)
            selected = positions[:3]
            replacements = {}
            for pos in selected:
                rng = random.Random(int(digest(dict(id=source['id'], position=pos, seed=args.corruption_seed, mode=mode)),16))
                replacements[pos] = corrupt(source['tokens'][pos], mode, words, rng)
            for count in [1,3]:
                tokens, labels, edits = list(source['tokens']), [0]*len(source['tokens']), []
                for pos in selected[:count]:
                    tokens[pos] = replacements[pos]; labels[pos] = 1
                    edits.append(dict(index=pos, before=source['tokens'][pos], after=tokens[pos]))
                tests.append(record(source, f'{mode}_{count}', tokens, labels, edits))
    payload = dict(dataset=name, fit=fit, cal=cal, test=tests, metadata=dict(
        version=VERSION, source=SOURCES[name], source_pool_sha256=file_hash(args.data_dir/'sources'/f'{name}.json'),
        lexicon_sha256=file_hash(args.data_dir/'lexicon.txt'), data_seed=args.data_seed, corruption_seed=args.corruption_seed,
        label_definition='Generator-modified word = 1; unmodified word = 0. Not exhaustive natural-error annotation.',
        lexical_definition='Typos/gibberish excluded from the pinned English lexicon; lexicon exclusion does not prove linguistic invalidity.',
        conditions=['typo_1','typo_3','gibberish_1','gibberish_3'], source_filter=pool['filter'],
        stats={k:stats(v) for k,v in [('fit',fit),('cal',cal),('test',tests)]}))
    validate(payload, words)
    return payload





# -----------------------------------------------------------------------------
# Thin standalone CLI; the original controlled-v1 functions above are unchanged.
# -----------------------------------------------------------------------------

def ensure_tokenad(data_dir, names=('sms_spam', 'restaurant_review', 'grammar'), *, offline=False):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in TOKENCORE:
            raise ValueError(f'Unknown TokenCore dataset: {name}')
        path = data_dir / (name + '.npy')
        if not path.exists():
            if offline:
                raise FileNotFoundError(f'{path} is missing in offline mode')
            url = (f'https://raw.githubusercontent.com/charles-cao/TokenCore/'
                   f'{TOKENCORE_COMMIT}/data/{name}.npy')
            download(url, path)
        observed = file_hash(path)
        if observed != TOKENCORE[name]:
            raise RuntimeError(f'{name}: checksum mismatch. Expected TokenCore official bytes; '
                               f'got {observed}. Delete {path} only after checking the source.')
        raw = np.load(path, allow_pickle=True).item()  # trusted pinned official .npy only
        if not {'tokens', 'labels'} <= set(raw):
            raise ValueError(f'{path}: expected tokens/labels keys')
        if len(raw['tokens']) != len(raw['labels']):
            raise ValueError(f'{path}: token-label document count mismatch')
        for i, (tokens, labels) in enumerate(zip(raw['tokens'], raw['labels'])):
            if not tokens or len(tokens) != len(labels):
                raise ValueError(f'{path}: invalid word-level labels in document {i}')
        print(f'[original] {name}: {len(raw["tokens"])} documents, sha256={observed[:16]}', flush=True)
    return data_dir


def ensure_controlled(data_dir, names=('ag_news', 'dbpedia'), *, offline=False):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    p = argparse.Namespace(data_dir=data_dir, normal_docs=1000, cal_docs=200,
                           test_sources=500, data_seed=20261008, corruption_seed=271828)
    for name in names:
        if name not in SOURCES:
            raise ValueError(f'Unknown controlled corpus: {name}')
        manifest = data_dir/'prepared'/f'{name}_1000_200_500_20261008_271828.json'
        if not manifest.exists():
            if offline:
                raise FileNotFoundError(f'{manifest} is missing in offline mode')
            print(f'[controlled] building {name}, including pinned public dataset and lexicon downloads', flush=True)
            prepared = prepare(name, p)
            serialized = json.dumps(prepared, ensure_ascii=False, indent=2).encode('utf-8')
            if hashlib.sha256(serialized).hexdigest() != CONTROLLED_MANIFEST_SHA256[name]:
                raise ValueError(f'{name}: generated bytes do not match the published snapshot; no manifest written')
            atomic_json(manifest, prepared)
        prepared = json.loads(manifest.read_text(encoding='utf-8'))
        if prepared.get('dataset') != name or any(k not in prepared for k in ('fit','cal','test','metadata')):
            raise ValueError(f'{manifest}: invalid manifest schema')
        expected_stats = {'fit':800, 'cal':200, 'test':2500}
        for split, count in expected_stats.items():
            if len(prepared[split]) != count:
                raise ValueError(f'{name} {split}: expected {count}, got {len(prepared[split])}')
        metadata = prepared['metadata']
        if (metadata.get('source') != SOURCES[name] or metadata.get('version') != VERSION
                or metadata.get('data_seed') != 20261008 or metadata.get('corruption_seed') != 271828):
            raise ValueError(f'{manifest}: source/protocol mismatch')
        if (metadata.get('source_pool_sha256') != CONTROLLED_SOURCE_SHA256[name]
                or metadata.get('lexicon_sha256') != CONTROLLED_LEXICON_SHA256):
            raise ValueError(f'{manifest}: source-pool or lexicon provenance mismatch')
        checksum = file_hash(manifest)
        if checksum != CONTROLLED_MANIFEST_SHA256[name]:
            raise ValueError(f'{name}: controlled dataset sha mismatch ({checksum}). '
                             'This does NOT match the archived five-corpus protocol; do not silently continue.')
        print(f'[controlled] {name}: FIT={len(prepared["fit"])} CAL={len(prepared["cal"])} '
              f'TEST={len(prepared["test"])} sha256={checksum[:16]}', flush=True)
    return data_dir/'prepared'


def main():
    parser=argparse.ArgumentParser(description='Recreate the fixed five-corpus KAIS dataset protocol')
    parser.add_argument('--datasets', nargs='+', choices=tuple(TOKENCORE)+tuple(SOURCES),
                        default=list(TOKENCORE)+list(SOURCES))
    parser.add_argument('--data-dir', default='./data', type=Path)
    parser.add_argument('--offline', action='store_true', help='Verify local files without downloads')
    args=parser.parse_args()
    old=[name for name in args.datasets if name in TOKENCORE]
    new=[name for name in args.datasets if name in SOURCES]
    if old: ensure_tokenad(args.data_dir, old, offline=args.offline)
    if new: ensure_controlled(args.data_dir, new, offline=args.offline)
    print('PASS: requested datasets prepared with pinned upstream sources.')


if __name__ == '__main__':
    main()
