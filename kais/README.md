# KAIS (DiFA+): main-table reproduction

This is the standalone KAIS detector and data preparation for the five main-table datasets. It trains KAIS and writes its four metrics for each dataset to `kais_main_table.csv`. The published baseline rows are not rerun.

From the repository or extracted archive root, use Linux, Python 3.10, and a CUDA-capable GPU. For the reference PyTorch/CUDA setup:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r kais/requirements.txt
python kais/run.py
```

`run.py` prepares and verifies the data automatically. The collaborator ZIP includes the exact five data inputs; only the BERT weights need downloading if they are not already cached. If you already have the model cached, use `python kais/run.py --offline --model-cache-dir /path/to/cache`. Data preparation alone: `python kais/prepare_data.py --data-dir data`.

- **Inputs:** SMS Spam, Restaurant Review, and Grammar from TokenCore, plus the pinned AG News and DBpedia controlled splits. The two new datasets use 800 FIT and 200 CAL documents and 500 test backgrounds per corpus. Only the generated edits are labeled anomalous.
- **Protocol:** BERT-base-uncased, seed 42, 30 teacher epochs, fixed shared scoring. AG News and DBpedia report the equal mean of the four condition-level metrics, each measured against the same clean backgrounds.
- **Repository:** Copy only `kais/` into [DiFA](https://github.com/Vongolia11/DiFA). The script builds the two controlled datasets from pinned upstream data and verifies the resulting hashes. Keep the ZIP's `data/` snapshots out of public commits.
