# Reproducible model training

The production model is a MobileNetV3-Small transfer-learning classifier trained
on the seven HAM10000 lesion categories. Training data is not stored in this
repository.

## Data provenance

- Dataset: HAM10000
- Dataset DOI: <https://doi.org/10.7910/DVN/DBW86T>
- Paper DOI: <https://doi.org/10.1038/sdata.2018.161>
- Official ISIC collection: `212`
- Licence: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

The derived model is for non-commercial research and portfolio demonstration
only. It is not a medical device and must not be used for diagnosis.

## Download

Install the [official ISIC CLI](https://github.com/ImageMarkup/isic-cli), then:

```powershell
isic image download --collections 212 --limit 0 C:\datasets\ham10000
curl.exe -L `
  -o C:\datasets\ham10000\HAM10000_metadata.tab `
  https://dataverse.harvard.edu/api/access/datafile/4338392
```

The training command verifies that all 10,015 expected image IDs are present.

## Environment

Create a separate training environment. CUDA is optional; CPU training is
supported but considerably slower.

```powershell
python -m venv .venv-training
.\.venv-training\Scripts\python.exe -m pip install -r requirements-train.txt
```

For a CUDA build of PyTorch, use the install command recommended by the
[official PyTorch selector](https://docs.pytorch.org/get-started/locally/).

## Train and evaluate

```powershell
.\.venv-training\Scripts\python.exe ml\train.py `
  --images-dir C:\datasets\ham10000 `
  --metadata C:\datasets\ham10000\HAM10000_metadata.tab `
  --output-dir models
```

The script:

1. validates metadata and image coverage;
2. creates deterministic train, validation, and test partitions grouped by
   `lesion_id`;
3. prevents images of the same lesion from crossing partitions;
4. trains the classifier head, then fine-tunes the full network;
5. selects the checkpoint using validation balanced accuracy;
6. calibrates probabilities on validation logits;
7. reports untouched test accuracy, balanced accuracy, macro-F1, per-class
   metrics, subgroup slices, and selective coverage;
8. exports ONNX and verifies numerical parity with ONNX Runtime.

Generated evaluation files contain aggregate statistics and plots only. No
patient images are copied into `models/` or `docs/`.

If training completes but a later export or reporting step is interrupted, the
saved checkpoint can be evaluated without retraining:

```powershell
.\.venv-training\Scripts\python.exe ml\train.py `
  --images-dir C:\datasets\ham10000 `
  --metadata C:\datasets\ham10000\HAM10000_metadata.tab `
  --output-dir models `
  --workers 0 `
  --evaluate-checkpoint `
  --history-log C:\logs\training.out.log
```

After a successful run, build the deployable checksum and runtime card:

```powershell
.\.venv-training\Scripts\python.exe ml\build_model_config.py `
  --models-dir models
```

`model-config.json` binds preprocessing, calibration, evaluation evidence, and
the model SHA-256 to one versioned runtime artifact.
