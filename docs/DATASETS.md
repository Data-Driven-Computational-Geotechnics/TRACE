# Datasets

TRACE is trained and evaluated on the **Sand** (2D) and **Sand-3D** granular benchmarks introduced by Sanchez-Gonzalez et al., *Learning to simulate complex physics with graph networks*, ICML 2020.

## Download

The original TFRecord datasets are hosted by DeepMind:

```
https://console.cloud.google.com/storage/browser/learning_to_simulate_complex_physics
# or directly:
gsutil -m cp -r gs://learning_to_simulate_complex_physics/datasets/Sand    ./raw/Sand
gsutil -m cp -r gs://learning_to_simulate_complex_physics/datasets/Sand-3D ./raw/Sand-3D
```

Each dataset ships `metadata.json` plus `train / valid / test` TFRecord files.

## Conversion to HDF5

Our loaders read HDF5 with one group per trajectory containing the particle-position array of shape `(T, N, dim)`. Convert the TFRecords once (TensorFlow only needed for this step), producing:

```
datasets/008-Sand-2D/
  train.h5   valid.h5   test.h5   metadata.json
datasets/009-Sand-3D/
  train.h5   valid.h5   test.h5   metadata.json
```

`metadata.json` is copied unchanged from the original dataset. The configs in `configs/` reference these folders through the repo-relative path `datasets/`.

## Splits used in the paper

| | Sand-2D | Sand-3D |
| --- | --- | --- |
| Pretraining trajectories | 512 | 128 |
| Fine-tuning trajectories | 256 | 64 |
| Validation | 16 | 16 |
| Test | 30 | 100 |
| Steps per trajectory | 320 | 350 |
