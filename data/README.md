# Data Directory

Keep datasets out of Git. Place local datasets outside this repository or under ignored paths.

Suggested local dataset layout:

```text
/home/jack/datasets/
  MVTec-FS/
  MVTec_AD/
```

Suggested project layout after building manifests:

```text
data/
  manifests/
    mvtec_fs.csv
  example_manifest.csv
```

## MVTec-FS Manifest

Assume MVTec-FS has already been downloaded and extracted to:

```bash
/home/jack/datasets/MVTec-FS
```

Build a manifest with:

```bash
mkdir -p data/manifests
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/build_mvtec_fs_manifest.py \
  --dataset-root /home/jack/datasets/MVTec-FS \
  --output data/manifests/mvtec_fs.csv
```

The generated manifest uses this schema:

```csv
image_path,label,split,mask_path,object_name,defect_name,annotation_path,bbox,polygon_count
```

Notes:

- `image_path` is relative to `--dataset-root` by default.
- `label` is the defect type used for few-shot classification.
- `annotation_path` points to the LabelMe JSON if one is found.
- `bbox` is inferred from all LabelMe shape points and can be used by ROI baselines later.
- Use `--absolute-paths` if you want absolute paths in the manifest on a server.

## First Pipeline Check

After building the manifest, run:

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/run_prototype_baseline.py \
  --manifest data/manifests/mvtec_fs.csv \
  --split train \
  --n-way 5 \
  --k-shot 1 \
  --q-queries 5 \
  --episodes 20 \
  --feature-source hash \
  --feature-dim 64
```

This only checks the data pipeline. Hash features are not a meaningful visual baseline.