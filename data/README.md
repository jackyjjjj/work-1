# Data Directory

Keep datasets out of Git. Place local datasets here or point configs to another path.

Suggested layout:

```text
data/
  manifests/
    mvtec_fs_train.csv
    mvtec_fs_val.csv
  mvtec_fs/
  mvtec_ad/
  visa/
```

Manifest columns expected by the first loaders:

```csv
image_path,label,split,mask_path,object_name,defect_name
```

Only `image_path` and `label` are required. Extra columns are preserved as metadata.