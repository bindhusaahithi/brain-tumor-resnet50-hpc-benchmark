# Data Folder

Store MRI image data locally in this folder. The images are not committed because medical imaging datasets are large and may have licensing or privacy restrictions.

Expected layout:

```text
data/raw/
  Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/
data/processed/
```

Use `data/processed/` for any derived samples, resized images, or metadata created during experiments.
