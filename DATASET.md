# Dataset documentation — CCT20

- Dataset: Caltech Camera Traps (CCT20 benchmark subset)
- Official source: https://lila.science/datasets/caltech-camera-traps
- Associated paper: Sara Beery, Grant Van Horn, Pietro Perona, “Recognition in Terra Incognita,” ECCV 2018.
- License/terms: Community Data License Agreement (permissive variant); verify the current official terms before use.
- Download date: TODO — fill in after downloading.
- Annotation files used: TODO — record exact filenames and source URLs.
- Local image root: `data/raw/CCT20/images` (adjust if archive layout differs).
- Class selection rule: choose the eight most frequent non-empty animal categories from the official training split with at least 500 training images each. If fewer than eight qualify, the PRD allows lowering the threshold to 300 after documenting team approval.
- Final ordered classes: TODO — populate from generated candidate config and freeze with team approval.
- Split policy: preserve official location/camera-aware splits; never randomly split adjacent frames from the same sequence across train and test.
- Filtering: TODO — record missing/unreadable files, unsupported extensions, excluded empty images, excluded non-selected categories, and any other filtering.
- Attribution: cite the dataset and the associated paper above in the report/README.
- Data governance: raw images and large artifacts are local only and must not be committed to Git.
