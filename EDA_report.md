# WSIMS Member 2 — Initial CCT metadata EDA report

## Scope and limitations
Analyzed the five supplied COCO-style JSON metadata files: `train_annotations.json`, `cis_val_annotations.json`, `cis_test_annotations.json`, `trans_val_annotations.json`, and `trans_test_annotations.json`. The uploaded files contain metadata only; no image files or frozen `config/classes.json` / Member 1 manifest were supplied in this chat. Therefore, image decoding, missing-file checks, exact byte duplicates, and actual sample-image grids could not be completed here. No image-quality results are inferred from metadata.

## Metadata structure
Each JSON contains:
- `info`: dataset attribution/version metadata.
- `categories`: 16 category IDs and names (including `empty`).
- `images`: image-level records with `file_name`, `width`, `height`, `id`, `seq_id`, `location`, `frame_num`, `seq_num_frames`, and `date_captured`.
- `annotations`: image-to-category labels (`image_id`, `category_id`), and for some records `bbox`.

Image counts and category-presence counts are not the same as independent wildlife observations. A camera sequence can contain multiple similar frames, and annotations may include multiple category labels per image.

## Split summary
| Split | Images | Annotations | Unique sequence IDs | Unique locations |
|---|---:|---:|---:|---:|
| train | 13,553 | 14,071 | 5,527 | 10 |
| cis_val | 3,484 | 3,582 | 1,421 | 10 |
| cis_test | 15,827 | 16,395 | 6,424 | 10 |
| trans_val | 1,725 | 1,865 | 589 | 1 |
| trans_test | 23,275 | 24,028 | 8,982 | 9 |

## Training category presence counts
Counts below are the number of training images with at least one annotation for that category, not annotation-box counts.

| Category | Train images |
|---|---:|
| opossum | 2,726 |
| rabbit | 2,443 |
| coyote | 1,409 |
| cat | 1,287 |
| squirrel | 1,250 |
| raccoon | 1,027 |
| dog | 820 |
| bobcat | 786 |
| car | 668 |
| bird | 484 |
| rodent | 375 |
| skunk | 229 |
| deer | 42 |
| badger | 9 |
| fox | 5 |
| empty | 0 |

The eight most frequent non-empty animal categories in the supplied training metadata with at least 500 training images are opossum, rabbit, coyote, cat, squirrel, raccoon, dog, and bobcat. This is a data-driven candidate list only, **not** a replacement for the frozen `config/classes.json`; Member 1/team lead must confirm the final mapping and whether counts refer to image-level category presence or a filtered manifest. The PRD requires eight frequent non-empty animals and excludes `empty`.

## Imbalance and split-leakage observations
- The training metadata is highly imbalanced: the most common animal category (opossum, 2,726 images) has about 3.5 times the count of the least frequent of the candidate eight (bobcat, 786 images).
- `bird` has 484 training images, below the PRD's 500-image threshold. It is not among the top eight eligible animal categories if the threshold is applied to training counts.
- The `cis_val` and `train` files share **224 `seq_id` values**. This needs investigation with Member 1 before treating them as independent evaluation data.
- The metadata shows the same 10 location IDs in `train`, `cis_val`, and `cis_test`. Thus, these split names should not be assumed to guarantee location-disjoint evaluation without confirming the dataset's split semantics.
- No shared `seq_id` values were found across other pairwise split comparisons in this metadata audit. No `file_name` or image `id` duplicates were found within each individual split; cross-split filename/id checks are included in the generated script's future manifest-level audit.
- `trans_val` contains only one location ID, while `trans_test` has nine. Confirm intended semantics and use the official split policy; do not randomly repartition adjacent frames.

## Image dimensions
The metadata declares image width and height, which are exported row-by-row and plotted. This is not a validation of actual image dimensions: files must be available to compare metadata against decoded dimensions. No images were attached, so missing/unreadable files, hashes, exact duplicates, low-light examples, and sample grids remain pending.

## Recommendations
1. Keep official split boundaries and the shared Member 1 manifest; do not move frames between splits to resolve overlap.
2. Ask Member 1 to investigate the 224 `train`/`cis_val` shared sequence IDs and explain the `cis_*`/`trans_*` split definitions. Report metrics separately by official split if that is the benchmark intent.
3. Use only the frozen `config/classes.json` for model class order. Treat the eight categories above as a candidate pending coordination, not an independent class mapping.
4. Address imbalance with training-only class weights or a documented sampler/augmentation policy. Do not oversample validation/test.
5. Once the local image root and manifest are available, run file validation and exact SHA-256 duplicate checks; inspect perceptual/sequence similarity separately because byte hashes cannot identify visually similar frames.
6. For image samples, include both typical and challenging low-light/night examples per selected class where present. Avoid claiming each frame is an independent wildlife sighting.

## Reproduction
Install `pandas`, `matplotlib`, and `Pillow`, then run from the repository root:
```bash
python scripts/eda_cct.py --metadata-dir data/annotations --output-dir reports/eda --image-root data/images
```
Omit `--image-root` if images are not locally available. The script does not edit the shared class mapping or split assignments.
