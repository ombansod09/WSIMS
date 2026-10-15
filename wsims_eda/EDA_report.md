# WSIMS Member 2 — Final EDA report (CCT20)

## Status and evidence boundary

This report recomputes the metadata-level EDA from the five annotation JSON files available in this chat. The ten separately supplied JPEG samples were also inspected for their actual file properties and simple luminance statistics. Member 1's local validation findings are incorporated as **handoff-reported validation results**; the validated manifest, `validation_report.json`, and frozen `config/classes.json` were not available here for independent inspection. No dataset files, labels, splits, or class configuration were modified.

## Executive summary

- Metadata files contain **57,864 image records** and **59,941 annotations**.
- Ten supplied sample JPEGs were all readable, RGB JPEGs at **1024 × 747**. This is only a sample, not a full-image audit.
- Training category presence is imbalanced: opossum has 2,726 images and bobcat 786 among the frozen eight classes (about **3.47×**).
- Metadata audit finds **224 shared `seq_id` values between `train` and `cis_val`**. This remains a material evaluation concern requiring investigation; do not alter official splits.
- Member 1 reports all 57,864 dataset files resolve and decode, all records are `scaled_resolution`, eight exact-duplicate groups (16 files) exist with no cross-split exact-duplicate group, and one `cis_test` duplicate pair has `coyote`/`deer` label disagreement. These findings were not independently re-run because the complete image root and Member 1 manifest/report are not mounted here.
- The frozen eight-class order reported by Member 1 is: `opossum`, `rabbit`, `coyote`, `cat`, `squirrel`, `raccoon`, `dog`, `bobcat`. Preserve it exactly.

## Scope and method

Recomputed category presence by image, split summaries, split-pair sequence/location/filename/image-ID overlap, and per-image metadata rows directly from the five supplied JSON files. Each image is counted once per category it has at least one annotation for; multi-label images can contribute to multiple category counts. Dimensions in the metadata CSV are declared dimensions, not actual decoded dimensions. Actual properties were measured only for the ten uploaded JPEG samples.

## Official split summary

| Official split | Image records | Annotations | Unique sequence IDs | Unique locations |
|---|---:|---:|---:|---:|
| `train` | 13,553 | 14,071 | 5,527 | 10 |
| `cis_val` | 3,484 | 3,582 | 1,421 | 10 |
| `cis_test` | 15,827 | 16,395 | 6,424 | 10 |
| `trans_val` | 1,725 | 1,865 | 589 | 1 |
| `trans_test` | 23,275 | 24,028 | 8,982 | 9 |
| **Total** | **57,864** | **59,941** | — | — |

The five official split definitions are preserved. Do not randomly repartition, merge, or move images. The intended use of `cis_val`/`cis_test` as in-distribution and `trans_val`/`trans_test` as out-of-distribution evaluation should be documented against the project's benchmark protocol; this is a likely interpretation from the split names, not confirmed here.

## Training category presence

Counts are images with at least one annotation for the category, not counts of bounding boxes or independent animal encounters.

| Category | Training images |
|---|---:|
| `opossum` | 2,726 |
| `rabbit` | 2,443 |
| `coyote` | 1,409 |
| `cat` | 1,287 |
| `squirrel` | 1,250 |
| `raccoon` | 1,027 |
| `dog` | 820 |
| `bobcat` | 786 |
| `car` | 668 |
| `bird` | 484 |
| `rodent` | 375 |
| `skunk` | 229 |
| `deer` | 42 |
| `badger` | 9 |
| `fox` | 5 |
| `empty` | 0 |

The eight frozen classes reported by Member 1 all exceed the PRD's 500-training-image threshold. `bird` has 484 training images, while `car` has 668 but is not an animal category. The frozen mapping takes precedence over independently reselecting classes.

### Unsupported categories

Metadata categories outside the frozen eight include `empty`, `bird`, `rodent`, `skunk`, `deer`, `badger`, `car`, and `fox`. A policy decision is still needed before creating the model-training manifest:
1. For a single-label eight-animal classifier, exclude images whose only annotated category is unsupported; do not map them to a supported animal.
2. For images containing both supported and unsupported labels, retain a supported target only if the annotation schema and task semantics confirm that this is valid; otherwise flag for review.
3. Treat `empty` as a negative/background case only if the modeling design explicitly supports a background/empty class. It is not one of the eight frozen animal classes.
4. Apply filtering consistently and document per-split counts. Never remove records from or change the official source split; make any task-specific filtered manifest as a derived artifact.

This is a recommendation, not a policy already approved by the team.

## Imbalance

Among the eight frozen classes, opossum (2,726) is about 3.47 times as frequent as bobcat (786) in training. Consider training-only class weighting or a sampler after establishing the exact filtered training set. Do not rebalance validation/test data, and report per-class precision/recall/F1 in addition to overall metrics.

## Sequence, location, and overlap audit

The recomputed metadata audit finds:
- `train` vs `cis_val`: 224 shared sequence IDs; 10 shared locations; 0 shared file names; 0 shared image IDs.
- `train` vs `cis_test`: 0 shared sequence IDs; 10 shared locations; 0 shared file names; 0 shared image IDs.
- `train` vs `trans_val`: 0 shared sequence IDs; 0 shared locations; 0 shared file names; 0 shared image IDs.
- `train` vs `trans_test`: 0 shared sequence IDs; 0 shared locations; 0 shared file names; 0 shared image IDs.
- `cis_val` vs `cis_test`: 0 shared sequence IDs; 10 shared locations; 0 shared file names; 0 shared image IDs.
- `cis_val` vs `trans_val`: 0 shared sequence IDs; 0 shared locations; 0 shared file names; 0 shared image IDs.
- `cis_val` vs `trans_test`: 0 shared sequence IDs; 0 shared locations; 0 shared file names; 0 shared image IDs.
- `cis_test` vs `trans_val`: 0 shared sequence IDs; 0 shared locations; 0 shared file names; 0 shared image IDs.
- `cis_test` vs `trans_test`: 0 shared sequence IDs; 0 shared locations; 0 shared file names; 0 shared image IDs.
- `trans_val` vs `trans_test`: 0 shared sequence IDs; 0 shared locations; 0 shared file names; 0 shared image IDs.

**Priority issue:** `train` and `cis_val` share 224 `seq_id` values. The metadata analysis does not determine whether these IDs indicate identical bursts, broader camera deployments, or a dataset convention. Inspect the specific sequence records and benchmark documentation. Preserve official splits while documenting the result; do not randomly split adjacent frames. The same 10 location IDs occur in `train`, `cis_val`, and `cis_test`, so those splits are not location-disjoint by metadata. No shared filename or image ID was found across split pairs in the supplied metadata.

Member 1 separately reports eight byte-identical duplicate groups, all within `train` or `cis_test`, with no group crossing splits. Hash duplicates do not rule out resized/recompressed near-duplicates or sequence-related leakage. One pair in `cis_test` reportedly has inconsistent `coyote` and `deer` labels and requires manual annotation review. No duplicate-audit CSV was created as instructed.

## Image appearance and sample-level quality

Visual inspection of the ten supplied samples shows night/infrared-looking scenes, low-contrast animals, small animals, empty-looking scenes, strong bright/overexposed regions, and possible foreground obstruction. These are qualitative sample observations, not estimates of prevalence across all 57,864 images.

`sample_image_properties.csv` records the ten JPEGs' dimensions/mode/format plus mean and standard deviation of grayscale luminance and fractions of sampled pixels below 40 or above 240. These simple thresholds are exploratory indicators only; they are not universal definitions of poor quality. The ten files were all readable 1024 × 747 RGB JPEGs. The full dataset-wide properties in Member 1's handoff are reported separately above.

## Dimension validation

Member 1's handoff reports these actual-vs-metadata patterns, covering all 57,864 records:

| Metadata dimensions | Actual dimensions | Records |
|---|---|---:|
| 2048 × 1494 | 1024 × 747 | 52,912 |
| 2048 × 1536 | 1024 × 768 | 3,511 |
| 2560 × 1920 | 1024 × 768 | 1,441 |

All records were reportedly marked `scaled_resolution`; aspect ratios match within the validator's tolerance. This suggests resizing/downscaling but does not validate labels, image content, or bounding boxes.

## Data-quality findings and next actions

1. **Sequence overlap:** investigate the 224 shared `train`/`cis_val` sequence IDs against the official benchmark definition; do not move records.
2. **Conflicting duplicate labels:** manually inspect the reported `cis_test` coyote/deer pair with image and annotation evidence; no automatic label edits.
3. **Unsupported categories:** get an explicit modeling policy and then generate a derived manifest with per-split retention/exclusion counts.
4. **Near-duplicates:** if time allows, run perceptual-hash or embedding similarity checks on full image files, and report whether candidate matches cross splits. Exact hashes alone are insufficient.
5. **Annotation quality:** spot-check bounding boxes and category labels; this EDA has not validated them.
6. **Evaluation protocol:** document the intended ID/OOD usage and report metrics separately by official split.
7. **Modeling imbalance:** choose weighting/sampling based on filtered train counts only.
8. **Shared interfaces:** consume the validated manifest and frozen class mapping from the repository; do not hard-code machine-specific paths.

## Artifacts

- `split_summary.csv`: split-level counts.
- `class_counts_by_split.csv`: image-level category presence by split.
- `image_metadata_manifest.csv`: per-image metadata and annotation categories.
- `split_overlap_audit.csv`: pairwise split overlaps (not a duplicate-file audit).
- `sample_image_properties.csv`: properties for the ten uploaded JPEG samples.
- `class_distribution_by_split.png`, `split_sizes.png`: plots.
- `eda_cct.py`: reproducible metadata EDA script (does not alter dataset or classes).
