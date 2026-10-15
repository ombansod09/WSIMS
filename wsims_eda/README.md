# WSIMS Member 2 EDA deliverables

`EDA_report.md` is the consolidated EDA report. CSVs and plots are reproducible from the five official annotation JSON files. `sample_image_properties.csv` covers only the ten JPEGs supplied in this chat. Member 1's local validation summary is incorporated as handoff-reported evidence, not independently revalidated here because the full image root and validation manifest/report are not available in this runtime.

Run metadata EDA:
```bash
python eda_cct.py --metadata-dir path/to/eccv_18_annotation_files --output-dir reports/eda
```
Requirements: Python 3.10+, pandas, matplotlib. The script never edits the dataset, split membership, or class mapping. It does not create a duplicate-audit CSV.

`split_overlap_audit.csv` is a split-pair metadata overlap table (sequence/location/name/ID intersection counts), not a duplicate-file audit.
