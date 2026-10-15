# Member 1 handoff checklist

- [ ] Downloaded CCT20 benchmark images and official train/validation annotations from LILA.
- [ ] Recorded download date, license/terms, source URLs, and attribution in `DATASET.md`.
- [ ] Confirmed annotation `file_name` paths resolve against the local image root.
- [ ] Generated candidate class mapping from training split counts; excluded empty/no-animal.
- [ ] If fewer than eight classes reach 500 training images, team approved the 300-image fallback and it is documented.
- [ ] Reviewed `classes.json` with the team lead and changed status to `FROZEN` only after approval.
- [ ] Built `manifest.csv` and ran image validation.
- [ ] Reviewed missing/unreadable/unsupported records; no silent deletion.
- [ ] Reviewed exact duplicates and camera sequence/location leakage.
- [ ] Confirmed train/validation/test are official location-aware splits and are disjoint as required.
- [ ] Shared manifest path, class order, input-size expectation (224x224 RGB), and preprocessing contract with Members 2–5.
- [ ] Added data paths to `.gitignore`; did not commit raw images.
- [ ] Opened a pull request and included commands plus validation summary.
