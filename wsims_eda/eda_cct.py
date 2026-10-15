#!/usr/bin/env python3
"""Reproducible metadata-only EDA for CCT20. Does not edit data/splits/classes."""
import argparse, json
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt

SPLITS = ["train", "cis_val", "cis_test", "trans_val", "trans_test"]
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--metadata-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("reports/eda"))
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    data={s:json.loads((a.metadata_dir/f"{s}_annotations.json").read_text(encoding="utf-8")) for s in SPLITS}
    cats={int(c["id"]):c["name"] for c in data["train"]["categories"]}
    for s,d in data.items():
        if {int(c["id"]):c["name"] for c in d["categories"]} != cats:
            raise ValueError(f"Category definitions differ in {s}")
    sums=[]; counts=[]; rows=[]; overlap=[]; seqs={}; locs={}; records={}
    for s,d in data.items():
        ims=d["images"]; records[s]=ims
        ann=defaultdict(set)
        for x in d["annotations"]: ann[x["image_id"]].add(cats[int(x["category_id"])])
        seqs[s]={x.get("seq_id") for x in ims if x.get("seq_id") is not None}
        locs[s]={x.get("location") for x in ims if x.get("location") is not None}
        sums.append(dict(split=s,images=len(ims),annotations=len(d["annotations"]),unique_seq_ids=len(seqs[s]),unique_locations=len(locs[s]),
                         duplicate_image_ids=len(ims)-len({x.get("id") for x in ims}),
                         duplicate_file_names=len(ims)-len({x.get("file_name") for x in ims})))
        cc=Counter()
        for im in ims:
            labels=ann.get(im.get("id"),set()); cc.update(labels)
            rows.append(dict(split=s,image_id=im.get("id"),file_name=im.get("file_name"),width=im.get("width"),height=im.get("height"),
                seq_id=im.get("seq_id"),location=im.get("location"),frame_num=im.get("frame_num"),
                seq_num_frames=im.get("seq_num_frames"),date_captured=im.get("date_captured"),categories="|".join(sorted(labels))))
        counts.extend(dict(split=s,category=c,images_with_category=cc[c]) for c in cats.values())
    for i,s1 in enumerate(SPLITS):
        for s2 in SPLITS[i+1:]:
            overlap.append(dict(split_a=s1,split_b=s2,shared_seq_ids=len(seqs[s1]&seqs[s2]),
                shared_locations=len(locs[s1]&locs[s2]),
                shared_file_names=len({x.get("file_name") for x in records[s1]}&{x.get("file_name") for x in records[s2]}),
                shared_image_ids=len({x.get("id") for x in records[s1]}&{x.get("id") for x in records[s2]})))
    sdf=pd.DataFrame(sums); cdf=pd.DataFrame(counts); idf=pd.DataFrame(rows); odf=pd.DataFrame(overlap)
    sdf.to_csv(a.output_dir/"split_summary.csv",index=False)
    cdf.to_csv(a.output_dir/"class_counts_by_split.csv",index=False)
    idf.to_csv(a.output_dir/"image_metadata_manifest.csv",index=False)
    odf.to_csv(a.output_dir/"split_overlap_audit.csv",index=False)
    cdf.pivot(index="category",columns="split",values="images_with_category").fillna(0).sort_values("train").plot(kind="barh",figsize=(11,8))
    plt.title("CCT20 image-level category presence by official split"); plt.xlabel("Images with category"); plt.tight_layout()
    plt.savefig(a.output_dir/"class_distribution_by_split.png",dpi=160); plt.close()
    sdf.set_index("split")["images"].plot(kind="bar",figsize=(9,5)); plt.title("CCT20 image count by official split")
    plt.ylabel("Image records"); plt.tight_layout(); plt.savefig(a.output_dir/"split_sizes.png",dpi=160); plt.close()
    print("Wrote metadata EDA to",a.output_dir)
if __name__=="__main__": main()
