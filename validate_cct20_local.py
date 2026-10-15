#!/usr/bin/env python3
"""CCT20 local validation for the project's small-image folder and separate annotation folder."""
import argparse, csv, json, os, hashlib
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, UnidentifiedImageError

SPLITS=("train","cis_val","cis_test","trans_val","trans_test")
EXTS={".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def same_aspect_ratio(w1,h1,w2,h2,tol=0.003):
    if not all((w1,h1,w2,h2)): return False
    return abs((w1/h1)-(w2/h2)) <= tol

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root",required=True,type=Path)
    ap.add_argument("--image-dir",default="eccv_18_all_images_sm")
    ap.add_argument("--metadata-dir",default="eccv_18_annotation_files")
    ap.add_argument("--classes-file",default="config/classes.json")
    ap.add_argument("--hash-files",action="store_true")
    a=ap.parse_args()
    root=a.project_root.expanduser().resolve()
    ir=(root/a.image_dir).resolve(); mr=(root/a.metadata_dir).resolve()
    cp=(root/a.classes_file).resolve()
    for p,label in ((root,"Project root"),(ir,"Image directory"),(mr,"Metadata directory"),(cp,"Class config")):
        if not p.exists(): raise SystemExit(f"{label} not found: {p}")
    data={}
    for split in SPLITS:
        p=mr/f"{split}_annotations.json"
        if not p.is_file(): raise SystemExit(f"Metadata file not found: {p}")
        data[split]=json.loads(p.read_text(encoding="utf-8"))
    config=json.loads(cp.read_text(encoding="utf-8"))
    frozen_classes=config.get("classes",config.get("class_names",[]))
    original_map=config.get("original_category_id_to_class",{})
    train_categories={str(c["id"]):str(c["name"]) for c in data["train"].get("categories",[])}
    bybase=defaultdict(list); byrel={}; discovered=0
    print(f"Indexing images under {ir}")
    for dp,_,names in os.walk(ir):
        for name in names:
            p=Path(dp)/name
            if p.suffix.lower() not in EXTS: continue
            discovered+=1
            bybase[name.casefold()].append(p)
            byrel[p.relative_to(ir).as_posix().casefold()]=p
    print(f"Image files discovered: {discovered:,}")
    rows=[]; statuses=Counter(); hashes=defaultdict(list); ambiguous=0
    for split,payload in data.items():
        anns=defaultdict(set); ann_counts=Counter()
        for an in payload.get("annotations",[]):
            iid=str(an.get("image_id")); cid=str(an.get("category_id"))
            anns[iid].add(cid); ann_counts[iid]+=1
        imgs=payload.get("images",[])
        for i,meta in enumerate(imgs,1):
            iid=str(meta.get("id","")); fn=str(meta.get("file_name",""))
            norm=fn.replace("\\","/").lstrip("./")
            path=None; match="unmatched"
            direct=ir/norm
            if direct.is_file(): path=direct; match="relative_path"
            elif norm.casefold() in byrel: path=byrel[norm.casefold()]; match="relative_path_case_insensitive"
            else:
                candidates=bybase.get(Path(norm).name.casefold(),[])
                if len(candidates)==1: path=candidates[0]; match="basename"
                elif len(candidates)>1: match="ambiguous_basename"; ambiguous+=1
            aw=ah=sz=sha=fmt=""
            status="missing"
            if path:
                try:
                    sz=path.stat().st_size
                    with Image.open(path) as im: im.verify()
                    with Image.open(path) as im:
                        aw,ah=im.size; fmt=im.format or ""
                    mw=int(meta.get("width") or 0); mh=int(meta.get("height") or 0)
                    if mw and mh and (mw!=aw or mh!=ah):
                        if same_aspect_ratio(mw,mh,aw,ah):
                            status="scaled_resolution"
                        else:
                            status="dimension_mismatch"
                    else: status="ok"
                    if a.hash_files:
                        sha=sha256(path); hashes[sha].append(str(path))
                except (OSError,ValueError,UnidentifiedImageError,SyntaxError):
                    status="unreadable"
            statuses[status]+=1
            cids=anns.get(iid,set())
            labels=sorted({train_categories.get(cid,"unknown:"+cid) for cid in cids})
            mapped=sorted({original_map.get(cid,train_categories.get(cid,"")) for cid in cids})
            rows.append({
                "split":split,"image_id":iid,"file_name_metadata":fn,
                "resolved_path":str(path) if path else "","path_match_method":match,
                "file_validation_status":status,"width_metadata":meta.get("width",""),
                "height_metadata":meta.get("height",""),"width_actual":aw,"height_actual":ah,
                "image_format_actual":fmt,"file_size_bytes":sz,"sha256":sha,
                "seq_id":meta.get("seq_id",""),"location":meta.get("location",""),
                "frame_num":meta.get("frame_num",""),"date_captured":meta.get("date_captured",""),
                "annotation_count":ann_counts[iid],"category_names":"|".join(labels),
                "mapped_labels":"|".join(x for x in mapped if x)
            })
            if i%5000==0: print(f"{split}: {i:,}/{len(imgs):,}")
    out=root/"member1_validation_output"; out.mkdir(exist_ok=True)
    manifest=out/"validated_manifest.csv"
    with manifest.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else [])
        w.writeheader(); w.writerows(rows)
    duplicate_groups=[{"sha256":h,"count":len(set(ps)),"paths":sorted(set(ps))}
                      for h,ps in hashes.items() if len(set(ps))>1]
    report={
        "project_root":str(root),"image_directory":str(ir),"metadata_directory":str(mr),
        "classes_config":str(cp),"classes_status":config.get("status","unspecified"),
        "classes":frozen_classes,"image_files_discovered":discovered,
        "metadata_image_records":len(rows),"validation_status_counts":dict(statuses),
        "ambiguous_basename_records":ambiguous,"hashing_enabled":a.hash_files,
        "exact_duplicate_hash_groups":duplicate_groups,"official_splits_preserved":True,
        "interpretation":{"scaled_resolution":"Actual and metadata dimensions differ but aspect ratios match within tolerance; likely alternate image resolution, not automatically a corrupt file.",
                         "dimension_mismatch":"Aspect ratio differs or dimensions are otherwise inconsistent; manually review."},
        "notes":["A successful decode verifies file readability, not label correctness.",
                 "SHA-256 detects byte-identical files only, not near-duplicate frames.",
                 "Do not alter frozen classes.json or official split assignments without team approval."]
    }
    (out/"validation_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("\nValidation complete.")
    print(f"Manifest: {manifest}")
    print(f"Report: {out/'validation_report.json'}")
    print(f"Statuses: {dict(statuses)}")
    if a.hash_files: print(f"Exact duplicate groups: {len(duplicate_groups)}")
if __name__=="__main__": main()
