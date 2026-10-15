"""Evaluate one specified official split; no automatic test split is selected."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from src.features.preprocessing import decode_image, load_classes
from src.models.manifest import read_training_manifest

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--classes",default="config/classes.json")
    parser.add_argument("--model",required=True)
    parser.add_argument("--split",required=True,choices=["cis_val","cis_test","trans_val","trans_test"])
    parser.add_argument("--output-dir",default="artifacts/metrics/baseline_cnn")
    args=parser.parse_args()
    try:
        import tensorflow as tf
        from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                     precision_recall_fscore_support)
    except ImportError as exc:
        raise SystemExit(f"Evaluation dependency unavailable: {exc}") from exc
    classes=load_classes(args.classes)
    df=read_training_manifest(args.manifest,classes,require_existing_paths=True)
    subset=df[df.split==args.split]
    if subset.empty: raise ValueError(f"No records in split {args.split}")
    model=tf.keras.models.load_model(args.model)
    y_true=[]; y_pred=[]
    for row in subset.itertuples(index=False):
        x=decode_image(row.image_path)[None,...]
        scores=model.predict(x,verbose=0)[0]
        y_true.append(classes.index(row.mapped_class))
        y_pred.append(int(np.argmax(scores)))
    p,r,f1,support=precision_recall_fscore_support(y_true,y_pred,labels=range(len(classes)),zero_division=0)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    result={
        "split":args.split,"n":len(y_true),"accuracy":float(accuracy_score(y_true,y_pred)),
        "macro_precision":float(np.mean(p)),"macro_recall":float(np.mean(r)),"macro_f1":float(np.mean(f1)),
        "per_class":{c:{"precision":float(p[i]),"recall":float(r[i]),"f1":float(f1[i]),"support":int(support[i])}
                     for i,c in enumerate(classes)},
        "class_order":classes,
        "protocol_note":"Confirm the project-approved ID/OOD protocol. Do not use held-out test metrics for model selection."
    }
    (out/f"{args.split}_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    np.savetxt(out/f"{args.split}_confusion_matrix.csv",confusion_matrix(y_true,y_pred,labels=range(len(classes))),delimiter=",",fmt="%d")
    (out/f"{args.split}_classification_report.json").write_text(
        json.dumps(classification_report(y_true,y_pred,labels=range(len(classes)),target_names=classes,output_dict=True,zero_division=0),indent=2),encoding="utf-8")
    print(json.dumps({"split":args.split,"n":len(y_true),"accuracy":result["accuracy"],"macro_f1":result["macro_f1"]},indent=2))
if __name__=="__main__": main()
