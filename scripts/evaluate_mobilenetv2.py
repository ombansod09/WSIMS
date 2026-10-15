from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

def main():
    p=argparse.ArgumentParser(description="Evaluate MobileNetV2 on one named official split; never tunes on test")
    p.add_argument('--manifest',required=True); p.add_argument('--split',required=True,choices=['cis_val','cis_test','trans_val','trans_test'])
    p.add_argument('--model',default='artifacts/models/mobilenetv2_best.keras'); p.add_argument('--classes',default='config/classes.json'); p.add_argument('--out-dir',default='artifacts/metrics')
    a=p.parse_args()
    import tensorflow as tf
    from src.models.mobilenet_model import load_classes
    from src.features.mobilenet_preprocessing import IMAGE_SIZE
    classes=load_classes(a.classes); df=pd.read_csv(a.manifest); req={'image_path','mapped_class','split','file_status'}
    if req-set(df.columns): raise ValueError(f'Missing columns: {sorted(req-set(df.columns))}')
    if not set(df.mapped_class).issubset(set(classes)): raise ValueError('Unsupported label found')
    df=df[df.split==a.split].copy()
    if df.empty: raise ValueError(f'No rows for split {a.split}')
    model=tf.keras.models.load_model(a.model); y_true=[]; y_pred=[]; lat=[]
    for row in df.itertuples(index=False):
        raw=tf.io.read_file(row.image_path); im=tf.io.decode_image(raw,channels=3,expand_animations=False); im.set_shape([None,None,3])
        x=tf.image.resize(tf.cast(im,tf.float32),IMAGE_SIZE,method='bilinear',antialias=True)[None,...]
        t=time.perf_counter(); pred=model(x,training=False).numpy()[0]; lat.append((time.perf_counter()-t)*1000)
        y_true.append(classes.index(row.mapped_class)); y_pred.append(int(np.argmax(pred)))
    precision,recall,f1,_=precision_recall_fscore_support(y_true,y_pred,labels=list(range(8)),zero_division=0)
    report={'model':'MobileNetV2','split':a.split,'n':len(df),'accuracy':float(accuracy_score(y_true,y_pred)),
      'macro_precision':float(np.mean(precision)),'macro_recall':float(np.mean(recall)),'macro_f1':float(np.mean(f1)),
      'per_class':classification_report(y_true,y_pred,labels=list(range(8)),target_names=classes,output_dict=True,zero_division=0),
      'confusion_matrix':confusion_matrix(y_true,y_pred,labels=list(range(8))).tolist(),
      'class_order':classes,'latency_ms_per_image_mean':float(np.mean(lat)),'latency_ms_per_image_p50':float(np.percentile(lat,50)),
      'latency_ms_per_image_p95':float(np.percentile(lat,95)), 'latency_note':'single-image wall time around model call; excludes image file decode/resize and is hardware/runtime dependent'}
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); dest=out/f'mobilenetv2_{a.split}_metrics.json'; dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame(report['confusion_matrix'],index=classes,columns=classes).to_csv(out/f'mobilenetv2_{a.split}_confusion_matrix.csv')
    pd.DataFrame(report['per_class']).T.to_csv(out/f'mobilenetv2_{a.split}_classification_report.csv')
    print(json.dumps({k:v for k,v in report.items() if k not in ('per_class','confusion_matrix')},indent=2)); print(dest)
if __name__=='__main__':main()
