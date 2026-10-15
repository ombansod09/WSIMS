import json
import pandas as pd
import pytest
from src.models.manifest import read_training_manifest

CLASSES=["opossum","rabbit","coyote","cat","squirrel","raccoon","dog","bobcat"]

def write_manifest(tmp_path, rows):
    path=tmp_path/"manifest.csv"
    pd.DataFrame(rows).to_csv(path,index=False)
    return path

def test_scaled_resolution_is_a_readable_validation_status(tmp_path):
    img=tmp_path/"dummy.jpg"; img.write_bytes(b"placeholder")
    path=write_manifest(tmp_path,[{"image_path":str(img),"mapped_class":"opossum","split":"train","file_status":"scaled_resolution"}])
    df=read_training_manifest(path,CLASSES)
    assert len(df)==1

def test_unsupported_class_fails_closed(tmp_path):
    img=tmp_path/"dummy.jpg"; img.write_bytes(b"x")
    path=write_manifest(tmp_path,[{"image_path":str(img),"mapped_class":"deer","split":"train","file_status":"scaled_resolution"}])
    with pytest.raises(ValueError,match="unsupported/unknown"):
        read_training_manifest(path,CLASSES)

def test_unknown_split_fails(tmp_path):
    img=tmp_path/"dummy.jpg"; img.write_bytes(b"x")
    path=write_manifest(tmp_path,[{"image_path":str(img),"mapped_class":"opossum","split":"random_train","file_status":"ok"}])
    with pytest.raises(ValueError,match="Unknown/non-official"):
        read_training_manifest(path,CLASSES)

def test_missing_file_fails(tmp_path):
    path=write_manifest(tmp_path,[{"image_path":str(tmp_path/"absent.jpg"),"mapped_class":"opossum","split":"train","file_status":"ok"}])
    with pytest.raises(FileNotFoundError):
        read_training_manifest(path,CLASSES)
