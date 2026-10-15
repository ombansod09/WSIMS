import json
import pytest
from src.features.preprocessing import load_classes

def test_unfrozen_config_blocks_training(tmp_path):
    p=tmp_path/"classes.json"
    p.write_text(json.dumps({"status":"NOT_FROZEN","class_names":[],"class_to_index":{}}))
    with pytest.raises(ValueError,match="not frozen"):
        load_classes(p)

def test_exact_frozen_mapping_loads(tmp_path):
    names=["opossum","rabbit","coyote","cat","squirrel","raccoon","dog","bobcat"]
    p=tmp_path/"classes.json"
    p.write_text(json.dumps({"status":"FROZEN","class_names":names,"class_to_index":{c:i for i,c in enumerate(names)}}))
    assert load_classes(p)==names
