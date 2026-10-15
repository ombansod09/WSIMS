import json
import numpy as np
from PIL import Image
from src.features.mobilenet_preprocessing import image_to_rgb_float
from src.models.mobilenet_model import EXPECTED_CLASSES, load_classes

def test_rgb_preprocessing_shape_range():
    im=Image.new('RGBA',(60,40),(120,80,20,100))
    x=image_to_rgb_float(im)
    assert x.shape==(224,224,3) and x.dtype==np.float32
    assert np.isfinite(x).all() and x.min()>=0 and x.max()<=255

def test_frozen_classes(tmp_path):
    cfg={'status':'FROZEN','class_names':EXPECTED_CLASSES,'class_to_index':{n:i for i,n in enumerate(EXPECTED_CLASSES)}}
    p=tmp_path/'classes.json'; p.write_text(json.dumps(cfg))
    assert load_classes(p)==EXPECTED_CLASSES

def test_wrong_order_rejected(tmp_path):
    cfg={'status':'FROZEN','class_names':EXPECTED_CLASSES[::-1],'class_to_index':{n:i for i,n in enumerate(EXPECTED_CLASSES[::-1])}}
    p=tmp_path/'classes.json'; p.write_text(json.dumps(cfg))
    try: load_classes(p)
    except ValueError: pass
    else: raise AssertionError('Incorrect class order was accepted')

def test_model_load_and_probability_shape_when_tensorflow_available(tmp_path):
    import pytest
    tf = pytest.importorskip('tensorflow')
    from src.models.mobilenet_model import build_model
    model = build_model(weights=None)
    x = np.zeros((1, 224, 224, 3), dtype=np.float32)
    y = model(x, training=False).numpy()
    assert y.shape == (1, 8)
    assert np.isfinite(y).all()
    assert np.allclose(y.sum(axis=1), 1.0, atol=1e-5)
    path = tmp_path / 'smoke.keras'
    model.save(path)
    loaded = tf.keras.models.load_model(path)
    assert loaded(x, training=False).shape == (1, 8)
