import numpy as np
from PIL import Image
from src.features.preprocessing import decode_pil_image

def test_decode_pil_image_rgb_shape_and_range():
    image = Image.new("RGBA", (80, 40), (255, 128, 0, 100))
    arr = decode_pil_image(image)
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.float32
    assert np.isfinite(arr).all()
    assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0
