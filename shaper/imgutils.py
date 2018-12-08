from PIL import Image
import numpy as np


def resize(img, w, h):
    result = Image.fromarray(img)
    result = result.resize((w, h))
    return np.array(result)
