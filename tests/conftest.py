from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "cv2" not in sys.modules:
    cv2_stub = types.SimpleNamespace(
        IMREAD_COLOR=1,
        COLOR_BGR2GRAY=0,
        INTER_CUBIC=0,
        ADAPTIVE_THRESH_GAUSSIAN_C=0,
        THRESH_BINARY=0,
        imdecode=lambda *args, **kwargs: None,
        cvtColor=lambda *args, **kwargs: None,
        resize=lambda *args, **kwargs: None,
        bilateralFilter=lambda *args, **kwargs: None,
        adaptiveThreshold=lambda *args, **kwargs: None,
    )
    sys.modules["cv2"] = cv2_stub

if "tensorflow" not in sys.modules:
    keras_image = types.SimpleNamespace(load_img=lambda *a, **k: None, img_to_array=lambda *a, **k: None)
    keras_mod = types.SimpleNamespace(
        models=types.SimpleNamespace(load_model=lambda *a, **k: None),
        preprocessing=types.SimpleNamespace(image=keras_image),
    )
    tf_stub = types.SimpleNamespace(keras=keras_mod)
    sys.modules["tensorflow"] = tf_stub
    sys.modules["tensorflow.keras"] = keras_mod
    sys.modules["tensorflow.keras.preprocessing"] = keras_mod.preprocessing
    sys.modules["tensorflow.keras.preprocessing.image"] = keras_image
