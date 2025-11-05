# app.py
import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2, os, io, zipfile, tempfile
from PIL import Image, ImageDraw, ImageFont  # <-- thêm PIL để vẽ chữ có dấu

st.set_page_config(page_title="🦶 Flatfoot X-ray Classifier", layout="centered")
st.title("🦶 Phân loại bàn chân bẹt từ X-ray")
st.caption("Hỗ trợ model: .keras, .h5, .tflite, SavedModel (.zip)")

# ==== Labels ====
LABEL_MAP = {
    0: "Bình thường",
    1: "Bẹt nhẹ",
    2: "Bẹt trung bình",
    3: "Bẹt nặng",
    4: "Không xác định",
}

# ==== Utils ====

def ensure_3ch(x):
    return np.repeat(x, 3, axis=-1)

def preprocess(image_bytes, input_shape, norm="0_1"):
    H, W, C = input_shape
    data = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # grayscale + enhance (đúng pipeline bạn train)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    gray = cv2.filter2D(gray, -1, k)

    gray = cv2.resize(gray, (W, H))
    x = gray.astype("float32") / 255.0  # chuẩn hoá [0,1]

    x = np.expand_dims(x, -1)
    if C == 3:
        x = ensure_3ch(x)
    x = np.expand_dims(x, 0)

    return rgb, x

def save_tmp(ext, data):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    f.write(data)
    f.close()
    return f.name

def softmax_if_needed(y):
    if y.ndim > 2:
        y = y.reshape((y.shape[0], -1))
    m = np.max(y[0])
    e = np.exp(y[0] - m)
    s = e / np.sum(e)
    return s.reshape(1, -1).astype(np.float32)

def draw_vn_text(rgb_np, text, xy=(20, 20), size=36, color=(0, 255, 0)):
    """Vẽ chữ Unicode bằng PIL để không lỗi dấu tiếng Việt."""
    pil = Image.fromarray(rgb_np)
    draw = ImageDraw.Draw(pil)
    try:
        # đặt font của bạn tại ./fonts/Roboto-Regular.ttf (hoặc Arial.ttf)
        font = ImageFont.truetype("fonts/Roboto-Regular.ttf", size=size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(xy, text, font=font, fill=color)
    return np.array(pil)

# ==== Loaders ====

@st.cache_resource
def load_keras(path):
    m = load_model(path)
    h, w, c = [int(m.inputs[0].shape[i]) for i in (1, 2, 3)]
    return (lambda x: softmax_if_needed(m.predict(x, verbose=0))), (h, w, c)

@st.cache_resource
def load_tflite(path):
    inter = tf.lite.Interpreter(model_path=path)
    inter.allocate_tensors()
    in_det = inter.get_input_details()[0]
    out_det = inter.get_output_details()[0]
    h, w, c = in_det["shape"][1:4]

    def pred(x):
        x2 = x.astype(in_det["dtype"])
        if in_det["dtype"] == np.uint8:
            x2 = (np.clip(x2, 0.0, 1.0) * 255).astype(np.uint8)
        inter.set_tensor(in_det["index"], x2)
        inter.invoke()
        y = inter.get_tensor(out_det["index"])
        return softmax_if_needed(y)

    return pred, (int(h), int(w), int(c))

@st.cache_resource
def load_zip(bytes_data):
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(bytes_data)) as z:
        z.extractall(tmp)
    # SavedModel -> load bằng tf.keras.models.load_model(tmp)
    m = tf.keras.models.load_model(tmp)
    h, w, c = [int(m.inputs[0].shape[i]) for i in (1, 2, 3)]
    return (lambda x: softmax_if_needed(m.predict(x, verbose=0))), (h, w, c)

def load_model_auto(path_or_bytes, ext):
    if ext in [".keras", ".h5"]:
        return load_keras(path_or_bytes)
    if ext == ".tflite":
        return load_tflite(path_or_bytes)
    if ext == ".zip":
        if isinstance(path_or_bytes, str):
            # nếu truyền đường dẫn .zip
            with open(path_or_bytes, "rb") as f:
                return load_zip(f.read())
        return load_zip(path_or_bytes)
    st.error("❌ Không hỗ trợ định dạng này")
    return None, None

# ==== UI ====

st.subheader("1) Nạp model")
mode = st.radio("Chọn:", ["Upload model", "Model local"], horizontal=True)

predict_fn = None
input_shape = (224, 224, 3)

if mode == "Upload model":
    file = st.file_uploader("Tải model", type=["keras", "h5", "tflite", "zip"])
    if file:
        ext = os.path.splitext(file.name)[1].lower()
        p = save_tmp(ext, file.read())
        predict_fn, input_shape = load_model_auto(p, ext)
        if predict_fn is not None:
            st.success(f"✅ Model loaded: input={input_shape}")
else:
    path = st.text_input("Đường dẫn model", "flatfoot_model.tflite")
    if st.button("Load model"):
        if not os.path.exists(path):
            st.error("❌ Không thấy file")
        else:
            ext = os.path.splitext(path)[1].lower()
            predict_fn, input_shape = load_model_auto(path, ext)
            if predict_fn is not None:
                st.success(f"✅ Loaded: input={input_shape}")

st.subheader("2) Ảnh")
img = st.file_uploader("Chọn ảnh X-ray", type=["jpg", "jpeg", "png"])

if st.button("🔍 Dự đoán"):
    if predict_fn is None:
        st.error("⚠️ Chưa nạp model")
    elif img is None:
        st.error("⚠️ Chưa chọn ảnh")
    else:
        rgb, x = preprocess(img.read(), input_shape)
        if rgb is None:
            st.error("❌ Không đọc được ảnh")
        else:
            probs = predict_fn(x)
            cls = int(np.argmax(probs))
            conf = float(np.max(probs))

            st.success(f"✅ **Kết quả:** {LABEL_MAP.get(cls, cls)} ({conf:.1%})")

            # Vẽ text tiếng Việt không lỗi dấu
            overlay_text = f"{LABEL_MAP.get(cls, cls)} {conf:.0%}"
            rgb = draw_vn_text(rgb, overlay_text, xy=(20, 20), size=42, color=(0, 255, 0))

            st.image(rgb, caption="Ảnh + dự đoán", width="stretch")

            st.write("### Xác suất:")
            for i, p in enumerate(probs[0]):
                st.write(f"- **{i} – {LABEL_MAP.get(i, i)}:** {p:.4f}")

st.caption("✅ Chuẩn hoá [0,1] như khi training • Đã fix lỗi font tiếng Việt khi overlay bằng PIL")
