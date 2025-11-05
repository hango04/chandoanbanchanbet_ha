# app.py
import streamlit as st
import numpy as np, os, io, zipfile, tempfile
from PIL import Image, ImageDraw, ImageFont

# ================== CẤU HÌNH UI ==================
st.set_page_config(page_title="🦶 Flatfoot X-ray Classifier", layout="centered")
st.title("🦶 Phân loại bàn chân bẹt từ X-ray")
st.caption("Hỗ trợ model: .keras, .h5, .tflite, SavedModel (.zip)")

# ================== NHÃN ==================
LABEL_MAP = {
    0: "Bình thường",
    1: "Bẹt nhẹ",
    2: "Bẹt trung bình",
    3: "Bẹt nặng",
    4: "Không xác định",
}

# ================== TIỆN ÍCH ==================
def _lazy_cv2():
    try:
        import cv2
        return cv2
    except Exception as e:
        st.error("❌ Lỗi OpenCV. Nếu chạy trên Streamlit Cloud, hãy thêm vào packages.txt: `libgl1` và `libglib2.0-0`.")
        raise

def ensure_3ch(x):
    return np.repeat(x, 3, axis=-1)

def preprocess(image_bytes, input_shape):
    """
    Tiền xử lý ảnh: BGR->GRAY, equalizeHist, sharpen, resize, chuẩn hoá [0,1].
    """
    cv2 = _lazy_cv2()
    H, W, C = input_shape

    data = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    gray = cv2.filter2D(gray, -1, k)
    gray = cv2.resize(gray, (W, H))

    x = gray.astype("float32") / 255.0
    x = np.expand_dims(x, -1)
    if C == 3:
        x = ensure_3ch(x)
    x = np.expand_dims(x, 0)  # (1, H, W, C)

    return rgb, x

def softmax(y):
    if y.ndim > 2:
        y = y.reshape((y.shape[0], -1))
    m = np.max(y[0])
    e = np.exp(y[0] - m)
    s = e / np.sum(e)
    return s.reshape(1, -1).astype(np.float32)

def save_tmp(ext, data):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    f.write(data)
    f.close()
    return f.name

# ---------- Font hỗ trợ tiếng Việt ----------
def _find_font_path():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "assets/Roboto-Regular.ttf",  # nếu bạn commit font vào repo
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None  # fallback mặc định (có thể lỗi dấu)

def draw_vietnamese_text(rgb_np, text, pos=(16, 16)):
    im = Image.fromarray(rgb_np)
    draw = ImageDraw.Draw(im)

    W = im.size[0]
    fs = max(16, int(W * 0.035))  # cỡ chữ theo chiều rộng ảnh
    fp = _find_font_path()
    font = ImageFont.truetype(fp, fs) if fp else ImageFont.load_default()

    draw.text(
        pos,
        text,
        fill=(0, 255, 0),
        font=font,
        stroke_width=max(1, int(fs * 0.10)),
        stroke_fill=(0, 0, 0),
    )
    return np.array(im)

# ================== LOAD MODEL (lười import TF) ==================
@st.cache_resource(show_spinner=False)
def load_keras(path):
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    m = load_model(path)
    h, w, c = [int(m.inputs[0].shape[i]) for i in (1, 2, 3)]

    def pred(x):
        y = m.predict(x, verbose=0)
        return softmax(y)

    return pred, (h, w, c)

@st.cache_resource(show_spinner=False)
def load_tflite(path):
    import tensorflow as tf

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
        return softmax(y)

    return pred, (h, w, c)

@st.cache_resource(show_spinner=False)
def load_zip(bytes_data):
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(bytes_data)) as z:
        z.extractall(tmp)
    return load_keras(tmp)

def load_model_auto(path_or_bytes, ext):
    ext = ext.lower()
    if ext in [".keras", ".h5"]:
        return load_keras(path_or_bytes)
    if ext == ".tflite":
        return load_tflite(path_or_bytes)
    if ext == ".zip":
        # nếu truyền bytes -> đọc luôn
        if isinstance(path_or_bytes, (bytes, bytearray)):
            return load_zip(path_or_bytes)
        # nếu là đường dẫn zip
        with open(path_or_bytes, "rb") as f:
            return load_zip(f.read())
    st.error("❌ Định dạng model không được hỗ trợ.")
    return None, None

# ================== UI ==================
st.subheader("1) Nạp model")
model_file = st.file_uploader("Tải model", type=["keras", "h5", "tflite", "zip"])

predict_fn, input_shape = None, None
if model_file:
    ext = os.path.splitext(model_file.name)[1].lower()
    # .zip cần bytes; các loại khác có thể lưu tạm
    if ext == ".zip":
        raw = model_file.read()
        try:
            predict_fn, input_shape = load_model_auto(raw, ext)
            st.success(f"✅ Model loaded: input={input_shape}")
        except Exception as e:
            st.exception(e)
    else:
        p = save_tmp(ext, model_file.read())
        try:
            predict_fn, input_shape = load_model_auto(p, ext)
            st.success(f"✅ Model loaded: input={input_shape}")
        except Exception as e:
            st.exception(e)

st.subheader("2) Ảnh")
img = st.file_uploader("Chọn ảnh X-ray", type=["jpg", "jpeg", "png"])

if st.button("🔍 Dự đoán"):
    if not predict_fn:
        st.error("⚠️ Chưa tải model.")
    elif img is None:
        st.error("⚠️ Chưa chọn ảnh.")
    else:
        try:
            rgb, x = preprocess(img.read(), input_shape)
            if rgb is None:
                st.error("❌ Lỗi đọc ảnh.")
            else:
                probs = predict_fn(x)
                cls = int(np.argmax(probs))
                conf = float(np.max(probs))
                st.success(f"✅ **Kết quả:** {LABEL_MAP.get(cls, cls)} ({conf:.1%})")

                # Vẽ chữ tiếng Việt đúng dấu bằng PIL + viền đậm
                rgb = draw_vietnamese_text(rgb, f"{LABEL_MAP.get(cls, cls)} {conf:.0%}", pos=(16, 16))
                st.image(rgb, caption="Ảnh + dự đoán", width="stretch")

                st.write("### Xác suất:")
                for i, p in enumerate(probs[0]):
                    st.write(f"- **{i} – {LABEL_MAP.get(i, i)}:** {p:.4f}")
        except Exception as e:
            st.exception(e)

st.caption(
    "Mẹo: Nếu chạy cloud mà lỗi font, thêm `fonts-dejavu-core` hoặc `fonts-noto-core` vào **packages.txt**. "
    "Nếu lỗi `libGL.so.1` → thêm `libgl1` và `libglib2.0-0`."
)
