# app.py
import streamlit as st
import numpy as np, cv2, os, io, tempfile, zipfile
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="🦶 Flatfoot X-ray Classifier", layout="centered")
st.title("🦶 Phân loại bàn chân bẹt từ X-ray")
st.caption("Hỗ trợ model: .keras, .h5, .tflite, SavedModel (.zip)")

LABEL_MAP = {
    0:"Bình thường",
    1:"Bẹt nhẹ",
    2:"Bẹt trung bình",
    3:"Bẹt nặng",
    4:"Không xác định"
}

def ensure_3ch(x): 
    return np.repeat(x, 3, axis=-1)

def preprocess(image_bytes, input_shape):
    H, W, C = input_shape
    data = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)
    gray = cv2.filter2D(gray, -1, k)

    gray = cv2.resize(gray, (W, H))
    x = gray.astype("float32") / 255.0

    x = np.expand_dims(x, -1)
    if C == 3: x = ensure_3ch(x)
    x = np.expand_dims(x, 0)
    return rgb, x

def softmax(y):
    m = np.max(y[0])
    e = np.exp(y[0]-m)
    s = e / np.sum(e)
    return s.reshape(1,-1)

def save_tmp(ext, data):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    f.write(data); f.close()
    return f.name

@st.cache_resource
def load_keras(path):
    m = load_model(path)
    h,w,c = [int(m.inputs[0].shape[i]) for i in (1,2,3)]
    return lambda x: softmax(m.predict(x,verbose=0)), (h,w,c)

@st.cache_resource
def load_tflite(path):
    inter = tf.lite.Interpreter(model_path=path); inter.allocate_tensors()
    in_det = inter.get_input_details()[0]
    out_det = inter.get_output_details()[0]
    h,w,c = in_det["shape"][1:4]

    def pred(x):
        x2 = x.astype(in_det["dtype"])
        if in_det["dtype"] == np.uint8:
            x2 = (x2 * 255).astype(np.uint8)
        inter.set_tensor(in_det["index"], x2)
        inter.invoke()
        y = inter.get_tensor(out_det["index"])
        return softmax(y)
    return pred, (h,w,c)

@st.cache_resource
def load_zip(bytes_data):
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(bytes_data)) as z: z.extractall(tmp)
    return load_keras(tmp)

def load_model_auto(path, ext):
    if ext in [".keras",".h5"]: return load_keras(path)
    if ext == ".tflite": return load_tflite(path)
    if ext == ".zip": return load_zip(path)
    st.error("❌ Sai định dạng model")
    return None, None

# ==== UI ====

st.subheader("1) Nạp model")
file = st.file_uploader("Tải model", type=["keras","h5","tflite","zip"])

predict_fn = None
input_shape = None

if file:
    ext = os.path.splitext(file.name)[1].lower()
    p = save_tmp(ext, file.read())
    predict_fn, input_shape = load_model_auto(p, ext)
    st.success(f"✅ Model loaded: input={input_shape}")

st.subheader("2) Ảnh")
img = st.file_uploader("Chọn ảnh X-ray", type=["jpg","jpeg","png"])

if st.button("🔍 Dự đoán"):
    if not predict_fn:
        st.error("⚠️ Chưa tải model")
    elif img is None:
        st.error("⚠️ Chưa chọn ảnh")
    else:
        rgb, x = preprocess(img.read(), input_shape)
        if rgb is None:
            st.error("❌ Lỗi đọc ảnh")
        else:
            probs = predict_fn(x)
            cls = int(np.argmax(probs))
            conf = float(np.max(probs))

            st.success(f"✅ **Kết quả:** {LABEL_MAP[cls]} ({conf:.1%})")

            # PIL vẽ chữ chống lỗi font tiếng Việt
            im = Image.fromarray(rgb)
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()
            draw.text((10, 10), f"{LABEL_MAP[cls]} {conf:.0%}", fill=(0,255,0), font=font)

            st.image(im, caption="Ảnh + dự đoán", width="stretch")

            st.write("### Xác suất:")
            for i,p in enumerate(probs[0]):
                st.write(f"**{i} – {LABEL_MAP[i]}:** {p:.4f}")

st.caption("✅ Không lỗi font, chuẩn hoá 0..1 đúng training")
