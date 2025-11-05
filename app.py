# app.py
import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2, os, io, zipfile, tempfile, requests

st.set_page_config(page_title="🦶 Dự đoán bàn chân (X-ray)", layout="centered")
st.title("🦶 Dự đoán nhãn bàn chân từ ảnh (1 ảnh)")
st.caption("Hỗ trợ: .keras, .h5, .tflite, SavedModel (.zip), .onnx (cần onnxruntime).")

LABEL_MAP = {0:"Binh thuong",1:"Bet nhe",2:"Bet trung bình",3:"Bet nang",4:"Khong xac dinh"}

# ---------- utils ----------
def ensure_3ch(x1ch): return np.repeat(x1ch, 3, axis=-1)

def preprocess_image_for_shape(image_bytes, target_hw=(224,224), channels=3, norm="neg1_1"):
    fb = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(fb, cv2.IMREAD_COLOR)
    if bgr is None: return None, None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
    gray = cv2.filter2D(gray, -1, kernel)
    H,W = target_hw
    gray = cv2.resize(gray, (W,H)).astype("float32")
    gray = (gray/127.5 - 1.0) if norm=="neg1_1" else (gray/255.0)
    x = np.expand_dims(gray, -1)
    if channels==3: x = ensure_3ch(x)
    x = np.expand_dims(x, 0)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, x

def _save_temp(suffix, data):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data); tmp.flush(); tmp.close()
    return tmp.name

# ---------- loaders ----------
@st.cache_resource(show_spinner=False)
def load_keras_or_h5(path):
    model = load_model(path)
    h,w,c = [int(model.inputs[0].shape[i]) for i in (1,2,3)]
    def predict_fn(x): return model.predict(x.astype(np.float32), verbose=0)
    return predict_fn, (h,w,c), "neg1_1"

@st.cache_resource(show_spinner=False)
def load_savedmodel_zip(zip_bytes):
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf: zf.extractall(tmpdir)
    model = tf.keras.models.load_model(tmpdir)
    h,w,c = [int(model.inputs[0].shape[i]) for i in (1,2,3)]
    def predict_fn(x): return model.predict(x.astype(np.float32), verbose=0)
    return predict_fn, (h,w,c), "neg1_1"

@st.cache_resource(show_spinner=False)
def load_tflite(path):
    inter = tf.lite.Interpreter(model_path=path); inter.allocate_tensors()
    in_det, out_det = inter.get_input_details()[0], inter.get_output_details()[0]
    ishape = in_det["shape"]; h,w,c = int(ishape[1]),int(ishape[2]),int(ishape[3])
    def predict_fn(x):
        x_in = x.astype(in_det["dtype"])
        inter.set_tensor(in_det["index"], x_in); inter.invoke()
        y = inter.get_tensor(out_det["index"]); return y.astype(np.float32)
    return predict_fn, (h,w,c), "neg1_1"

@st.cache_resource(show_spinner=False)
def load_onnx(path):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    ishape = sess.get_inputs()[0].shape
    is_nchw = (len(ishape)==4 and isinstance(ishape[1],int) and ishape[1] in (1,3))
    if is_nchw: c,h,w = int(ishape[1]),int(ishape[2]),int(ishape[3])
    else:       h,w,c = int(ishape[1]),int(ishape[2]),int(ishape[3])
    def predict_fn(x):
        x_in = x.astype(np.float32)
        if is_nchw: x_in = np.transpose(x_in,(0,3,1,2))
        y = sess.run(None,{in_name:x_in})[0]
        if y.ndim>2: y = y.reshape((y.shape[0],-1))
        return y.astype(np.float32)
    return predict_fn, (h,w,c), "neg1_1"

@st.cache_resource(show_spinner=False)
def download_to_tmp(url, filename_hint="model.keras"):
    # đơn giản cho mọi URL (Google Drive nên dùng gdown nếu link chia sẻ)
    if "drive.google.com" in url:
        try:
            import gdown
        except ImportError:
            raise RuntimeError("Thiếu gdown. Thêm vào requirements.txt: gdown")
        out = os.path.join(tempfile.gettempdir(), filename_hint)
        gdown.download(url, out, quiet=False, fuzzy=True)
        return out
    # URL thường
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    out = os.path.join(tempfile.gettempdir(), filename_hint)
    with open(out, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024*1024):
            if chunk: f.write(chunk)
    return out

# ===== 1) nguồn model =====
st.subheader("1) Nguồn model")
src = st.radio("Chọn nguồn:", ["Upload file", "File trong app", "Tải từ URL"], horizontal=True)

predict_fn = None
input_shape = (224,224,3)
norm_default = "neg1_1"  # MobileNetV2

if src == "Upload file":
    mf = st.file_uploader("Chọn model (.keras, .h5, .tflite, .zip, .onnx)",
                          type=["keras","h5","tflite","zip","onnx"])
    if mf:
        suf = os.path.splitext(mf.name)[1].lower()
        try:
            with st.spinner("Đang tải model..."):
                if suf in [".keras",".h5"]:
                    path = _save_temp(suf, mf.read())
                    predict_fn, input_shape, norm_default = load_keras_or_h5(path)
                elif suf==".tflite":
                    path = _save_temp(".tflite", mf.read())
                    predict_fn, input_shape, norm_default = load_tflite(path)
                elif suf==".zip":
                    predict_fn, input_shape, norm_default = load_savedmodel_zip(mf.read())
                elif suf==".onnx":
                    path = _save_temp(".onnx", mf.read())
                    predict_fn, input_shape, norm_default = load_onnx(path)
            st.success(f"✅ Model: {mf.name} | Input: {input_shape}")
        except Exception as e:
            st.error(f"Không load được model: {e}")

elif src == "File trong app":
    local_path = st.text_input("Đường dẫn trong app", value="flatfoot_model_best.keras")
    if st.button("🔄 Nạp model"):
        if not os.path.exists(local_path):
            st.error(f"Không thấy file: {local_path}")
        else:
            try:
                predict_fn, input_shape, norm_default = load_keras_or_h5(local_path)
                st.success(f"✅ Loaded | Input: {input_shape}")
            except Exception as e:
                st.error(f"Lỗi nạp: {e}")

else:  # Tải từ URL
    url = st.text_input("URL model (Drive/HF/S3…)", placeholder="https://…")
    fname = st.text_input("Tên file lưu tạm", value="model.keras")
    if st.button("⬇️ Tải & nạp"):
        try:
            with st.spinner("Đang tải model từ URL…"):
                path = download_to_tmp(url, fname)
                ext = os.path.splitext(path)[1].lower()
                if ext in [".keras",".h5"]:
                    predict_fn, input_shape, norm_default = load_keras_or_h5(path)
                elif ext==".tflite":
                    predict_fn, input_shape, norm_default = load_tflite(path)
                elif ext==".onnx":
                    predict_fn, input_shape, norm_default = load_onnx(path)
                elif ext==".zip":
                    with open(path,"rb") as f:
                        predict_fn, input_shape, norm_default = load_savedmodel_zip(f.read())
                else:
                    raise RuntimeError(f"Định dạng không hỗ trợ: {ext}")
            st.success(f"✅ Downloaded & loaded | Input: {input_shape}")
        except Exception as e:
            st.error(f"Lỗi tải/nạp: {e}")

# ===== 2) ảnh & dự đoán =====
st.subheader("2) Ảnh & Dự đoán")
img = st.file_uploader("Chọn ảnh X-ray", type=["jpg","jpeg","png","bmp","tif","tiff"])

# dùng đúng chuẩn của model
norm_to_use = norm_default

if st.button("🚀 Dự đoán"):
    if predict_fn is None:
        st.error("Vui lòng nạp model trước.")
    elif img is None:
        st.error("Vui lòng chọn ảnh.")
    else:
        H,W,C = input_shape
        with st.spinner("Tiền xử lý & suy luận…"):
            rgb, x = preprocess_image_for_shape(img.read(), (H,W), C, norm_to_use)
            if rgb is None:
                st.error("Không đọc được ảnh.")
            else:
                try:
                    probs = predict_fn(x).astype(np.float32)
                    cls, conf = int(np.argmax(probs[0])), float(np.max(probs[0]))
                    desc = LABEL_MAP.get(cls, f"Label {cls}")
                    st.success(f"**Kết quả:** `{cls}` – **{desc}** | **{conf:.2%}**")

                    text = f"Nhan {cls}: {desc} ({conf:.2%})"
                    h_img, w_img = rgb.shape[:2]
                    scale = max(0.6, min(1.2, w_img/800))
                    cv2.putText(rgb, text, (20, int(40*scale)),
                                cv2.FONT_HERSHEY_SIMPLEX, scale, (0,255,0), 2, cv2.LINE_AA)
                    st.image(rgb, caption="Ảnh có gắn nhãn dự đoán", use_container_width=True)

                    st.markdown("#### Xác suất từng lớp")
                    for i, p in enumerate(probs[0].tolist()):
                        st.write(f"- **{i}** ({LABEL_MAP.get(i,str(i))}): {p:.6f}")
                except Exception as e:
                    st.error(f"Lỗi suy luận: {e}")

st.divider()
st.caption(
    "Trên Streamlit Cloud: upload thường bị giới hạn ~200 MB và không thể tăng bằng `st.set_option`. "
    "Hãy dùng **File trong app** (đưa .keras vào repo, ưu tiên Git LFS) hoặc **Tải từ URL** để dùng model lớn. "
    "Model MobileNetV2 yêu cầu chuẩn hoá **[-1,1]**; app mặc định đã dùng `neg1_1` để tránh kẹt 1 nhãn."
)
