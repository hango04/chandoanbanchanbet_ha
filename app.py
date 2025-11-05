# app.py
import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2, os, io, zipfile, tempfile, requests, math

st.set_page_config(page_title="🦶 Dự đoán bàn chân (X-ray)", layout="centered")
st.title("🦶 Dự đoán nhãn bàn chân từ ảnh (1 ảnh)")
st.caption("Hỗ trợ: .keras, .h5, .tflite, SavedModel (.zip), .onnx (onnxruntime), TorchScript .pt/.pth")

LABEL_MAP = {0:"Binh thuong",1:"Bet nhe",2:"Bet trung bình",3:"Bet nang",4:"Khong xac dinh"}

# ============== Utils ==============

def ensure_3ch(x1ch): 
    return np.repeat(x1ch, 3, axis=-1)

def normalize_img(gray, mode):
    if mode == "neg1_1":
        return (gray/127.5 - 1.0).astype("float32")
    elif mode == "imagenet":
        # trả về 3 kênh sau khi scale 0..1 rồi chuẩn hoá mean/std ImageNet
        x = (gray/255.0).astype("float32")
        x = np.expand_dims(x, -1)
        x = ensure_3ch(x)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (x - mean) / std
    else:  # "0_1"
        return (gray/255.0).astype("float32")

def preprocess_image_for_shape(image_bytes, target_hw=(224,224), channels=3, norm="neg1_1", keep_gray_pipeline=True):
    fb = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(fb, cv2.IMREAD_COLOR)
    if bgr is None: 
        return None, None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # grayscale pipeline (hist eq + sharpen) cho X-ray
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if keep_gray_pipeline else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
    gray = cv2.filter2D(gray, -1, kernel)

    H, W = target_hw
    gray = cv2.resize(gray, (W, H))

    if norm == "imagenet":
        x = normalize_img(gray, norm)  # đã thành (H,W,3)
    else:
        x = normalize_img(gray, norm)  # (H,W)
        x = np.expand_dims(x, -1)      # (H,W,1)
        if channels == 3:
            x = ensure_3ch(x)

    x = np.expand_dims(x, 0)           # (1,H,W,C)
    return rgb, x.astype(np.float32)

def _save_temp(suffix, data):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data); tmp.flush(); tmp.close()
    return tmp.name

def _softmax_if_needed(y):
    # y: (1, num_classes) hoặc (1,*,num_classes)
    if y.ndim > 2:
        y = y.reshape((y.shape[0], -1))
    row = y[0]
    s = np.sum(row)
    # nếu sum không gần 1 và giá trị không âm -> softmax
    if not (0.98 <= s <= 1.02):
        # tránh overflow
        m = np.max(row)
        ex = np.exp(row - m)
        y = ex / np.sum(ex)
        y = y.reshape(1, -1)
        return y.astype(np.float32)
    return y.astype(np.float32)

# ============== Loaders ==============

@st.cache_resource(show_spinner=False)
def load_keras_or_h5(path):
    model = load_model(path)
    h,w,c = [int(model.inputs[0].shape[i]) for i in (1,2,3)]
    def predict_fn(x): 
        y = model.predict(x.astype(np.float32), verbose=0)
        return _softmax_if_needed(y)
    return predict_fn, (h,w,c), "neg1_1"

@st.cache_resource(show_spinner=False)
def load_savedmodel_zip(zip_bytes):
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf: 
        zf.extractall(tmpdir)
    model = tf.keras.models.load_model(tmpdir)
    h,w,c = [int(model.inputs[0].shape[i]) for i in (1,2,3)]
    def predict_fn(x): 
        y = model.predict(x.astype(np.float32), verbose=0)
        return _softmax_if_needed(y)
    return predict_fn, (h,w,c), "neg1_1"

@st.cache_resource(show_spinner=False)
def load_tflite(path):
    inter = tf.lite.Interpreter(model_path=path); inter.allocate_tensors()
    in_det, out_det = inter.get_input_details()[0], inter.get_output_details()[0]
    ishape = in_det["shape"]
    h,w,c = int(ishape[1]), int(ishape[2]), int(ishape[3])
    in_dtype = in_det["dtype"]

    def predict_fn(x):
        x_in = x
        # nếu model yêu cầu uint8 thì scale về 0..255
        if in_dtype == np.uint8:
            x_in = np.clip(x_in, 0.0, 1.0) * 255.0
            x_in = x_in.astype(np.uint8)
        else:
            x_in = x_in.astype(in_dtype)
        inter.set_tensor(in_det["index"], x_in)
        inter.invoke()
        y = inter.get_tensor(out_det["index"]).astype(np.float32)
        return _softmax_if_needed(y)

    # mặc định norm cho TFLite thường là [-1,1] hoặc [0,1]; chọn an toàn là [-1,1]
    return predict_fn, (h,w,c), "neg1_1"

@st.cache_resource(show_spinner=False)
def load_onnx(path):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    in0 = sess.get_inputs()[0]
    in_name = in0.name
    ishape = in0.shape
    is_nchw = (len(ishape)==4 and isinstance(ishape[1], int) and ishape[1] in (1,3))
    if is_nchw: 
        c,h,w = int(ishape[1]), int(ishape[2]), int(ishape[3])
    else:
        h,w,c = int(ishape[1]), int(ishape[2]), int(ishape[3])

    def predict_fn(x):
        x_in = x.astype(np.float32)
        if is_nchw:
            x_in = np.transpose(x_in, (0,3,1,2))
        y = sess.run(None, {in_name: x_in})[0]
        return _softmax_if_needed(y)

    return predict_fn, (h,w,c), "neg1_1"

@st.cache_resource(show_spinner=False)
def load_torchscript(path):
    import torch
    dev = torch.device("cpu")
    try:
        model = torch.jit.load(path, map_location=dev)
    except Exception as e:
        raise RuntimeError(f"Không load được TorchScript (.pt/.pth): {e}")

    # TorchScript không luôn để lộ input shape → cho phép nhập tay ở UI
    def predict_fn_with_shape(x, nchw=False):
        with torch.no_grad():
            xt = torch.from_numpy(x.astype(np.float32))
            if nchw:  # (N,C,H,W)
                xt = xt.permute(0,3,1,2).contiguous()
            y = model(xt)
            y = y.detach().cpu().numpy().astype(np.float32)
            return _softmax_if_needed(y)
    # trả None để báo cần nhập tay
    return predict_fn_with_shape, None, "imagenet"

@st.cache_resource(show_spinner=False)
def download_to_tmp(url, filename_hint="model.keras"):
    if "drive.google.com" in url:
        try:
            import gdown
        except ImportError:
            raise RuntimeError("Thiếu gdown. Thêm vào requirements.txt: gdown")
        out = os.path.join(tempfile.gettempdir(), filename_hint)
        gdown.download(url, out, quiet=False, fuzzy=True)
        return out
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    out = os.path.join(tempfile.gettempdir(), filename_hint)
    with open(out, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024*1024):
            if chunk: f.write(chunk)
    return out

# ============== UI: chọn nguồn model ==============

st.subheader("1) Nguồn model")
src = st.radio("Chọn nguồn:", ["Upload file", "File trong app", "Tải từ URL"], horizontal=True)

predict_fn = None
input_shape = (224,224,3)
norm_default = "neg1_1"  # MobileNetV2
torchscript = False       # cờ cho .pt/.pth
torchscript_nchw = True   # giả định NCHW cho Torch
manual_input = False

def _load_by_ext(ext, path_or_bytes, name_for_log="model"):
    ext = ext.lower()
    if ext in [".keras", ".h5"]:
        return load_keras_or_h5(path_or_bytes)
    elif ext == ".tflite":
        return load_tflite(path_or_bytes)
    elif ext == ".onnx":
        return load_onnx(path_or_bytes)
    elif ext == ".zip":
        return load_savedmodel_zip(path_or_bytes if isinstance(path_or_bytes, bytes) else open(path_or_bytes,"rb").read())
    elif ext in [".pt", ".pth"]:
        return load_torchscript(path_or_bytes)
    else:
        raise RuntimeError(f"Định dạng không hỗ trợ: {ext}")

if src == "Upload file":
    mf = st.file_uploader("Chọn model (.keras, .h5, .tflite, .zip, .onnx, .pt, .pth)",
                          type=["keras","h5","tflite","zip","onnx","pt","pth"])
    if mf:
        ext = os.path.splitext(mf.name)[1].lower()
        try:
            with st.spinner("Đang tải model..."):
                if ext in [".keras",".h5",".tflite",".onnx",".pt",".pth"]:
                    path = _save_temp(ext, mf.read())
                    loaded = _load_by_ext(ext, path, mf.name)
                elif ext==".zip":
                    loaded = _load_by_ext(ext, mf.read(), mf.name)
                predict_fn, inferred_shape, norm_default = loaded
                if inferred_shape is None:
                    # TorchScript → cần nhập tay
                    torchscript = True
                    manual_input = True
                    st.warning("TorchScript không suy ra được input shape. Hãy nhập tay ở mục bên dưới.")
                else:
                    input_shape = inferred_shape
                st.success(f"✅ Model: {mf.name} | Input: {input_shape if inferred_shape else 'chưa biết'}")
        except Exception as e:
            st.error(f"Không load được model: {e}")

elif src == "File trong app":
    local_path = st.text_input("Đường dẫn trong app", value="flatfoot_model_best.keras")
    if st.button("🔄 Nạp model"):
        if not os.path.exists(local_path):
            st.error(f"Không thấy file: {local_path}")
        else:
            try:
                ext = os.path.splitext(local_path)[1].lower()
                loaded = _load_by_ext(ext, local_path, local_path)
                predict_fn, inferred_shape, norm_default = loaded
                if inferred_shape is None:
                    torchscript = True
                    manual_input = True
                    st.warning("TorchScript không suy ra được input shape. Hãy nhập tay ở mục bên dưới.")
                else:
                    input_shape = inferred_shape
                st.success(f"✅ Loaded | Input: {input_shape if inferred_shape else 'chưa biết'}")
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
                loaded = _load_by_ext(ext, path, fname)
                predict_fn, inferred_shape, norm_default = loaded
                if inferred_shape is None:
                    torchscript = True
                    manual_input = True
                    st.warning("TorchScript không suy ra được input shape. Hãy nhập tay ở mục bên dưới.")
                else:
                    input_shape = inferred_shape
            st.success(f"✅ Downloaded & loaded | Input: {input_shape if inferred_shape else 'chưa biết'}")
        except Exception as e:
            st.error(f"Lỗi tải/nạp: {e}")

# ============== Tuỳ chọn input & chuẩn hoá ==============

st.subheader("2) Cấu hình input & chuẩn hoá")

colA, colB, colC = st.columns(3)
with colA:
    norm_to_use = st.selectbox("Chuẩn hoá", ["neg1_1","0_1","imagenet"], index=["neg1_1","0_1","imagenet"].index(norm_default))
with colB:
    keep_gray = st.checkbox("Dùng pipeline xám (X-ray)", value=True)
with colC:
    nchw_torch = st.checkbox("TorchScript dùng NCHW", value=True)

if manual_input:
    st.info("Nhập tay kích thước cho model (ví dụ 224×224×3).")
    i1, i2, i3 = st.columns(3)
    H = i1.number_input("H", value=224, min_value=16, max_value=4096, step=8)
    W = i2.number_input("W", value=224, min_value=16, max_value=4096, step=8)
    C = i3.number_input("C", value=3, min_value=1, max_value=4, step=1)
    input_shape = (int(H), int(W), int(C))

# ============== Ảnh & dự đoán ==============

st.subheader("3) Ảnh & Dự đoán")
img = st.file_uploader("Chọn ảnh X-ray", type=["jpg","jpeg","png","bmp","tif","tiff"])

if st.button("🚀 Dự đoán"):
    if predict_fn is None:
        st.error("Vui lòng nạp model trước.")
    elif img is None:
        st.error("Vui lòng chọn ảnh.")
    else:
        H,W,C = input_shape
        with st.spinner("Tiền xử lý & suy luận…"):
            rgb, x = preprocess_image_for_shape(img.read(), (H,W), C, norm_to_use, keep_gray_pipeline=keep_gray)
            if rgb is None:
                st.error("Không đọc được ảnh.")
            else:
                try:
                    if torchscript:
                        # predict_fn là hàm yêu cầu biết NCHW/NHWC
                        probs = predict_fn(x, nchw=nchw_torch).astype(np.float32)
                    else:
                        probs = predict_fn(x).astype(np.float32)

                    cls, conf = int(np.argmax(probs[0])), float(np.max(probs[0]))
                    desc = LABEL_MAP.get(cls, f"Label {cls}")
                    st.success(f"**Kết quả:** `{cls}` – **{desc}** | **{conf:.2%}**")

                    text = f"Nhan {cls}: {desc} ({conf:.2%})"
                    h_img, w_img = rgb.shape[:2]
                    scale = max(0.6, min(1.2, w_img/800))
                    cv2.putText(rgb, text, (20, int(40*scale)),
                                cv2.FONT_HERSHEY_SIMPLEX, scale, (0,255,0), 2, cv2.LINE_AA)
                    st.image(rgb, caption="Ảnh có gắn nhãn dự đoán", use_container_width="stretch")

                    st.markdown("#### Xác suất từng lớp")
                    for i, p in enumerate(probs[0].tolist()):
                        st.write(f"- **{i}** ({LABEL_MAP.get(i,str(i))}): {p:.6f}")
                except Exception as e:
                    st.error(f"Lỗi suy luận: {e}")

st.divider()
st.caption(
    "Gợi ý: Model lớn nên để sẵn trong repo (Git LFS) hoặc tải qua URL (Drive/HF/S3). "
    "Nếu là TorchScript (.pt/.pth) thì ưu tiên file **scripted/trace** (torch.jit.save). "
    "Chuẩn hoá đầu vào có thể khác giữa các model → hãy chọn đúng mục 'Chuẩn hoá'."
)
