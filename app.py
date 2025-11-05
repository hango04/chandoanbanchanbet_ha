# app_streamlit.py
import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2, os, io, zipfile, tempfile, json

st.set_page_config(page_title="🦶 Dự đoán bàn chân (X-ray)", layout="centered")
st.title("🦶 Dự đoán nhãn bàn chân từ ảnh (1 ảnh)")
st.caption("Upload model: .keras, .h5, .tflite, SavedModel (.zip), hoặc .onnx (cần onnxruntime).")

# ====== LABEL MAP CỐ ĐỊNH (không cần upload) ======
LABEL_MAP = {
    0: "Binh thuong",
    1: "Bet nhe",
    2: "Bet trung bình",
    3: "Bet nang",
    4: "Khong xac dinh"
}

# ---------- Utilities ----------
def ensure_3ch(x1ch: np.ndarray) -> np.ndarray:
    # (H,W,1) -> (H,W,3)
    return np.repeat(x1ch, 3, axis=-1)

def preprocess_image_for_shape(image_bytes: bytes, target_hw=(224,224), channels=3, norm="neg1_1"):
    """
    Đọc bytes -> trả (rgb_for_show, x[1,H,W,C] float32)
    - target_hw: (H,W)
    - channels: 1 hoặc 3 (theo input model)
    - norm: "neg1_1" (mặc định, tương đương mb_preprocess) hoặc "0_1"
    """
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR
    if bgr is None:
        return None, None

    # X-ray style tiền xử lý nhẹ
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    gray = cv2.filter2D(gray, -1, kernel)

    h, w = target_hw
    gray = cv2.resize(gray, (w, h)).astype("float32")

    if norm == "0_1":
        gray = gray / 255.0
    else:  # "neg1_1"
        gray = (gray / 127.5) - 1.0

    if channels == 1:
        x = np.expand_dims(gray, axis=-1)  # (H,W,1)
    else:
        x = np.expand_dims(gray, axis=-1)
        x = ensure_3ch(x)                  # (H,W,3)

    x = np.expand_dims(x, axis=0)         # (1,H,W,C)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, x

# ---------- Loaders cho các định dạng ----------
def _save_temp(suffix: str, data: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush(); tmp.close()
    return tmp.name

@st.cache_resource(show_spinner=False)
def load_keras_or_h5(tmp_path: str):
    model = load_model(tmp_path)
    ishape = model.inputs[0].shape  # (None,H,W,C)
    h = int(ishape[1]); w = int(ishape[2]); c = int(ishape[3])

    def predict_fn(x: np.ndarray):
        return model.predict(x.astype(np.float32), verbose=0)

    # Nếu train với MobileNetV2 + mb_preprocess -> dùng -1..1
    return predict_fn, (h, w, c), "neg1_1"

@st.cache_resource(show_spinner=False)
def load_savedmodel_zip(zip_bytes: bytes):
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(tmpdir)
    model = tf.keras.models.load_model(tmpdir)
    ishape = model.inputs[0].shape
    h = int(ishape[1]); w = int(ishape[2]); c = int(ishape[3])

    def predict_fn(x: np.ndarray):
        return model.predict(x.astype(np.float32), verbose=0)

    return predict_fn, (h, w, c), "neg1_1"

@st.cache_resource(show_spinner=False)
def load_tflite(tmp_path: str):
    interpreter = tf.lite.Interpreter(model_path=tmp_path)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    ishape = in_det["shape"]  # [1,224,224,1] hoặc [1,224,224,3]
    h = int(ishape[1]); w = int(ishape[2]); c = int(ishape[3])

    def predict_fn(x: np.ndarray):
        x_in = x.astype(in_det["dtype"])
        interpreter.set_tensor(in_det["index"], x_in)
        interpreter.invoke()
        y = interpreter.get_tensor(out_det["index"])
        return y.astype(np.float32)

    # Thường 0..1, nhưng nếu convert từ Keras MobileNetV2 hãy để -1..1 (user đổi trong UI)
    return predict_fn, (h, w, c), "neg1_1"

@st.cache_resource(show_spinner=False)
def load_onnx(tmp_path: str):
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError("Thiếu onnxruntime. Cài: pip install onnxruntime") from e

    sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    ishape = sess.get_inputs()[0].shape  # [1,C,H,W] hoặc [1,H,W,C]
    # Suy NCHW
    is_nchw = (len(ishape) == 4 and isinstance(ishape[1], int) and ishape[1] in (1,3))
    if is_nchw:
        c = int(ishape[1]); h = int(ishape[2]); w = int(ishape[3])
    else:
        h = int(ishape[1]); w = int(ishape[2]); c = int(ishape[3])

    def predict_fn(x: np.ndarray):
        x_in = x.astype(np.float32)
        if is_nchw:
            x_in = np.transpose(x_in, (0, 3, 1, 2))  # NHWC->NCHW
        preds = sess.run(None, {in_name: x_in})[0]
        if preds.ndim > 2:
            preds = preds.reshape((preds.shape[0], -1))
        return preds.astype(np.float32)

    return predict_fn, (h, w, c), "neg1_1"

# ====== 1) Upload model ======
st.subheader("1) Upload model")
model_file = st.file_uploader(
    "Chọn file model (.keras, .h5, .tflite, .zip SavedModel, .onnx)",
    type=["keras", "h5", "tflite", "zip", "onnx"]
)
predict_fn = None
input_shape = (224, 224, 3)
norm_default = "0_1"


if model_file is not None:
    suffix = os.path.splitext(model_file.name)[1].lower()
    try:
        with st.spinner("Đang tải model..."):
            if suffix in [".keras", ".h5"]:
                path = _save_temp(suffix, model_file.read())
                predict_fn, input_shape, norm_default = load_keras_or_h5(path)
            elif suffix == ".tflite":
                path = _save_temp(".tflite", model_file.read())
                predict_fn, input_shape, norm_default = load_tflite(path)
            elif suffix == ".zip":
                predict_fn, input_shape, norm_default = load_savedmodel_zip(model_file.read())
            elif suffix == ".onnx":
                path = _save_temp(".onnx", model_file.read())
                predict_fn, input_shape, norm_default = load_onnx(path)
        st.success(f"✅ Model đã tải: {model_file.name} | Input shape: {input_shape}")
    except Exception as e:
        st.error(f"Không load được model: {e}")

# ====== 2) Ảnh + Chuẩn hoá ======
st.subheader("2) Ảnh & Chuẩn hoá")
img_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

norm_to_use = "0_1"


if st.button("🚀 Dự đoán"):
    if predict_fn is None:
        st.error("Vui lòng upload model trước.")
    elif img_file is None:
        st.error("Vui lòng chọn một ảnh.")
    else:
        H, W, C = input_shape
        with st.spinner("Đang tiền xử lý & dự đoán..."):
            rgb, x = preprocess_image_for_shape(
                img_file.read(), target_hw=(H, W), channels=C, norm=norm_to_use
            )
            if rgb is None:
                st.error("Không đọc được ảnh. Vui lòng thử ảnh khác.")
            else:
                try:
                    probs = predict_fn(x)
                    cls = int(np.argmax(probs))
                    conf = float(np.max(probs))
                    desc = LABEL_MAP.get(cls, f"Label {cls}")

                    st.success(f"**Kết quả:** Nhãn `{cls}` – **{desc}** với độ tin cậy **{conf:.2%}**")

                    # Vẽ text lên ảnh
                    text = f"Nhan {cls}: {desc} ({conf:.2%})"
                    h_img, w_img = rgb.shape[:2]
                    scale = max(0.6, min(1.2, w_img / 800))
                    cv2.putText(rgb, text, (20, int(40*scale)),
                                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2, cv2.LINE_AA)
                    st.image(rgb, caption="Ảnh có gắn nhãn dự đoán", use_container_width=True)

                    # Bảng xác suất
                    st.markdown("#### Xác suất từng lớp")
                    for i, p in enumerate(probs[0].tolist()):
                        st.write(f"- **{i}** ({LABEL_MAP.get(i, str(i))}): {p:.6f}")

                except Exception as e:
                    st.error(f"Lỗi khi dự đoán: {e}")

st.divider()
st.caption("Gợi ý: Dữ liệu đầu vào nên là **ảnh X-ray bàn chân** giống domain lúc huấn luyện. \
Nếu model của bạn dùng pipeline khác, hãy đổi lựa chọn chuẩn hoá tương ứng.")
