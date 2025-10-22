import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import tempfile, os, zipfile

# ====== Cấu hình trang ======
st.set_page_config(page_title="Dự đoán bàn chân", layout="centered")

st.title("🦶 Dự đoán nhãn bàn chân từ ảnh (1 ảnh)")
st.caption("Hỗ trợ model: .keras, .h5, .tflite, SavedModel (.zip). ONNX (.onnx) nếu có onnxruntime.")

# ====== Ánh xạ nhãn ======
LABEL_MAP = {
    0: "Binh thuong",
    1: "Bet nhe",
    2: "Bet trung bình",
    3: "Bet nang",
    4: "Khong xac dinh"
}

# ---------- Utilities ----------
def ensure_3ch(x1ch):
    # (H,W,1) -> (H,W,3)
    return np.repeat(x1ch, 3, axis=-1)

def preprocess_image_for_shape(image_bytes, target_hw=(224,224), channels=1, norm="0_1"):
    """
    Đọc bytes -> trả (rgb_for_show, x[1,H,W,C] float32)
    - target_hw: (H,W)
    - channels: 1 hoặc 3
    - norm: "0_1" (mặc định) hoặc "-1_1"
    """
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR
    if bgr is None:
        return None, None

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
    elif norm == "-1_1":
        gray = (gray / 127.5) - 1.0

    if channels == 1:
        x = np.expand_dims(gray, axis=-1)  # (H,W,1)
    else:
        x = np.expand_dims(gray, axis=-1)  # (H,W,1)
        x = ensure_3ch(x)                  # (H,W,3)

    x = np.expand_dims(x, axis=0)         # (1,H,W,C)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), x


# ----- Loaders cho các định dạng -----
def load_keras_or_h5(tmp_path):
    model = load_model(tmp_path)
    # Suy input shape
    ishape = model.inputs[0].shape  # (None,H,W,C)
    h = int(ishape[1]); w = int(ishape[2]); c = int(ishape[3])
    def predict_fn(x):
        # đảm bảo dtype float32 cho Keras
        probs = model.predict(x.astype(np.float32), verbose=0)
        return probs
    return predict_fn, (h, w, c), "0_1"  # đa số model của bạn dùng [0,1]

def load_savedmodel_zip(zip_file_bytes):
    # Giải nén zip ra thư mục tạm rồi load_model
    tmpdir = tempfile.mkdtemp()
    zf = zipfile.ZipFile(zip_file_bytes)
    zf.extractall(tmpdir)
    zf.close()
    model = tf.keras.models.load_model(tmpdir)
    ishape = model.inputs[0].shape
    h = int(ishape[1]); w = int(ishape[2]); c = int(ishape[3])
    def predict_fn(x):
        return model.predict(x.astype(np.float32), verbose=0)
    return predict_fn, (h, w, c), "0_1"

def load_tflite(tmp_path):
    interpreter = tf.lite.Interpreter(model_path=tmp_path)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    ishape = in_det["shape"]  # e.g. [1,224,224,1] or [1,224,224,3]
    h = int(ishape[1]); w = int(ishape[2]); c = int(ishape[3])

    # TFLite thường dùng float32 input. Nếu model của bạn tiền xử lý [-1,1], bạn đổi norm bên dưới.
    def predict_fn(x):
        # Nhét đúng dtype theo TFLite
        x_in = x.astype(in_det["dtype"])
        interpreter.set_tensor(in_det["index"], x_in)
        interpreter.invoke()
        y = interpreter.get_tensor(out_det["index"])
        return y.astype(np.float32)

    # Mặc định norm "0_1". Nếu model cần -1..1, đổi "norm_hint" thành "-1_1".
    norm_hint = "0_1"
    return predict_fn, (h, w, c), norm_hint

def load_onnx(tmp_path):
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError("Thiếu onnxruntime. Cài: pip install onnxruntime") from e

    sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    ishape = sess.get_inputs()[0].shape  # [1,C,H,W] hoặc [1,H,W,C] tuỳ model
    # Chuẩn: nhiều ONNX dùng NCHW. Ta cố gắng suy luận:
    # nếu ishape có 4 chiều và ishape[1] in {1,3} => coi là NCHW
    is_nchw = (len(ishape) == 4 and isinstance(ishape[1], int) and ishape[1] in (1,3))
    if is_nchw:
        c = int(ishape[1]); h = int(ishape[2]); w = int(ishape[3])
    else:
        # fallback coi như NHWC
        h = int(ishape[1]); w = int(ishape[2]); c = int(ishape[3])

    def predict_fn(x):
        # ONNX thường muốn float32; nếu NCHW thì transpose
        x_in = x.astype(np.float32)
        if is_nchw:
            x_in = np.transpose(x_in, (0, 3, 1, 2))  # NHWC->NCHW
        preds = sess.run(None, {in_name: x_in})[0]
        # đảm bảo trả probs shape (1,num_classes)
        if preds.ndim > 2:
            preds = preds.reshape((preds.shape[0], -1))
        return preds.astype(np.float32)

    norm_hint = "0_1"
    return predict_fn, (h, w, c), norm_hint


# ====== Tải model tự động trong thư mục dự án ======
st.subheader("1) Tải model")

default_model_path = "flatfoot_model_VietNam_light.keras"
predict_fn = None
input_shape = (224, 224, 3)  # vì bạn huấn luyện với MobileNetV2 (RGB)
norm_hint = "0_1"

# Nếu file model có sẵn trong repo
if os.path.exists(default_model_path):
    with st.spinner("Đang tải model từ thư mục..."):
        try:
            model = load_model(default_model_path)
            ishape = model.inputs[0].shape
            h, w, c = int(ishape[1]), int(ishape[2]), int(ishape[3])

            def predict_fn(x):
                return model.predict(x.astype(np.float32), verbose=0)

            input_shape = (h, w, c)
            st.success(f"✅ Model đã tải: {default_model_path} (Input shape: {input_shape})")

        except Exception as e:
            st.error(f"Không load được model mặc định: {e}")

else:
    st.error(f"❌ Không tìm thấy model {default_model_path}. Hãy chắc rằng file nằm cùng thư mục app.py")


# ====== Upload ảnh và dự đoán ======
st.subheader("2) Tải ảnh để dự đoán")
img_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

# Cho phép chỉnh norm nếu biết mô hình cần -1..1
norm_opt = st.selectbox("Chuẩn hoá đầu vào", ["0_1 (mặc định)", "-1_1"], index=0)
norm_to_use = "0_1" if "0_1" in norm_opt else "-1_1"

predict_btn = st.button("🚀 Dự đoán")

if predict_btn:
    if predict_fn is None:
        st.error("Vui lòng tải model trước.")
    elif img_file is None:
        st.error("Vui lòng chọn một ảnh.")
    else:
        H, W, C = input_shape
        with st.spinner("Đang tiền xử lý & dự đoán..."):
            rgb, x = preprocess_image_for_shape(img_file.read(), target_hw=(H, W), channels=C, norm=norm_to_use)
            if rgb is None:
                st.error("Không đọc được ảnh. Vui lòng thử ảnh khác.")
            else:
                try:
                    probs = predict_fn(x)
                    cls = int(np.argmax(probs))
                    conf = float(np.max(probs))
                    desc = LABEL_MAP.get(cls, "Khong ro")

                    st.write(f"**Kết quả:** Nhãn `{cls}` – **{desc}** với độ tin cậy **{conf:.2%}**")

                    # Vẽ text lên ảnh
                    text = f"Nhan {cls}: {desc} ({conf:.2%})"
                    h_img, w_img = rgb.shape[:2]
                    scale = max(0.6, min(1.2, w_img / 800))
                    cv2.putText(rgb, text, (20, int(40*scale)),
                                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2, cv2.LINE_AA)

                    st.image(rgb, caption="Ảnh có gắn nhãn dự đoán", use_container_width=True)
                except Exception as e:
                    st.error(f"Lỗi khi dự đoán: {e}")

st.divider()
st.caption("Mẹo: Nếu TFLite/ONNX cần chuẩn hoá đầu vào [-1,1], hãy đổi mục 'Chuẩn hoá đầu vào' ở trên.")
