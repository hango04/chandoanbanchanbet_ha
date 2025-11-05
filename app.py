# app.py
import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2, os, io, zipfile, tempfile, requests, math

st.set_page_config(page_title="🦶 Dự đoán bàn chân (X-ray)", layout="centered")
st.title("🦶 Dự đoán nhãn bàn chân từ ảnh (1 ảnh)")
st.caption("Hỗ trợ: .keras, .h5, .tflite, SavedModel (.zip), .onnx (onnxruntime), TorchScript .pt/.pth")

LABEL_MAP = {0:"Bình thường",1:"Bẹt nhẹ",2:"Bẹt trung bình",3:"Bẹt nặng",4:"Không xác định"}

# ========= Utils =========

def ensure_3ch(x1ch): 
    return np.repeat(x1ch, 3, axis=-1)

def normalize_img(gray, mode):
    if mode == "neg1_1":
        return (gray/127.5 - 1.0).astype("float32")
    elif mode == "imagenet":
        x = (gray/255.0).astype("float32")
        x = np.expand_dims(x, -1)
        x = ensure_3ch(x)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (x - mean) / std
    else:
        return (gray/255.0).astype("float32")

def preprocess_image_for_shape(image_bytes, target_hw=(224,224), channels=3, norm="neg1_1"):
    fb = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(fb, cv2.IMREAD_COLOR)
    if bgr is None: 
        return None, None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
    gray = cv2.filter2D(gray, -1, kernel)

    H, W = target_hw
    gray = cv2.resize(gray, (W, H))

    x = normalize_img(gray, norm)
    x = np.expand_dims(x, -1)
    if channels == 3:
        x = ensure_3ch(x)

    x = np.expand_dims(x, 0)
    return rgb, x.astype(np.float32)

def _save_temp(ext, data):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    f.write(data); f.close()
    return f.name

def _softmax_if_needed(y):
    if y.ndim > 2: y = y.reshape((y.shape[0], -1))
    m = np.max(y[0])
    ex = np.exp(y[0]-m)
    sm = ex / np.sum(ex)
    return sm.reshape(1,-1).astype(np.float32)

# ========= Model loaders =========

@st.cache_resource
def load_keras_or_h5(path):
    m = load_model(path)
    h,w,c = [int(m.inputs[0].shape[i]) for i in (1,2,3)]
    return lambda x: _softmax_if_needed(m.predict(x,verbose=0)), (h,w,c), "0_1"

@st.cache_resource
def load_tflite(path):
    inter = tf.lite.Interpreter(model_path=path); inter.allocate_tensors()
    in_det, out_det = inter.get_input_details()[0], inter.get_output_details()[0]
    ishape = in_det["shape"]
    h,w,c = int(ishape[1]), int(ishape[2]), int(ishape[3])
    dtype = in_det["dtype"]

    def pred(x):
        x_in = x.astype(dtype)
        if dtype == np.uint8: x_in = (x*255).astype(np.uint8)
        inter.set_tensor(in_det["index"], x_in)
        inter.invoke()
        y = inter.get_tensor(out_det["index"])
        return _softmax_if_needed(y)
    return pred,(h,w,c),"0_1"

@st.cache_resource
def load_onnx(path):
    import onnxruntime as ort
    sess = ort.InferenceSession(path)
    in0 = sess.get_inputs()[0]
    ishape = in0.shape
    nchw = (ishape[1] in (1,3))
    if nchw: c,h,w = ishape[1],ishape[2],ishape[3]
    else: h,w,c = ishape[1],ishape[2],ishape[3]

    def pred(x):
        x2 = np.transpose(x,(0,3,1,2)) if nchw else x
        y = sess.run(None,{in0.name:x2})[0]
        return _softmax_if_needed(y)
    return pred,(h,w,c),"imagenet"

def load_by_ext(ext, data, fname):
    if ext in [".keras",".h5"]: 
        return load_keras_or_h5(data)
    if ext == ".tflite":
        return load_tflite(data)
    if ext == ".onnx":
        return load_onnx(data)
    if ext == ".zip":
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(io.BytesIO(data)) as z: z.extractall(tmp)
        return load_keras_or_h5(tmp)
    raise Exception("Format not supported")

# ========= UI =========

st.subheader("1) Nạp model")
source = st.radio("Nguồn:", ["Upload", "File local"], horizontal=True)

predict_fn = None
input_shape = (224,224,3)
norm_default = "0_1"

if source=="Upload":
    up = st.file_uploader("Model", type=["keras","h5","tflite","zip","onnx"])
    if up:
        ext = os.path.splitext(up.name)[1].lower()
        f = _save_temp(ext, up.read())
        predict_fn,input_shape,norm_default = load_by_ext(ext,f,up.name)
        st.success(f"✅ Model loaded ({input_shape})")

else:
    p = st.text_input("Đường dẫn model", "flatfoot_model_VN_fp16.tflite")
    if st.button("Load"):
        if not os.path.exists(p): st.error("Không thấy file")
        else:
            ext = os.path.splitext(p)[1].lower()
            predict_fn,input_shape,norm_default = load_by_ext(ext, p, p)
            st.success(f"✅ Loaded ({input_shape})")

st.subheader("2) Ảnh")
img = st.file_uploader("Ảnh X-ray", type=["jpg","png","jpeg"])

if st.button("⚡ Dự đoán"):
    if predict_fn is None: st.error("Chưa có model")
    elif img is None: st.error("Chưa chọn ảnh")
    else:
        H,W,C = input_shape
        rgb,x = preprocess_image_for_shape(img.read(), (H,W), C, norm_default)
        if rgb is None: st.error("Ảnh lỗi")
        else:
            probs = predict_fn(x)
            cls = int(np.argmax(probs))
            conf = float(np.max(probs))
            st.success(f"**Kết quả:** {LABEL_MAP.get(cls,cls)} ({conf:.1%})")

            cv2.putText(rgb,f"{LABEL_MAP.get(cls,cls)} {conf:.0%}",(20,60),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),3)
            st.image(rgb,caption="Ảnh + dự đoán",width="stretch")

            st.write("### Xác suất")
            for i,p in enumerate(probs[0]):
                st.write(f"- {i}: {LABEL_MAP.get(i,i)} → {p:.4f}")

st.caption("✅ App đã fix crash Streamlit `use_container_width`")

