import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from io import BytesIO

# ====== Cấu hình trang ======
st.set_page_config(page_title="Dự đoán bàn chân", layout="centered")

st.title("🦶 Dự đoán nhãn bàn chân từ ảnh (1 ảnh)")
st.caption("Upload model (.keras) và một ảnh; app sẽ dự đoán và in nhãn lên ảnh.")

# ====== Ánh xạ nhãn ======
LABEL_MAP = {
    0: "Binh thuong",
    1: "Bet nhe",
    2: "Bet trung bình",
    3: "Bet nang",
    4: "Khong xac dinh"
}

# ====== Tiền xử lý ======
def preprocess_image_from_bytes(image_bytes):
    # Đọc bytes -> mảng numpy BGR
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
    gray = cv2.resize(gray, (224, 224)).astype("float32") / 255.0
    x = np.expand_dims(gray, axis=-1)   # (224,224,1)
    x = np.expand_dims(x, axis=0)       # (1,224,224,1)
    return bgr, x  # trả lại ảnh gốc BGR để overlay text

# ====== Tải model ======
st.subheader("1) Tải model (.keras)")
model_file = st.file_uploader("Chọn file model (.keras)", type=["keras", "h5"])

model = None
if model_file is not None:
    # Lưu vào buffer và load trực tiếp
    with st.spinner("Đang tải model..."):
        # Một số phiên bản load_model cần đường dẫn file.
        # Streamlit cho phép dùng NamedTemporaryFile nếu cần. Ở đây load từ bytes:
        try:
            # Cách 1: ghi tạm ra file rồi load (ổn định hơn trong nhiều môi trường)
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
                tmp.write(model_file.read())
                tmp_path = tmp.name
            model = load_model(tmp_path)
            os.unlink(tmp_path)  # xóa file tạm
            st.success("✅ Model đã được tải.")
        except Exception as e:
            st.error(f"Không load được model: {e}")

# ====== Upload ảnh và dự đoán ======
st.subheader("2) Tải ảnh để dự đoán")
img_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

predict_btn = st.button("🚀 Dự đoán")

if predict_btn:
    if model is None:
        st.error("Vui lòng tải model trước.")
    elif img_file is None:
        st.error("Vui lòng chọn một ảnh.")
    else:
        with st.spinner("Đang tiền xử lý & dự đoán..."):
            # Tiền xử lý
            bgr, x = preprocess_image_from_bytes(img_file.read())
            if bgr is None:
                st.error("Không đọc được ảnh. Vui lòng thử ảnh khác.")
            else:
                # Dự đoán
                pred = model.predict(x, verbose=0)
                cls = int(np.argmax(pred))
                conf = float(np.max(pred))
                desc = LABEL_MAP.get(cls, "Khong ro")

                st.write(f"**Kết quả:** Nhãn `{cls}` – **{desc}** với độ tin cậy **{conf:.2%}**")

                # Vẽ text lên ảnh gốc (BGR)
                text = f"Nhan {cls}: {desc} ({conf:.2%})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                # Scale font theo chiều rộng ảnh để chữ dễ nhìn
                h, w = bgr.shape[:2]
                scale = max(0.6, min(1.2, w / 800))
                cv2.putText(bgr, text, (20, int(40*scale)),
                            font, scale, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

                # BGR -> RGB để hiển thị đúng màu trên Streamlit
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                st.image(rgb, caption="Ảnh có gắn nhãn dự đoán", use_container_width=True)

st.divider()
st.caption("Gợi ý: Để chạy nhanh trên Cloud, hãy dùng model nhẹ hoặc giảm kích thước model khi train.")
