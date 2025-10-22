# app.py
import os, io, zipfile, shutil
import streamlit as st
import numpy as np
import pandas as pd

from VietNam import (  # dùng lại các hàm đã refactor trong VietNam.py
    preprocess_image, show_before_after, safe_read_csv, load_dataset,
    build_model, plot_training_history, plot_right_wrong_counts, plot_confusion
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf


st.set_page_config(page_title="Chẩn đoán bàn chân", layout="wide")
st.title("🦶 Chẩn đoán bàn chân (VN) – Streamlit")

st.markdown("Tải **CSV** và **ảnh (ZIP)** rồi bấm **Train**. App sẽ không yêu cầu file có sẵn trong repo.")

# Khu vực upload
csv_file = st.file_uploader("📄 Tải lên CSV (VietNam.csv)", type=["csv"])
zip_file = st.file_uploader("🗂️ Tải lên ảnh (ZIP thư mục images/VietNam/...)", type=["zip"])

name_col = st.text_input("Tên cột chứa tên ảnh", value="tên")
label_col = st.text_input("Tên cột chứa nhãn", value="nhãn")
num_classes = st.number_input("Số lớp (num_classes)", min_value=2, max_value=20, value=5, step=1)

# Nơi làm việc tạm thời
work_dir = "/tmp/dataset"
img_dir = os.path.join(work_dir, "images", "VietNam")
csv_path = os.path.join(work_dir, "data", "VietNam.csv")

# Chuẩn bị thư mục
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

# Khi người dùng upload CSV
if csv_file is not None:
    with open(csv_path, "wb") as f:
        f.write(csv_file.read())
    st.success(f"Đã lưu CSV vào {csv_path}")

# Khi người dùng upload ZIP ảnh
if zip_file is not None:
    # Xóa ảnh cũ & giải nén mới
    if os.path.exists(os.path.join(work_dir, "images")):
        shutil.rmtree(os.path.join(work_dir, "images"))
    os.makedirs(os.path.join(work_dir, "images"), exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(zip_file.read())) as z:
        z.extractall(os.path.join(work_dir, "images"))
    st.success("Đã giải nén ảnh vào /tmp/dataset/images")

    # Tìm thư mục con VietNam nếu người dùng nén cả folder khác tên
    # Nếu không có VietNam, cố gắng đoán:
    if not os.path.exists(img_dir):
        # lấy folder ảnh sâu nhất chứa nhiều file .jpg/.png
        candidate = None
        for root, dirs, files in os.walk(os.path.join(work_dir, "images")):
            if any(f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) for f in files):
                candidate = root
                break
        if candidate:
            img_dir = candidate
            st.info(f"Ảnh tìm thấy ở: {img_dir}")

st.divider()

# Tham số huấn luyện
col1, col2, col3 = st.columns(3)
with col1:
    epochs = st.number_input("Epochs", 1, 200, 15, 1)  # giữ thấp cho Cloud
with col2:
    batch = st.number_input("Batch size", 1, 256, 32, 1)
with col3:
    val_split = st.slider("Validation split", 0.05, 0.4, 0.2, 0.05)

train_btn = st.button("🚀 Train")

if train_btn:
    # Kiểm tra đủ file
    if not os.path.exists(csv_path):
        st.error("Chưa có CSV. Hãy upload trước.")
        st.stop()
    if not os.path.exists(img_dir):
        st.error("Chưa có thư mục ảnh. Hãy upload ZIP ảnh trước.")
        st.stop()

    with st.spinner("Đang nạp dữ liệu..."):
        try:
            # dùng load_dataset từ VietNam.py
            X, y = load_dataset(csv_path, img_dir, name_col=name_col, label_col=label_col, num_classes=num_classes)
        except Exception as e:
            st.exception(e)
            st.stop()

    y_int = np.argmax(y, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_int
    )

    model = build_model(num_classes=num_classes)

    cb = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]

    with st.spinner("Đang huấn luyện..."):
        history = model.fit(
            X_train, y_train,
            validation_split=val_split,
            epochs=int(epochs),
            batch_size=int(batch),
            callbacks=cb,
            verbose=0
        )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    st.success(f"✅ Accuracy (test): **{acc:.2%}**")

    # Dự đoán & metrics
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    precision = precision_score(y_true_classes, y_pred_classes, average=None, zero_division=0)
    recall    = recall_score(y_true_classes, y_pred_classes, average=None, zero_division=0)
    f1        = f1_score(y_true_classes, y_pred_classes, average=None, zero_division=0)

    st.write("**Precision:**", precision)
    st.write("**Recall:**", recall)
    st.write("**F1-score:**", f1)

    # Vẽ biểu đồ
    st.subheader("Biểu đồ huấn luyện")
    import matplotlib.pyplot as plt
    fig1 = plt.figure(figsize=(8,3)); 
    # tái sử dụng hàm vẽ trong VietNam.py
    from VietNam import plot_training_history as _pth
    _pth(history)  # sẽ hiện ngay nhờ Streamlit hook
    st.pyplot(fig1)

    st.subheader("Đúng/Sai theo nhãn")
    fig2 = plt.figure(figsize=(8,3))
    from VietNam import plot_right_wrong_counts as _prwc
    _prwc(y_true_classes, y_pred_classes, num_classes)
    st.pyplot(fig2)

    st.subheader("Confusion Matrix")
    fig3 = plt.figure(figsize=(5,4))
    from VietNam import plot_confusion as _pc
    _pc(y_true_classes, y_pred_classes, num_classes)
    st.pyplot(fig3)

    # Lưu model
    os.makedirs("/app/models", exist_ok=True) if os.path.exists("/app") else os.makedirs("./models", exist_ok=True)
    model_path = "./models/flatfoot_streamlit.keras"
    model.save(model_path)
    st.success(f"💾 Model saved to `{model_path}`")
