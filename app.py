# app.py — Streamlit UI: upload CSV + ảnh (ZIP) rồi train (self-contained, không import VietNam.py)
import os, io, zipfile, shutil
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

st.set_page_config(page_title="Chẩn đoán bàn chân", layout="wide")
st.title("🦶 Chẩn đoán bàn chân (VN) – Streamlit")

# ---------- Helpers (tự chứa) ----------
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.equalizeHist(img)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.resize(img, (224, 224)).astype("float32") / 255.0
    return img

def load_dataset(csv_path, image_folder, name_col='tên', label_col='nhãn', num_classes=5):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không thấy CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin1")
    df.columns = df.columns.str.strip().str.lower()
    if name_col not in df.columns:
        raise KeyError(f"Thiếu cột '{name_col}'")
    if label_col not in df.columns:
        raise KeyError(f"Thiếu cột '{label_col}'")

    X, y, miss, bad = [], [], 0, 0
    for _, row in df.iterrows():
        fn = str(row[name_col]).strip()
        lb = int(row[label_col])
        p = os.path.join(image_folder, fn)
        if not os.path.exists(p):
            miss += 1
            continue
        im = preprocess_image(p)
        if im is None:
            bad += 1
            continue
        X.append(im)
        y.append(lb)
    if miss or bad:
        st.info(f"Ảnh thiếu: {miss} | Ảnh đọc lỗi: {bad}")
    if not X:
        raise RuntimeError("Không nạp được mẫu nào.")
    X = np.array(X, dtype=np.float32).reshape(-1, 224, 224, 1)
    y = to_categorical(np.array(y, dtype=np.int32), num_classes=num_classes)
    return X, y

def build_model(num_classes=5):
    m = Sequential([
        Conv2D(32,(3,3),activation='relu',input_shape=(224,224,1)),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128,activation='relu'),
        Dense(num_classes,activation='softmax')
    ])
    m.compile(optimizer=Adam(5e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def plot_training_history(history):
    fig, ax = plt.subplots(1,2,figsize=(10,4))
    ax[0].plot(history.history.get('accuracy', []), label='Train')
    ax[0].plot(history.history.get('val_accuracy', []), label='Val')
    ax[0].set_title('Accuracy'); ax[0].legend()

    ax[1].plot(history.history.get('loss', []), label='Train')
    ax[1].plot(history.history.get('val_loss', []), label='Val')
    ax[1].set_title('Loss'); ax[1].legend()
    st.pyplot(fig)

def plot_right_wrong_counts(y_true, y_pred, num_classes):
    corr = np.zeros(num_classes); inc = np.zeros(num_classes)
    for t,p in zip(y_true,y_pred):
        if t==p: corr[t]+=1
        else: inc[t]+=1
    x = np.arange(num_classes); labels=[f'Lớp {i}' for i in range(num_classes)]
    fig = plt.figure(figsize=(10,4))
    plt.bar(x-0.35/2, corr, 0.35, label='Đúng')
    plt.bar(x+0.35/2, inc, 0.35, label='Sai')
    plt.xticks(x, labels); plt.legend(); plt.title('Đúng/Sai theo nhãn'); plt.tight_layout()
    st.pyplot(fig)

def plot_confusion(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest'); plt.title('Confusion Matrix'); plt.colorbar()
    ticks = np.arange(num_classes); names=[f'Nhãn {i}' for i in range(num_classes)]
    plt.xticks(ticks, names, rotation=45); plt.yticks(ticks, names)
    thresh = cm.max()/2 if cm.max()>0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i,j],'d'),
                     ha="center", color="white" if cm[i,j]>thresh else "black")
    plt.ylabel('Thực tế'); plt.xlabel('Dự đoán'); plt.tight_layout()
    st.pyplot(fig)

# ---------- UI ----------
st.markdown("**Bước 1:** Upload CSV và ảnh (nén thành ZIP).  \n"
            "**Bước 2:** Nhấn Train. (Nếu dataset lớn, hãy giảm epochs/batch)")

csv_file = st.file_uploader("📄 CSV (có cột 'tên' & 'nhãn')", type=["csv"])
zip_file = st.file_uploader("🗂️ Ảnh (ZIP)", type=["zip"])

name_col = st.text_input("Tên cột ảnh", value="tên")
label_col = st.text_input("Tên cột nhãn", value="nhãn")
num_classes = st.number_input("Số lớp", 2, 20, 5, 1)

work_dir = "/tmp/dataset"
csv_path = os.path.join(work_dir, "data", "VietNam.csv")
img_root = os.path.join(work_dir, "images")
img_dir  = os.path.join(img_root, "VietNam")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
os.makedirs(img_root, exist_ok=True)

if csv_file is not None:
    with open(csv_path, "wb") as f:
        f.write(csv_file.read())
    st.success(f"Đã lưu CSV → {csv_path}")

if zip_file is not None:
    if os.path.exists(img_root):
        shutil.rmtree(img_root)
    os.makedirs(img_root, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_file.read())) as z:
        z.extractall(img_root)
    # Nếu không có folder VietNam, đoán thư mục ảnh
    if not os.path.exists(img_dir):
        for root, _, files in os.walk(img_root):
            if any(f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff")) for f in files):
                img_dir = root
                break
    st.success(f"Ảnh đã giải nén. Thư mục dùng: {img_dir}")

c1, c2, c3 = st.columns(3)
with c1: epochs = st.number_input("Epochs", 1, 200, 15)
with c2: batch  = st.number_input("Batch size", 1, 256, 32)
with c3: val_split = st.slider("Validation split", 0.05, 0.4, 0.2, 0.05)

if st.button("🚀 Train"):
    if not os.path.exists(csv_path):
        st.error("Chưa upload CSV."); st.stop()
    if not os.path.exists(img_dir):
        st.error("Chưa upload ZIP ảnh."); st.stop()

    with st.spinner("Đang nạp dữ liệu..."):
        X, y = load_dataset(csv_path, img_dir, name_col=name_col, label_col=label_col, num_classes=int(num_classes))
    y_int = np.argmax(y, axis=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_int)

    model = build_model(num_classes=int(num_classes))
    cb = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]

    with st.spinner("Đang huấn luyện..."):
        history = model.fit(Xtr, ytr, validation_split=float(val_split),
                            epochs=int(epochs), batch_size=int(batch), verbose=0, callbacks=cb)

    loss, acc = model.evaluate(Xte, yte, verbose=0)
    st.success(f"✅ Test accuracy: **{acc:.2%}**")

    y_pred = model.predict(Xte, verbose=0)
    y_pred_cls = np.argmax(y_pred, axis=1)
    y_true_cls = np.argmax(yte, axis=1)

    st.write("**Precision:**", precision_score(y_true_cls, y_pred_cls, average=None, zero_division=0))
    st.write("**Recall:**",    recall_score(y_true_cls, y_pred_cls, average=None, zero_division=0))
    st.write("**F1-score:**",  f1_score(y_true_cls, y_pred_cls, average=None, zero_division=0))

    plot_training_history(history)
    plot_right_wrong_counts(y_true_cls, y_pred_cls, int(num_classes))
    plot_confusion(y_true_cls, y_pred_cls, int(num_classes))

    os.makedirs("./models", exist_ok=True)
    model_path = "./models/flatfoot_streamlit.keras"
    model.save(model_path)
    st.success(f"💾 Model saved to `{model_path}`")
