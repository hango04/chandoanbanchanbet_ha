import os
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# -----------------------------
# 1) Tiền xử lý & tiện ích
# -----------------------------
def preprocess_image(path):
    """Đọc ảnh xám, cân bằng histogram, sharpen nhẹ, resize 224x224, chuẩn hóa [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Không thể đọc ảnh: {path}")
        return None
    img = cv2.equalizeHist(img)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return img


def show_before_after(path):
    """Hiển thị ảnh gốc và sau xử lý (chỉ dùng để kiểm tra nhanh local)."""
    original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    processed = preprocess_image(path)
    if original is None or processed is None:
        print("⚠️ Không thể hiển thị ảnh.")
        return
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1); plt.imshow(original, cmap='gray'); plt.title('Ảnh gốc'); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(processed, cmap='gray'); plt.title('Ảnh sau xử lý'); plt.axis('off')
    plt.tight_layout(); plt.show()


def safe_read_csv(csv_path):
    """Đọc CSV an toàn, tự thử utf-8 rồi latin1; chuẩn hóa tên cột (lower, strip)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file CSV: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin1")
        print("⚠️ Đã chuyển sang encoding 'latin1' do lỗi UTF-8.")
    df.columns = df.columns.str.strip().str.lower()
    return df


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# -----------------------------
# 2) Load dữ liệu từ CSV + ảnh
# -----------------------------
def load_dataset(csv_path, image_folder, name_col='tên', label_col='nhãn', num_classes=5):
    df = safe_read_csv(csv_path)
    print("📋 Các cột:", df.columns.tolist())

    # Kiểm tra cột bắt buộc
    if name_col not in df.columns:
        raise KeyError(f"❌ Không có cột '{name_col}' trong CSV.")
    if label_col not in df.columns:
        raise KeyError(f"❌ Không có cột '{label_col}' trong CSV.")

    X, y = [], []
    missing, unreadable = 0, 0

    for _, row in df.iterrows():
        filename = str(row[name_col]).strip()
        label = row[label_col]
        img_path = os.path.join(image_folder, filename)
        if not os.path.exists(img_path):
            missing += 1
            continue
        img = preprocess_image(img_path)
        if img is None:
            unreadable += 1
            continue
        X.append(img)
        y.append(int(label))

    if missing or unreadable:
        print(f"ℹ️ Ảnh thiếu: {missing} | Ảnh đọc lỗi: {unreadable}")

    if len(X) == 0:
        raise RuntimeError("❌ Không nạp được mẫu nào. Kiểm tra đường dẫn ảnh & CSV.")

    X = np.array(X, dtype=np.float32).reshape(-1, 224, 224, 1)
    y = to_categorical(np.array(y, dtype=np.int32), num_classes=num_classes)
    return X, y


# -----------------------------
# 3) Model
# -----------------------------
def build_model(num_classes=5):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# -----------------------------
# 4) Vẽ biểu đồ
# -----------------------------
def plot_training_history(history):
    plt.figure(figsize=(10, 4))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout(); plt.show()


def plot_right_wrong_counts(y_true_classes, y_pred_classes, num_classes):
    correct_counts = np.zeros(num_classes)
    incorrect_counts = np.zeros(num_classes)
    for t, p in zip(y_true_classes, y_pred_classes):
        if t == p: correct_counts[t] += 1
        else:      incorrect_counts[t] += 1

    labels = [f'Lớp {i}' for i in range(num_classes)]
    x = np.arange(num_classes); width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, correct_counts, width, label='Dự đoán đúng')
    plt.bar(x + width/2, incorrect_counts, width, label='Dự đoán sai')
    plt.xticks(x, labels); plt.ylabel('Số lượng mẫu')
    plt.title('Số lượng dự đoán đúng/sai theo nhãn'); plt.legend()
    plt.tight_layout(); plt.show()


def plot_confusion(y_true_classes, y_pred_classes, num_classes):
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Ma trận nhầm lẫn'); plt.colorbar()
    tick_marks = np.arange(num_classes)
    class_names = [f'Nhãn {i}' for i in range(num_classes)]
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Ghi số
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Thực tế'); plt.xlabel('Dự đoán'); plt.tight_layout(); plt.show()


# -----------------------------
# 5) Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train CNN phân loại bàn chân (VN).")
    parser.add_argument("--csv", default=os.getenv("CSV_PATH", "./data/VietNam.csv"),
                        help="Đường dẫn CSV (mặc định ./data/VietNam.csv)")
    parser.add_argument("--img_dir", default=os.getenv("IMG_DIR", "./images/VietNam"),
                        help="Thư mục ảnh (mặc định ./images/VietNam)")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", 60)))
    parser.add_argument("--batch", type=int, default=int(os.getenv("BATCH", 68)))
    parser.add_argument("--num_classes", type=int, default=int(os.getenv("NUM_CLASSES", 5)))
    parser.add_argument("--name_col", default=os.getenv("NAME_COL", "tên"))
    parser.add_argument("--label_col", default=os.getenv("LABEL_COL", "nhãn"))
    parser.add_argument("--demo_image", default=os.getenv("DEMO_IMAGE", ""))  # tuỳ chọn: xem trước/sau
    args = parser.parse_args()

    # Tạo thư mục models/
    ensure_dirs("./models")

    # Demo xem trước/sau nếu cung cấp demo_image
    if args.demo_image and os.path.exists(args.demo_image):
        show_before_after(args.demo_image)

    # Nạp dữ liệu
    print(f"📂 CSV: {args.csv}")
    print(f"🖼️  IMG DIR: {args.img_dir}")
    X, y = load_dataset(args.csv, args.img_dir,
                        name_col=args.name_col, label_col=args.label_col,
                        num_classes=args.num_classes)

    # Chia dữ liệu
    y_int = np.argmax(y, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_int
    )

    # Model
    model = build_model(num_classes=args.num_classes)

    # Callbacks
    ckpt_path = "./models/best_model.keras"
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Độ chính xác trên tập kiểm tra: {acc:.2%}")

    # Save final
    final_model_path = "./models/flatfoot_model_VietNam_final.keras"
    model.save(final_model_path)
    print(f"💾 Đã lưu mô hình: {final_model_path}")

    # Dự đoán & thống kê
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Metrics từng lớp
    precision = precision_score(y_true_classes, y_pred_classes, average=None, zero_division=0)
    recall = recall_score(y_true_classes, y_pred_classes, average=None, zero_division=0)
    f1 = f1_score(y_true_classes, y_pred_classes, average=None, zero_division=0)

    print("🔎 Precision:", precision)
    print("🔎 Recall   :", recall)
    print("🔎 F1-score :", f1)

    # Vẽ đồ thị
    plot_training_history(history)
    plot_right_wrong_counts(y_true_classes, y_pred_classes, args.num_classes)
    plot_confusion(y_true_classes, y_pred_classes, args.num_classes)


if __name__ == "__main__":
    # Giảm log TensorFlow khi deploy
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.get_logger().setLevel("ERROR")
    main()
