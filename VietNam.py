import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 1. Hàm tiền xử lý ảnh
def preprocess_image(path):
  
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Không thể đọc ảnh: {path}")
        return None
    img = cv2.equalizeHist(img)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img
# Hàm hiển thị ảnh gốc và ảnh sau xử lý
def show_before_after(path):
    original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    processed = preprocess_image(path)

    if original is None or processed is None:
        print("⚠️ Không thể hiển thị ảnh.")
        return

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Ảnh gốc')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title('Ảnh sau xử lý')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

#  Đường dẫn ảnh bạn muốn kiểm tra
image_path = 'd:/banchan/VietNam/021.jpg'  ######################################################## ← thay bằng tên ảnh thật

# Gọi hàm hiển thị
show_before_after(image_path)

# 2. Đọc dữ liệu từ file CSV với xử lý mã hóa và tên cột
csv_path = 'd:/banchan/VietNam.csv'                    ####                                           Bộ  #############
image_folder = 'd:/banchan/VietNam'                    ####              Bộ                               ############

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ Không tìm thấy file: {csv_path}")

try:
    df = pd.read_csv(csv_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(csv_path, encoding='latin1')
    print("⚠️ Đã chuyển sang encoding 'latin1' do lỗi mã hóa UTF-8")

# Chuẩn hóa tên cột
df.columns = df.columns.str.strip().str.lower()
print("📋 Các cột trong file:", df.columns.tolist())

# 3. Xử lý dữ liệu ảnh và nhãn
X = []
y = []

for _, row in df.iterrows():
    filename = row['tên']         # dùng đúng tên cột viết thường
    label = row['nhãn']           # nếu cần, đổi thành 'nhãn số' nếu cột tên vậy
    path = os.path.join(image_folder, filename)
    img = preprocess_image(path)
    if img is not None:
        X.append(img)
        y.append(label)

X = np.array(X).reshape(-1, 224, 224, 1)
y = to_categorical(y, num_classes=5)

# 4. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])

# 6. Compile và huấn luyện
model.compile(optimizer=Adam(learning_rate=0.00005),  # learning_rate=0.00005 là tốc độ học.
                 # Giá trị nhỏ giúp học chậm nhưng chính xác hơn, tránh dao động mạnh.
              loss='categorical_crossentropy',        # Hàm mất mát dùng để đo sai số giữa nhãn thật (y_true) và dự đoán (y_pred).
                 # 'categorical_crossentropy' dùng cho bài toán phân loại nhiều lớp (multi-class).
                                                     
              metrics=['accuracy'])                   #Độ chính xác của mô hình (accuracy).


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
   # Theo dõi giá trị "loss" trên tập validation (dữ liệu kiểm tra trong quá trình học).
   # Nếu sau 5 epoch liên tiếp mà val_loss không giảm → dừng huấn luyện sớm.
   # Sau khi dừng, mô hình tự động khôi phục trọng số tốt nhất (val_loss thấp nhất).
   # Giúp tránh overfitting

#  model.fit Đây là hàm để huấn luyện mô hình
history = model.fit(X_train, y_train,       #X_train: Dữ liệu đầu vào (features). y_train: Nhãn
                    validation_split=0.2,   # Tách 20% dữ liệu để kiểm tra.
                    #Lệnh này yêu cầu model.fit tự động dành ra 20% của X_train và y_train để làm dữ liệu kiểm thử (validation set).
                    epochs=60, #Số vòng lặp huấn luyện.
                    batch_size=68,
                    callbacks=[early_stop]) # Gắn callback "early_stop" để mô hình tự động dừng khi không cải thiện.

# 7. Đánh giá mô hình
loss, acc = model.evaluate(X_test, y_test)
print(f' Độ chính xác trên tập kiểm tra: {acc:.2%}')

# 8. Lưu mô hình
#model.save('flatfoot_model_VietNam.keras')
model.save('d:/banchan/flatfoot_model_VietNam.keras')
print(" Mô hình đã được lưu vào 'flatfoot_model_VietNam.keras'")

import matplotlib.pyplot as plt
import numpy as np

# Dự đoán
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Tính số lượng đúng và sai theo từng nhãn
num_classes = y_test.shape[1]
correct_counts = np.zeros(num_classes)
incorrect_counts = np.zeros(num_classes)

for true, pred in zip(y_true_classes, y_pred_classes):
    if true == pred:
        correct_counts[true] += 1
    else:
        incorrect_counts[true] += 1

# Vẽ biểu đồ
labels = [f'Lớp {i}' for i in range(num_classes)]
x = np.arange(num_classes)
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, correct_counts, width, label='Dự đoán đúng', color='mediumseagreen')
plt.bar(x + width/2, incorrect_counts, width, label='Dự đoán sai', color='tomato')

plt.xticks(x, labels)
plt.ylabel('Số lượng mẫu')
plt.title('Số lượng dự đoán đúng và sai theo từng nhãn')
plt.legend()
plt.tight_layout()
plt.show()

##########################################################################################################################
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# 1. Vẽ biểu đồ Accuracy và Loss
def plot_training_history(history):
    plt.figure(figsize=(10, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='green')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Gọi hàm vẽ biểu đồ huấn luyện
plot_training_history(history)

# 2. Vẽ biểu đồ Precision, Recall, F1-score cho từng lớp
# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Tính các chỉ số
precision = precision_score(y_true_classes, y_pred_classes, average=None)
recall = recall_score(y_true_classes, y_pred_classes, average=None)
f1 = f1_score(y_true_classes, y_pred_classes, average=None)

# Vẽ biểu đồ cột
labels = [f'Lớp {i}' for i in range(len(precision))]
x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(10, 5))
plt.bar(x - width, precision, width, label='Precision', color='skyblue')
plt.bar(x, recall, width, label='Recall', color='lightgreen')
plt.bar(x + width, f1, width, label='F1-score', color='salmon')

plt.xticks(x, labels)
plt.ylabel('Giá trị')
plt.title('Precision, Recall, F1-score theo từng lớp')
plt.legend()
plt.tight_layout()
plt.show()

#######################################################################################################################

# Thêm 2 thư viện này vào đầu file nếu chưa có
from sklearn.metrics import confusion_matrix
import seaborn as sns

# (Đoạn code tính y_pred_classes và y_true_classes của bạn đã có sẵn)
# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true_classes = np.argmax(y_test, axis=1)

# 3. Vẽ ma trận nhầm lẫn (Confusion Matrix)
print("📊 Đang tạo ma trận nhầm lẫn...")
cm = confusion_matrix(y_true_classes, y_pred_classes)
class_names = [f'Nhãn {i}' for i in range(num_classes)] # num_classes = 5

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)

plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
plt.ylabel('Nhãn thực tế (Actual)')
plt.xlabel('Nhãn dự đoán (Predicted)')
plt.show()
