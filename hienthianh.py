from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Ánh xạ nhãn sang mô tả tiếng Việt
LABEL_MAP = {
    0: "Binh thuong",
    1: "Bet nhe",
    2: "Bet trung bình",
    3: "Bet nang",
    4: "Khong xac đinh"
}

# Hàm tiền xử lý ảnh xám
def preprocess_image(img_gray):
    img = cv2.equalizeHist(img_gray)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # (224, 224, 1)
    img = np.expand_dims(img, axis=0)   # (1, 224, 224, 1)
    return img

# Hàm dự đoán và hiển thị kết quả
def predict_image(path, model_path='d:/banchan/flatfoot_model_usuk.keras'):
    # 1. Tải mô hình
    model = load_model(model_path)

    # 2. Đọc ảnh gốc
    original_img = cv2.imread(path)
    if original_img is None:
        print("❌ Không đọc được ảnh hoặc đường dẫn sai:", path)
        return

    # 3. Chuyển sang ảnh xám
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # 4. Tiền xử lý
    processed_img = preprocess_image(gray_img)

    # 5. Dự đoán
    prediction = model.predict(processed_img)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    label_description = LABEL_MAP.get(predicted_class, "Không rõ")

    # 6. In ra console
    print(f"📸 Ảnh: {path}")
    print(f"🔍 Dự đoán: Nhãn {predicted_class} - {label_description} với độ tin cậy {confidence:.2%}")

    # 7. Hiển thị kết quả lên ảnh
    text = f"Nhan {predicted_class}: {label_description} ({confidence:.2%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (20, 40)
    font_scale = 1
    font_color = (0, 255, 0)
    thickness = 2

    cv2.putText(original_img, text, position, font, font_scale, font_color, thickness)
    cv2.imshow("Ket qua du doan va hien thi anh", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Gọi hàm
predict_image('d:/banchan/usuk/59R.jpg')
