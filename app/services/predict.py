from ultralytics import YOLO
import cv2
import uuid
import os
import base64

# Tải mô hình YOLO đã huấn luyện
model = YOLO("model/yolo10l_final.pt")  # điều chỉnh path nếu cần

# Bản đồ nhãn
LABEL_MAP = {
    0: "plasma",
    1: "non-plasma"
}

# Màu vẽ box
COLOR_MAP = {
    0: (255, 0, 0),      # đỏ cho plasma
    1: (0, 255, 0)       # xanh cho non-plasma
}

async def predict_image(file):
    # Lưu ảnh tạm thời
    temp_name = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_name, "wb") as f:
        f.write(await file.read())

    # Đọc ảnh
    image = cv2.imread(temp_name)

    # Dự đoán với YOLO
    results = model(temp_name, conf=0.5, verbose=False)[0]

    plasma_count = 0
    non_plasma_count = 0
    boxes = []

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box[:4])
        label_id = int(cls)
        label = LABEL_MAP.get(label_id, "unknown")

        # Đếm
        if label_id == 0:
            plasma_count += 1
        elif label_id == 1:
            non_plasma_count += 1

        # Vẽ khung
        color = COLOR_MAP.get(label_id, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        boxes.append({
            "label": label,
            "box": [x1, y1, x2, y2]
        })

    # Encode ảnh thành base64
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Dọn file
    os.remove(temp_name)

    return {
        "plasma_cells": plasma_count,
        "non_plasma_cells": non_plasma_count,
        "boxes": boxes,
        "image_base64": image_base64
    }
