import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 1. Cấu hình & Danh sách nhãn bệnh
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES = 8 
MODEL_PATH = "best_model_fast20e.pth"
IMAGE_PATH = "test.jpg" 

# Mapping ID sang tên bệnh thực tế
CLASSES = {
    1: "akiec (Actinic keratoses)",
    2: "bcc (Basal cell carcinoma)",
    3: "bkl (Benign keratosis-like lesions)",
    4: "df (Dermatofibroma)",
    5: "mel (Melanoma)",
    6: "nv (Melanocytic nevi)",
    7: "vasc (Vascular lesions)"
}

# 2. Khởi tạo & Load Model
def get_model():
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model

model = get_model().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 3. Tiền xử lý ảnh Input
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Chuyển đổi tensor chuẩn hóa giống lúc train
image_tensor = torch.from_numpy(image_rgb / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

# 4. Dự đoán
with torch.no_grad():
    prediction = model(image_tensor)

# 5. Xử lý & Vẽ kết quả "Xịn"
boxes = prediction[0]['boxes'].cpu().numpy()
scores = prediction[0]['scores'].cpu().numpy()
labels = prediction[0]['labels'].cpu().numpy()

# Thiết lập ngưỡng lọc
THRESHOLD = 0.6 

plt.figure(figsize=(10, 8))
plt.imshow(image_rgb)
ax = plt.gca()

# Chỉ lấy khung có score cao nhất nếu có nhiều khung chồng lấn
if len(scores) > 0:
    # Sắp xếp lấy những khung tốt nhất
    best_indices = np.where(scores > THRESHOLD)[0]
    
    for i in best_indices:
        box = boxes[i].astype(int)
        label_id = labels[i]
        score = scores[i]
        label_name = CLASSES.get(label_id, "Unknown")

        # Vẽ khung màu xanh lá (Green) cho chuyên nghiệp
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                             fill=False, color='lime', linewidth=3)
        ax.add_patch(rect)

        # Tạo nhãn dán có nền đỏ chữ trắng để nổi bật
        label_text = f"{label_name}: {score:.2f}"
        ax.text(box[0], box[1]-7, label_text, fontsize=12, fontweight='bold',
                color='white', bbox=dict(facecolor='red', alpha=0.8, edgecolor='none'))

plt.title(f"HỆ THỐNG CHẨN ĐOÁN BỆNH DA LIỄU AI", fontsize=14, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.show()