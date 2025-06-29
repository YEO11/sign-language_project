import cv2
import torch
from torchvision import transforms
from models.tsm_model import TSMModel
from collections import deque
from PIL import Image, ImageFont, ImageDraw
import csv
import numpy as np

# 클래스 이름 불러오기
def load_class_names_exact(csv_path):
    class_names = []
    seen = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                label = parts[1]
                if label not in seen:
                    class_names.append(label)
                    seen.add(label)
    print(f"🔵 클래스 개수: {len(class_names)}")
    return class_names

# CSV에서 클래스 이름 추출
class_names = load_class_names_exact("dataset/label.csv")

# 모델 설정
num_classes = len(class_names)
model = TSMModel(num_classes=num_classes)
model.load_state_dict(torch.load("weights/tsm_epoch10.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

if device.type == "cuda":
    print(f"🔵 GPU: {torch.cuda.get_device_name(0)}")

# 프레임 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 웹캠 캡처
cap = cv2.VideoCapture(0)
frames = deque(maxlen=16)

# 한글 폰트 경로 설정
font_path = "fonts/NanumGothic-Regular.ttf"  # ← 실제 위치에 맞게 조정
font = ImageFont.truetype(font_path, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    img_tensor = transform(pil_img)
    frames.append(img_tensor)

    if len(frames) == 16:
        with torch.no_grad():
            input_tensor = torch.stack(list(frames))  # (T, C, H, W)
            input_tensor = input_tensor.unsqueeze(0).to(device)  # (1, T, C, H, W)

            pred = model(input_tensor)
            pred_label = torch.argmax(pred, dim=1).item()
            label = class_names[pred_label]

        # PIL로 draw (한글)
        draw = ImageDraw.Draw(pil_img)
        draw.text((30, 30), label, font=font, fill=(0, 255, 0))  # RGB

        # 다시 OpenCV로 변환
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    cv2.imshow("Sign Language Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
torch.cuda.empty_cache()
