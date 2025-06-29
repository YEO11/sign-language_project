import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.tsm_model import TSMModel
from utils.data_loader import SignLanguageDataset
from utils.plot_utils import save_metrics_plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"🔵 GPU: {torch.cuda.get_device_name(0)}")

# 설정
csv_path = 'dataset/label.csv'
video_dir = 'dataset/videos'
batch_size = 4
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("weights", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

# 데이터셋 불러오기
full_dataset = SignLanguageDataset(csv_path, video_dir, num_frames=16)
num_classes = full_dataset.num_classes

# TRAIN/VALIDATION (80:20)
val_size = int(len(full_dataset) * 0.2)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 모델 준비
model = TSMModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_loss_list, train_acc_list = [], []
val_loss_list, val_acc_list = [], []

for epoch in range(epochs):
    # TRAIN
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs} - Train]")

    for videos, labels in loop:
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    train_loss_list.append(train_loss / len(train_loader))
    train_acc_list.append(100 * correct / total)

    # VALIDATION
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_loss_list.append(val_loss / len(val_loader))
    val_acc_list.append(100 * val_correct / val_total)
    print(f"🔵 Validation - Loss: {val_loss_list[-1]:.4f}, Acc: {val_acc_list[-1]:.2f}%")

    # 모델 저장
    torch.save(model.state_dict(), f"weights/tsm_epoch{epoch+1}.pth")

# 시각화 그래프 저장
save_metrics_plot(train_loss_list, train_acc_list, "metrics/tsm_train_metrics.png")
save_metrics_plot(val_loss_list, val_acc_list, "metrics/tsm_val_metrics.png")
print("🔵 학습 완료!")
