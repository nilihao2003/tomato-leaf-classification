import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# =========================
# 基本设置
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

data_dir = os.path.expanduser("~/tomato_project/data/TLID_split")
output_dir = os.path.expanduser("~/tomato_project/outputs")
os.makedirs(output_dir, exist_ok=True)

weight_path = os.path.join(output_dir, "resnet18_best_server.pth")

class_names = [
    "0-Healthy",
    "1-Miner",
    "2-BacterialSpot",
    "3-PowderyMildew",
    "4-PowderyMildew_Miner",
    "5-BacterialSpot_Miner",
    "6-WhiteFly"
]

# =========================
# 数据变换
# =========================
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=test_tf)
test_loader = DataLoader(test_ds, batch_size=48, shuffle=False, num_workers=6, pin_memory=True)

print("Test size:", len(test_ds))
print("Classes from dataset:", test_ds.classes)

# =========================
# 加载模型
# =========================
num_classes = 7
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

state_dict = torch.load(weight_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# =========================
# 推理
# =========================
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# =========================
# 计算混淆矩阵
# =========================
cm = confusion_matrix(all_labels, all_preds)
cm_norm = confusion_matrix(all_labels, all_preds, normalize="true")

print("\nConfusion Matrix:")
print(cm)

# 保存原始矩阵 CSV
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_csv_path = os.path.join(output_dir, "resnet18_confusion_matrix.csv")
cm_df.to_csv(cm_csv_path, encoding="utf-8-sig")

# 保存归一化矩阵 CSV
cm_norm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
cm_norm_csv_path = os.path.join(output_dir, "resnet18_confusion_matrix_normalized.csv")
cm_norm_df.to_csv(cm_norm_csv_path, encoding="utf-8-sig")

# =========================
# 画原始 confusion matrix
# =========================
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False, values_format="d")
plt.title("Confusion Matrix of ResNet18 on TLID Test Set")
plt.tight_layout()
raw_png_path = os.path.join(output_dir, "resnet18_confusion_matrix.png")
plt.savefig(raw_png_path, dpi=300, bbox_inches="tight")
plt.close()

# =========================
# 画归一化 confusion matrix
# =========================
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False, values_format=".2f")
plt.title("Normalized Confusion Matrix of ResNet18 on TLID Test Set")
plt.tight_layout()
norm_png_path = os.path.join(output_dir, "resnet18_confusion_matrix_normalized.png")
plt.savefig(norm_png_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved files:")
print(raw_png_path)
print(norm_png_path)
print(cm_csv_path)
print(cm_norm_csv_path)