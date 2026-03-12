import os
import random
import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import datasets, transforms, models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# =========================
# 基本设置
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

data_dir = os.path.expanduser("~/tomato_project/data/TLID_split/test")
weight_path = os.path.expanduser("~/tomato_project/outputs/resnet18_best_server.pth")
save_dir = os.path.expanduser("~/tomato_project/outputs/gradcam_resnet18")
os.makedirs(save_dir, exist_ok=True)

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
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# 仅用于可视化，不做标准化
vis_transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

# =========================
# 加载模型
# =========================
num_classes = 7
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(weight_path, map_location=device))
model = model.to(device)
model.eval()

# ResNet18 常见 target layer：layer4[-1]
target_layers = [model.layer4[-1]]

# =========================
# 收集测试图片路径
# =========================
samples = []
for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    for fname in os.listdir(class_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            samples.append((os.path.join(class_dir, fname), class_idx, class_name))

print("Total test images found:", len(samples))

# =========================
# 先找每类 1 张“预测正确”的样本
# =========================
selected_correct = {}
selected_wrong = []

for img_path, true_idx, true_name in samples:
    img_pil = Image.open(img_path).convert("RGB")
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = int(torch.argmax(output, dim=1).item())

    if pred_idx == true_idx and true_idx not in selected_correct:
        selected_correct[true_idx] = (img_path, true_idx, pred_idx)

    # 额外收集 WhiteFly 的错误样本
    if true_idx == 6 and pred_idx != true_idx and len(selected_wrong) < 2:
        selected_wrong.append((img_path, true_idx, pred_idx))

    if len(selected_correct) == 7 and len(selected_wrong) >= 2:
        break

print("Correct samples selected:", len(selected_correct))
print("Wrong WhiteFly samples selected:", len(selected_wrong))

# =========================
# 生成并保存 Grad-CAM
# =========================
with GradCAM(model=model, target_layers=target_layers) as cam:
    # 正确分类样本
    for class_idx in range(7):
        if class_idx not in selected_correct:
            print(f"Warning: no correct sample found for class {class_idx}")
            continue

        img_path, true_idx, pred_idx = selected_correct[class_idx]
        img_pil = Image.open(img_path).convert("RGB")
        img_vis = vis_transform(img_pil)
        rgb_img = np.array(img_vis).astype(np.float32) / 255.0

        input_tensor = transform(img_pil).unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(pred_idx)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        out_name = f"correct_class{true_idx}_pred{pred_idx}_{os.path.basename(img_path)}"
        Image.fromarray(cam_image).save(os.path.join(save_dir, out_name))
        print("Saved:", out_name)

    # 错误分类样本（WhiteFly）
    for i, (img_path, true_idx, pred_idx) in enumerate(selected_wrong):
        img_pil = Image.open(img_path).convert("RGB")
        img_vis = vis_transform(img_pil)
        rgb_img = np.array(img_vis).astype(np.float32) / 255.0

        input_tensor = transform(img_pil).unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(pred_idx)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        out_name = f"wrong_{i}_true{true_idx}_pred{pred_idx}_{os.path.basename(img_path)}"
        Image.fromarray(cam_image).save(os.path.join(save_dir, out_name))
        print("Saved:", out_name)

print("\nDone. Grad-CAM images saved to:", save_dir)