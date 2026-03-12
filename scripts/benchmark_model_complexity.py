import os
import time
import csv
import warnings

import torch
import timm
from torch import nn
from torchvision import models
from thop import profile

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

# =========================
# 基本设置
# =========================
NUM_CLASSES = 7
INPUT_SIZE = (1, 3, 224, 224)   # batch_size=1，用于 latency
WARMUP = 50
RUNS = 200

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

project_dir = os.path.expanduser("~/tomato_project")
output_dir = os.path.join(project_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

# =========================
# 模型与权重路径
# =========================
model_weight_map = {
    "MobileNetV3-Large": os.path.join(output_dir, "mobilenetv3_best_server.pth"),
    "ShuffleNetV2_x1_0": os.path.join(output_dir, "shufflenetv2_best_server.pth"),
    "MobileNetV2": os.path.join(output_dir, "mobilenetv2_best_server.pth"),
    "EfficientNet-B0": os.path.join(output_dir, "efficientnetb0_best_server.pth"),
    "ResNet18": os.path.join(output_dir, "resnet18_best_server.pth"),
}

# =========================
# 构建模型
# =========================
def build_model(model_name: str):
    if model_name == "MobileNetV3-Large":
        model = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=False,
            num_classes=NUM_CLASSES
        )

    elif model_name == "ShuffleNetV2_x1_0":
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    elif model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    elif model_name == "ResNet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


# =========================
# 加载权重
# =========================
def load_weights_if_exists(model, weight_path: str):
    loaded = False

    if not os.path.exists(weight_path):
        print(f"[WARN] Weight file not found: {weight_path}")
        return model, loaded

    checkpoint = torch.load(weight_path, map_location="cpu")

    # 兼容不同保存格式
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 去掉可能存在的 "module." 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    loaded = True
    return model, loaded


# =========================
# 参数量
# =========================
def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# =========================
# MACs / FLOPs
# 说明：
# thop 返回的是 MACs
# 常用近似：FLOPs = 2 × MACs
# =========================
def compute_macs_flops(model, device):
    model.eval()
    dummy = torch.randn(*INPUT_SIZE).to(device)

    with torch.inference_mode():
        macs, params = profile(model, inputs=(dummy,), verbose=False)

    flops = macs * 2
    return macs, flops, params


# =========================
# 推理时间（batch_size=1）
# =========================
def measure_latency(model, device, warmup=WARMUP, runs=RUNS):
    model.eval()
    dummy = torch.randn(*INPUT_SIZE).to(device)

    with torch.inference_mode():
        # warmup
        for _ in range(warmup):
            _ = model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(runs):
            _ = model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

    avg_time_s = (end - start) / runs
    latency_ms = avg_time_s * 1000
    fps = 1.0 / avg_time_s

    return latency_ms, fps


# =========================
# 格式化
# =========================
def to_million(x):
    return x / 1e6

def to_giga(x):
    return x / 1e9


# =========================
# 主程序
# =========================
def main():
    model_names = [
        "MobileNetV3-Large",
        "ShuffleNetV2_x1_0",
        "MobileNetV2",
        "EfficientNet-B0",
        "ResNet18",
    ]

    results = []

    for model_name in model_names:
        print("\n" + "=" * 70)
        print(f"Benchmarking: {model_name}")

        model = build_model(model_name)
        weight_path = model_weight_map[model_name]
        model, loaded = load_weights_if_exists(model, weight_path)
        model = model.to(device)

        total_params, trainable_params = count_params(model)
        macs, flops, thop_params = compute_macs_flops(model, device)
        latency_ms, fps = measure_latency(model, device)

        row = {
            "Model": model_name,
            "WeightsLoaded": loaded,
            "TotalParams_M": round(to_million(total_params), 4),
            "TrainableParams_M": round(to_million(trainable_params), 4),
            "MACs_G": round(to_giga(macs), 4),
            "FLOPs_G": round(to_giga(flops), 4),
            "Latency_ms_bs1": round(latency_ms, 4),
            "FPS_bs1": round(fps, 4),
        }
        results.append(row)

        print(row)

    # 保存 CSV
    csv_path = os.path.join(output_dir, "model_complexity_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Model",
                "WeightsLoaded",
                "TotalParams_M",
                "TrainableParams_M",
                "MACs_G",
                "FLOPs_G",
                "Latency_ms_bs1",
                "FPS_bs1",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved CSV:", csv_path)


if __name__ == "__main__":
    main()