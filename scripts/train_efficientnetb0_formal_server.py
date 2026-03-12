import os
import copy
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.set_num_threads(6)
torch.set_num_interop_threads(6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

data_dir = os.path.expanduser("~/tomato_project/data/TLID_split")
output_dir = os.path.expanduser("~/tomato_project/outputs")
os.makedirs(output_dir, exist_ok=True)

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=test_tf)
test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=48, shuffle=True, num_workers=6, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=48, shuffle=False, num_workers=6, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=48, shuffle=False, num_workers=6, pin_memory=True)

print("Classes:", train_ds.classes)
print("Train size:", len(train_ds))
print("Val size:", len(val_ds))
print("Test size:", len(test_ds))

num_classes = len(train_ds.classes)

# EfficientNet-B0 模型
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

best_wts = copy.deepcopy(model.state_dict())
best_f1 = 0.0
patience = 5
counter = 0
epochs = 20

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    for phase, loader in [("train", train_loader), ("val", val_loader)]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        all_preds, all_labels = [], []

        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average="macro")

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")

        if phase == "val":
            if epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_wts = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1

    if counter >= patience:
        print("Early stopping triggered.")
        break

model.load_state_dict(best_wts)
torch.save(model.state_dict(), os.path.join(output_dir, "efficientnetb0_best_server.pth"))

print("\nBest val Macro-F1:", best_f1)

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
test_f1 = f1_score(all_labels, all_preds, average="macro")

print("\n=== Test Results ===")
print("Test Accuracy:", test_acc)
print("Test Macro-F1:", test_f1)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=train_ds.classes))
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))