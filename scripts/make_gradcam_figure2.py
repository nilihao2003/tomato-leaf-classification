import os
from PIL import Image
import matplotlib.pyplot as plt

project_dir = os.path.expanduser("~/tomato_project")
test_dir = os.path.join(project_dir, "data", "TLID_split", "test")
gradcam_dir = os.path.join(project_dir, "outputs", "gradcam_resnet18")
output_dir = os.path.join(project_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

samples = [
    {
        "title": "Healthy\n(True: Healthy, Pred: Healthy)",
        "raw_class": "0-Healthy",
        "raw_file": "IMG_503_0.jpg",
        "cam_file": "correct_class0_pred0_IMG_503_0.jpg",
    },
    {
        "title": "BacterialSpot\n(True: BacterialSpot, Pred: BacterialSpot)",
        "raw_class": "2-BacterialSpot",
        "raw_file": "IMG_1770_2.jpg",
        "cam_file": "correct_class2_pred2_IMG_1770_2.jpg",
    },
    {
        "title": "BacterialSpot_Miner\n(True: BacterialSpot_Miner, Pred: BacterialSpot_Miner)",
        "raw_class": "5-BacterialSpot_Miner",
        "raw_file": "IMG_1829_5.jpg",
        "cam_file": "correct_class5_pred5_IMG_1829_5.jpg",
    },
    {
        "title": "WhiteFly (misclassified)\n(True: WhiteFly, Pred: Healthy)",
        "raw_class": "6-WhiteFly",
        "raw_file": "IMG_53_6.jpg",
        "cam_file": "wrong_0_true6_pred0_IMG_53_6.jpg",
    },
]

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 16))

for i, s in enumerate(samples):
    raw_path = os.path.join(test_dir, s["raw_class"], s["raw_file"])
    cam_path = os.path.join(gradcam_dir, s["cam_file"])

    raw_img = Image.open(raw_path).convert("RGB")
    cam_img = Image.open(cam_path).convert("RGB")

    axes[i, 0].imshow(raw_img)
    axes[i, 0].set_title(f"{s['title']}\nOriginal", fontsize=10)
    axes[i, 0].axis("off")

    axes[i, 1].imshow(cam_img)
    axes[i, 1].set_title(f"{s['title']}\nGrad-CAM", fontsize=10)
    axes[i, 1].axis("off")

plt.tight_layout()
save_path = os.path.join(output_dir, "figure2_gradcam_resnet18.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", save_path)