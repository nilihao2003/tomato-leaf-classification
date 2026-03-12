# Tomato Leaf Disease Classification on TLID

Code for comparing lightweight CNN models on the public TLID dataset for greenhouse tomato leaf disease classification.

## Models

* MobileNetV3-Large
* ShuffleNetV2_x1_0
* MobileNetV2
* EfficientNet-B0
* ResNet18

## Dataset

TLID: [https://doi.org/10.17632/kt64b2kh89.2](https://doi.org/10.17632/kt64b2kh89.2)

Expected structure:

```text
data/TLID_split/
├─ train/
├─ val/
└─ test/
```

Classes:

* 0-Healthy
* 1-Miner
* 2-BacterialSpot
* 3-PowderyMildew
* 4-PowderyMildew_Miner
* 5-BacterialSpot_Miner
* 6-WhiteFly

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python scripts/train_mobilenetv3_formal_server.py
python scripts/train_shufflenet_formal_server.py
python scripts/train_mobilenetv2_formal_server.py
python scripts/train_efficientnetb0_formal_server.py
python scripts/train_resnet18_formal_server.py
```

## Analysis

```bash
python scripts/gradcam_resnet18_tlid.py
python scripts/benchmark_model_complexity.py
python scripts/plot_confusion_matrix_resnet18.py
python scripts/make_gradcam_figure2.py
```

## Main finding

EfficientNet-B0 was the best lightweight model, while ResNet18 achieved the best overall performance.

## Contact

Corresponding author: Lihao Ni
Email: nilihao@wzvcst.edu.cn

