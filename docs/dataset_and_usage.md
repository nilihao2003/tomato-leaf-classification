# Dataset and Usage Notes

## Dataset Source
This project uses the public Tomato Leaf Image Dataset (TLID).

- DOI: https://doi.org/10.17632/kt64b2kh89.2

## Dataset Organization
The original dataset should be downloaded from the official source and manually organized into:

- train
- val
- test

with the following seven classes:

- 0-Healthy
- 1-Miner
- 2-BacterialSpot
- 3-PowderyMildew
- 4-PowderyMildew_Miner
- 5-BacterialSpot_Miner
- 6-WhiteFly

## Usage
1. Download the dataset from the DOI link.
2. Split or place the images into train/val/test folders.
3. Run the training scripts in the `scripts/` directory.
4. Use the analysis scripts to generate Grad-CAM, model complexity results, and confusion matrices.

## Important Note
This repository does not redistribute the dataset itself. Users should obtain the dataset from the official public source.