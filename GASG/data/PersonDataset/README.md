---
license: apache-2.0
---

## Dataset Introduction

### Dataset Name:PersonDataset

This dataset is a synthetic dataset created using the Unity engine, specifically designed for depth estimation tasks in dense crowd scenarios. It includes left and right images, as well as the ground truth depth maps for the left images. The simulated camera first captures the left image from the left position, then moves one meter horizontally to the right to capture the right image. The camera's focal length is 600 pixels. Depth maps can be converted into disparity maps using the formula disp = (focal length × 1) / depth. The training to testing set ratio is 8:2, with a total of 8,000 image sets. Each set contains a left image, a right image, and the ground truth depth map corresponding to the left image. The resolution of each image is 906 × 415.


