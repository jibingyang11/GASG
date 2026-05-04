"""
PersonDataset loader.

Expected directory structure:
  PersonDataset/
    train/
      left/   *.png
      right/  *.png
      depth/  *.png  (16-bit depth maps)
    test/
      left/   *.png
      right/  *.png
      depth/  *.png  or depth_left_truth/ *.png

Download from: https://huggingface.co/datasets/jibingyang111/PersonDataset
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class PersonDataset(Dataset):
    """PersonDataset: Unity-synthesised dense-crowd stereo benchmark."""

    def __init__(self, root, split='train', height=256, width=512,
                 augment=False):
        """
        Args:
            root: path to PersonDataset root
            split: 'train' or 'test'
            height, width: resize resolution
            augment: whether to apply data augmentation (training only)
        """
        self.root = root
        self.split = split
        self.height = height
        self.width = width
        self.augment = augment and (split == 'train')

        split_dir = os.path.join(root, split)
        self.left_paths = sorted(glob.glob(
            os.path.join(split_dir, 'left', '*.png')))
        self.right_paths = sorted(glob.glob(
            os.path.join(split_dir, 'right', '*.png')))
        depth_dir = os.path.join(split_dir, 'depth')
        if not os.path.isdir(depth_dir):
            depth_dir = os.path.join(split_dir, 'depth_left_truth')
        self.depth_paths = sorted(glob.glob(os.path.join(depth_dir, '*.png')))

        assert len(self.left_paths) == len(self.right_paths), \
            f"Mismatch: {len(self.left_paths)} left vs {len(self.right_paths)} right"
        assert len(self.left_paths) == len(self.depth_paths), \
            f"Mismatch: {len(self.left_paths)} left vs {len(self.depth_paths)} depth"

        print(f"PersonDataset [{split}]: {len(self.left_paths)} samples")

    def __len__(self):
        return len(self.left_paths)

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        img = img.resize((self.width, self.height), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        return img

    def _load_depth(self, path):
        depth = Image.open(path)
        depth = depth.resize((self.width, self.height), Image.NEAREST)
        depth = np.array(depth, dtype=np.float32)
        if depth.ndim == 3:
            # HuggingFace PersonDataset stores grayscale depth as RGBA.
            depth = depth[..., 0]
        # PersonDataset stores 8-bit depth in decimetres. Keep generic
        # handling for future 16-bit exports in millimetres/centimetres.
        if depth.max() <= 255:
            depth = depth / 10.0
        elif depth.max() > 1000:
            depth = depth / 1000.0
        elif depth.max() > 100:
            depth = depth / 100.0
        return depth

    def __getitem__(self, idx):
        left = self._load_image(self.left_paths[idx])
        right = self._load_image(self.right_paths[idx])
        depth = self._load_depth(self.depth_paths[idx])

        if self.augment:
            # Random brightness/contrast
            if np.random.random() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                left = np.clip(left * brightness, 0, 1)
                right = np.clip(right * brightness, 0, 1)

        # Convert to [-1, 1] for GASG
        left_tensor = torch.from_numpy(left).permute(2, 0, 1).float() * 2 - 1
        right_tensor = torch.from_numpy(right).permute(2, 0, 1).float() * 2 - 1
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()

        return {
            'left': left_tensor,
            'right': right_tensor,
            'depth': depth_tensor,
            'left_path': self.left_paths[idx],
        }
