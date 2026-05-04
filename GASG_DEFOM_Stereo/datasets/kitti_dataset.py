"""
KITTI dataset loader for Eigen split evaluation.

Expected directory structure:
  kitti/
    raw/
      2011_09_26/
        2011_09_26_drive_0001_sync/
          image_02/data/  (left images)
          image_03/data/  (right images)
    gt_depths/
      *.npy  (Eigen split ground-truth depth maps)

Download KITTI from: http://www.cvlibs.net/datasets/kitti/
Eigen split files: use the standard 697-image test set.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# Standard Eigen test split (697 images)
# This file list should be downloaded from:
# https://github.com/nianticlabs/monodepth2/blob/master/splits/eigen/test_files.txt
EIGEN_TEST_FILE = "splits/eigen_test_files.txt"


class KITTIDataset(Dataset):
    """KITTI Eigen split dataset."""

    def __init__(self, root, split='test', height=256, width=512,
                 gt_depth_dir=None):
        """
        Args:
            root: path to KITTI raw data root
            split: 'test' (Eigen test split)
            height, width: resize resolution
            gt_depth_dir: path to pre-computed GT depth maps (.npy)
        """
        self.root = root
        self.height = height
        self.width = width
        self.gt_depth_dir = gt_depth_dir or os.path.join(root, 'gt_depths')

        # Load file list
        filelist_path = os.path.join(
            os.path.dirname(__file__), '..', EIGEN_TEST_FILE)
        if os.path.exists(filelist_path):
            with open(filelist_path, 'r') as f:
                lines = f.readlines()
            self.files = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    folder, frame_id, side = parts[0], parts[1], parts[2]
                    self.files.append((folder, frame_id, side))
        else:
            # Fallback: scan directory
            print(f"Warning: {filelist_path} not found. "
                  f"Please download the Eigen test split file list.")
            self.files = []

        print(f"KITTI [{split}]: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def _get_image_path(self, folder, frame_id, side):
        cam = 'image_02' if side == 'l' else 'image_03'
        return os.path.join(self.root, 'raw', folder, cam, 'data',
                            f'{int(frame_id):010d}.png')

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        orig_w, orig_h = img.size
        img = img.resize((self.width, self.height), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        return img, (orig_h, orig_w)

    def __getitem__(self, idx):
        folder, frame_id, side = self.files[idx]

        # Load left image
        left_path = self._get_image_path(folder, frame_id, 'l')
        left, orig_size = self._load_image(left_path)

        # Load right image (for stereo methods)
        right_path = self._get_image_path(folder, frame_id, 'r')
        if os.path.exists(right_path):
            right, _ = self._load_image(right_path)
        else:
            right = np.zeros_like(left)

        # Load GT depth
        gt_path = os.path.join(self.gt_depth_dir, f'{idx:010d}.npy')
        if os.path.exists(gt_path):
            gt_depth = np.load(gt_path)
        else:
            gt_depth = np.zeros((orig_size[0], orig_size[1]),
                                dtype=np.float32)

        # Convert to tensors
        left_tensor = torch.from_numpy(left).permute(2, 0, 1).float() * 2 - 1
        right_tensor = torch.from_numpy(right).permute(2, 0, 1).float() * 2 - 1

        return {
            'left': left_tensor,
            'right': right_tensor,
            'gt_depth': gt_depth,  # keep as numpy for flexible eval
            'orig_size': orig_size,
            'left_path': left_path,
        }


class KITTIEigen652Dataset(Dataset):
    """Pre-cropped KITTI Eigen-valid subset from xcll/kitti_eigen_test_652."""

    def __init__(self, root, height=256, width=512):
        self.root = root
        self.height = height
        self.width = width
        self.images_dir = os.path.join(root, 'images')
        self.depth_dir = os.path.join(root, 'depth_float32_h5')

        def _idx(path):
            return int(os.path.basename(path).split('_')[0].split('.')[0])

        self.image_paths = sorted(
            [os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir)
             if f.endswith('_condition.png')],
            key=_idx,
        )
        self.depth_paths = [
            os.path.join(self.depth_dir,
                         os.path.basename(p).split('_')[0] + '.h5')
            for p in self.image_paths
        ]
        missing = [p for p in self.depth_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing depth files, e.g. {missing[0]}")

        print(f"KITTI Eigen-valid 652: {len(self.image_paths)} samples")

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        orig_w, orig_h = img.size
        img = img.resize((self.width, self.height), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        return img, (orig_h, orig_w)

    def _load_depth(self, path):
        import h5py
        with h5py.File(path, 'r') as f:
            return np.array(f['depth'], dtype=np.float32)

    def __getitem__(self, idx):
        left, orig_size = self._load_image(self.image_paths[idx])
        gt_depth = self._load_depth(self.depth_paths[idx])
        left_tensor = torch.from_numpy(left).permute(2, 0, 1).float() * 2 - 1
        return {
            'left': left_tensor,
            'right': torch.zeros_like(left_tensor),
            'gt_depth': gt_depth,
            'orig_size': orig_size,
            'left_path': self.image_paths[idx],
        }
