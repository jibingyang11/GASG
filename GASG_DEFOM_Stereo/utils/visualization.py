"""Visualization helpers for depth maps and qualitative figures."""

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def colorize_depth(depth, min_d=None, max_d=None, cmap='magma'):
    """
    Convert depth map to colourized image.

    Args:
        depth: (H, W) numpy array
        min_d, max_d: depth range for normalization
        cmap: matplotlib colormap name

    Returns:
        (H, W, 3) numpy array in [0, 1]
    """
    if min_d is None:
        min_d = depth[depth > 0].min() if (depth > 0).any() else 0
    if max_d is None:
        max_d = depth.max()

    depth_norm = (depth - min_d) / (max_d - min_d + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)

    cm = plt.get_cmap(cmap)
    colored = cm(depth_norm)[:, :, :3]  # drop alpha
    return colored.astype(np.float32)


def tile_images(images, titles=None, cols=None, figsize_per_col=3,
                save_path=None, dpi=300):
    """
    Tile a list of images into a single figure.

    Args:
        images: list of (H, W, 3) numpy arrays in [0, 1]
        titles: optional list of titles
        cols: number of columns (default: len(images))
        save_path: if set, save to this path
        dpi: output resolution
    """
    n = len(images)
    if cols is None:
        cols = n
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(figsize_per_col * cols,
                                      figsize_per_col * rows * 0.6))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(np.clip(img, 0, 1))
        ax.axis('off')
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=10)

    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout(pad=0.5)
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def make_comparison_figure(left_img, depth_maps, method_names,
                           save_path, dpi=300):
    """
    Create a comparison figure: input + multiple depth maps.

    Args:
        left_img: (H, W, 3) input image in [0, 1]
        depth_maps: list of (H, W) depth arrays
        method_names: list of method names
        save_path: output path
    """
    images = [left_img]
    titles = ['Input']

    for d, name in zip(depth_maps, method_names):
        images.append(colorize_depth(d))
        titles.append(name)

    tile_images(images, titles=titles, save_path=save_path, dpi=dpi)
