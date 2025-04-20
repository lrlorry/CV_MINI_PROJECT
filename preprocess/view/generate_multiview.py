import os
import argparse
import numpy as np
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image)

def load_depth(depth_path):
    depth = np.load(depth_path)
    print(f"[✓] Depth loaded: shape={depth.shape}, min={depth.min():.3f}, max={depth.max():.3f}")
    return (depth - depth.min()) / (depth.max() - depth.min())  # Normalize to 0~1

def shift_image_by_depth(image, depth, scale):
    h, w = depth.shape
    shifted_image = np.zeros_like(image)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    shift = depth * scale  # positive: right, negative: left
    x_shifted = np.clip((xx + shift).astype(np.int32), 0, w - 1)

    for c in range(3):
        shifted_image[..., c] = image[yy, x_shifted, c]

    return shifted_image

def generate_multiviews(image, depth, num_views=5, max_shift=60):
    """Generate N views from left (-max_shift) to right (+max_shift)"""
    views = []
    mid = num_views // 2
    for i in range(num_views):
        alpha = (i - mid) / mid  # from -1 to +1
        scale = alpha * max_shift
        view = shift_image_by_depth(image, depth, scale)
        views.append(view)
    return views

def save_views(views, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, view in enumerate(views):
        path = os.path.join(output_dir, f"view_{i:02d}.png")
        Image.fromarray(view).save(path)
        print(f"[✓] Saved: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="jcsmr.jpg", help="Path to original image")
    parser.add_argument("--depth", type=str, default="depth/depth-anything-large-hf_3/jcsmr_depth.npy", help="Path to depth .npy")
    parser.add_argument("--output_dir", type=str, default="views/multiview", help="Directory to save views")
    parser.add_argument("--num_views", type=int, default=5, help="Total number of views")
    parser.add_argument("--max_shift", type=float, default=60.0, help="Max shift in pixels")
    args = parser.parse_args()

    image = load_image(args.image)
    depth = load_depth(args.depth)
    views = generate_multiviews(image, depth, args.num_views, args.max_shift)
    save_views(views, args.output_dir)

    print(f"\n[✓] Generated {len(views)} multi-view images in {args.output_dir}")

if __name__ == "__main__":
    main()
