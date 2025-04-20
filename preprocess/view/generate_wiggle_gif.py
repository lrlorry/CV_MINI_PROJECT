import os
import argparse
from PIL import Image

def load_images(left_path, right_path):
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")

    # Ensure same size
    if left.size != right.size:
        print("⚠️ Image sizes do not match! Resizing right image.")
        right = right.resize(left.size)

    return [left, right]

def generate_wiggle_gif(images, output_path, duration=100, loop=0):
    sequence = images + images[::-1]  # Left → Right → Left 循环
    sequence[0].save(output_path,
                     save_all=True,
                     append_images=sequence[1:],
                     duration=duration,
                     loop=loop)
    print(f"[✓] Wiggle GIF saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=str, default="views/view_left.png", help="Path to left view image")
    parser.add_argument("--right", type=str, default="views/view_right.png", help="Path to right view image")
    parser.add_argument("--output", type=str, default="output/wiggle.gif", help="Output path for wiggle GIF")
    parser.add_argument("--duration", type=int, default=100, help="Frame duration (ms)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    frames = load_images(args.left, args.right)
    generate_wiggle_gif(frames, args.output, duration=args.duration)

if __name__ == "__main__":
    main()
