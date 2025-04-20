import os
import argparse
from PIL import Image

def load_views(input_dir, prefix="view_", suffix=".png"):
    files = sorted([f for f in os.listdir(input_dir) if f.startswith(prefix) and f.endswith(suffix)])
    views = [Image.open(os.path.join(input_dir, f)).convert("RGB") for f in files]
    print(f"[✓] Loaded {len(views)} views from {input_dir}")
    return views

def make_wiggle_loop(views):
    """Creates: view_0 → view_1 → ... → view_n → view_n-1 → ... → view_1"""
    return views + views[-2:0:-1]  # exclude last and first repeat

def save_gif(frames, output_path, duration=100):
    frames[0].save(output_path,
                   save_all=True,
                   append_images=frames[1:],
                   duration=duration,
                   loop=0)
    print(f"[✓] Wiggle GIF saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="views/multiview_4_dataset", help="Directory with view_XX.png images")
    parser.add_argument("--output", type=str, default="output/multiview_4_dataset_wiggle.gif", help="Output GIF path")
    parser.add_argument("--duration", type=int, default=100, help="Frame duration (ms)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    views = load_views(args.input_dir)
    frames = make_wiggle_loop(views)
    save_gif(frames, args.output, args.duration)

if __name__ == "__main__":
    main()
