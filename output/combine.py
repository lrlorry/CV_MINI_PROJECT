from PIL import Image
import os

# 重新加载图像路径（因为代码状态重置，需重新声明）
image_paths = [
    "output/result_hsv_sketch_l1_1.jpg",
    "output/result_hsv_sketch_light.jpg",
    "output/result_hsv_sketch_loss_t.jpg",
    "output/result_hsv_sketch_night.jpg",
    "output/result_hsv_sketch_v1.jpg",
    "output/result_original_sketch_alpha_1.jpg",
    "output/result_original_sketch_v3_v2.jpg",
    "output/result_original_sketch_white.jpg",
    "output/result_original_sketch.jpg"
]

# 加载图像对象
images = [Image.open(path) for path in image_paths]

# 统一高度
min_height = min(img.height for img in images)
resized_images = [img.resize((int(img.width * min_height / img.height), min_height)) for img in images]

# 拼接成3行 × 3列
row_images = []
for i in range(0, 9, 3):
    row = Image.new('RGB', (sum(img.width for img in resized_images[i:i+3]), min_height))
    x_offset = 0
    for img in resized_images[i:i+3]:
        row.paste(img, (x_offset, 0))
        x_offset += img.width
    row_images.append(row)

# 拼接所有行
total_height = len(row_images) * min_height
final_width = max(row.width for row in row_images)
final_image = Image.new('RGB', (final_width, total_height))

y_offset = 0
for row in row_images:
    final_image.paste(row, (0, y_offset))
    y_offset += min_height

# 保存输出图像
output_path = "output/final_9styles_grid.png"
final_image.save(output_path)
output_path
