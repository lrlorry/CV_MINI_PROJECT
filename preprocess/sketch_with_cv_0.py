import os
import cv2
import numpy as np
from pathlib import Path

def create_sketch(image):
    """
    Convert an image to a pencil sketch style.
    
    Args:
        image: Input color image
        
    Returns:
        Sketch-style version of the input image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image
    inverted = 255 - gray
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    
    # Invert the blurred image
    inverted_blurred = 255 - blurred
    
    # Create the sketch by dividing the grayscale image by the inverted blurred image
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    
    return sketch

def create_sketch_enhanced(image):
    """
    Convert an image to an enhanced pencil sketch style with stronger edge lines.
    
    Args:
        image: Input color image
        
    Returns:
        Enhanced sketch-style image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Invert and blur (classic sketch)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blurred = 255 - blurred
    sketch_basic = cv2.divide(gray, inverted_blurred, scale=256.0)

    # # Step 2: Edge detection (Laplacian or Canny)
    # edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    # edges = cv2.GaussianBlur(edges, (3, 3), 0)
    # edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)[1]  # binary edge mask

    # # Step 3: Blend edges with sketch
    # enhanced = cv2.subtract(sketch_basic, edges // 3)

    # # Optional: contrast boost (clip + gamma)
    # enhanced = np.clip(enhanced * 1.2, 0, 255).astype(np.uint8)

    return sketch_basic

def main():
    # Define directories
    original_dir = Path("dataset/raw")
    sketch_dir = Path("dataset/sketch/cv0")
    
    # Create output directory if it doesn't exist
    os.makedirs(sketch_dir, exist_ok=True)
    
    # Get all image files from the original directory
    image_files = [f for f in os.listdir(original_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images in {original_dir}")
    
    # Process only the first 8 images if there are more
    count = 0
    for image_file in image_files:
        if count >= 8:
            break
            
        input_path = original_dir / image_file
        output_path = sketch_dir / f"sketch_{image_file}"
        
        print(f"Processing {input_path} -> {output_path}")
        
        # Read the image
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"Error: Could not read {input_path}")
            continue
            
        # Create sketch version
        sketch_image = create_sketch_enhanced(image)
        
        # Save the output image
        cv2.imwrite(str(output_path), sketch_image)
        
        count += 1
    
    print(f"Generated {count} sketch images in {sketch_dir}")

if __name__ == "__main__":
    main()