import os
import argparse
from PIL import Image
import torch
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
import cv2

def enhance_architectural_depth(depth_raw, rgb_image_path, output_dir):
    """
    Apply various enhancements to depth maps to better preserve architectural features
    
    Args:
        depth_raw: Raw depth map as numpy array
        rgb_image_path: Path to the original RGB image
        output_dir: Directory to save enhanced depth maps
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(rgb_image_path))[0]
    
    # Load RGB image for edge detection and to get original dimensions
    rgb_image = cv2.imread(rgb_image_path)
    original_height, original_width = rgb_image.shape[:2]
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    
    # 1. Basic normalization
    depth_normalized = (depth_raw - np.min(depth_raw)) / (np.max(depth_raw) - np.min(depth_raw))
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    norm_output = os.path.join(output_dir, f"{base_name}_depth_normalized.png")
    cv2.imwrite(norm_output, depth_uint8)
    print(f"[✓] Normalized depth map saved to {norm_output}")
    
    # 2. Edge-aware bilateral filtering
    depth_bilateral = cv2.bilateralFilter(depth_uint8, d=9, sigmaColor=75, sigmaSpace=75)
    bilateral_output = os.path.join(output_dir, f"{base_name}_depth_bilateral.png")
    cv2.imwrite(bilateral_output, depth_bilateral)
    print(f"[✓] Bilateral filtered depth map saved to {bilateral_output}")
    
    # 3. Extract edges from RGB image for depth enhancement
    edges = cv2.Canny(gray_image, 100, 200)
    edges_output = os.path.join(output_dir, f"{base_name}_edges.png")
    cv2.imwrite(edges_output, edges)
    
    # 4. Dilate edges to ensure they influence depth map
    kernel = np.ones((3,3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 5. Edge-enhanced depth map (preserves architectural boundaries)
    # Combine depth with edges to preserve structural features
    depth_with_edges = cv2.addWeighted(depth_bilateral, 0.85, dilated_edges, 0.15, 0)
    edge_enhanced_output = os.path.join(output_dir, f"{base_name}_depth_edge_enhanced.png")
    cv2.imwrite(edge_enhanced_output, depth_with_edges)
    print(f"[✓] Edge-enhanced depth map saved to {edge_enhanced_output}")
    
    # 6. Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    depth_clahe = clahe.apply(depth_uint8)
    clahe_output = os.path.join(output_dir, f"{base_name}_depth_clahe.png")
    cv2.imwrite(clahe_output, depth_clahe)
    print(f"[✓] CLAHE enhanced depth map saved to {clahe_output}")
    
    # 7. Try to use guided filter if available
    try:
        from cv2.ximgproc import guidedFilter
        depth_guided = guidedFilter(
            guide=gray_image, 
            src=depth_bilateral, 
            radius=16, 
            eps=100
        )
        guided_output = os.path.join(output_dir, f"{base_name}_depth_guided.png")
        cv2.imwrite(guided_output, depth_guided)
        print(f"[✓] Guided filtered depth map saved to {guided_output}")
    except Exception as e:
        print(f"Warning: Guided filter not available - {e}")
    
    # 8. Multi-scale detail enhancement
    # Generate depth maps at multiple scales and combine to preserve both coarse structure and fine details
    scales = [0.5, 0.75, 1.0]
    depth_maps = []
    
    for scale in scales:
        if scale != 1.0:
            h, w = depth_uint8.shape
            new_h, new_w = int(h * scale), int(w * scale)
            # Downsample
            depth_small = cv2.resize(depth_uint8, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Apply bilateral filter at this scale
            depth_small_bilateral = cv2.bilateralFilter(depth_small, d=9, sigmaColor=75, sigmaSpace=75)
            # Upsample back to original size
            depth_upscaled = cv2.resize(depth_small_bilateral, (w, h), interpolation=cv2.INTER_LINEAR)
            depth_maps.append(depth_upscaled)
        else:
            depth_maps.append(depth_bilateral)
    
    # Combine multi-scale maps
    depth_multiscale = np.zeros_like(depth_uint8, dtype=np.float32)
    weights = [0.25, 0.35, 0.4]  # More weight to fine details
    
    for i, (d_map, weight) in enumerate(zip(depth_maps, weights)):
        depth_multiscale += d_map * weight
    
    depth_multiscale = depth_multiscale.astype(np.uint8)
    multiscale_output = os.path.join(output_dir, f"{base_name}_depth_multiscale.png")
    cv2.imwrite(multiscale_output, depth_multiscale)
    print(f"[✓] Multi-scale enhanced depth map saved to {multiscale_output}")
    
    # 9. Structure-preserving normalization by emphasizing local gradients
    # Compute gradient magnitude of depth map
    sobelx = cv2.Sobel(depth_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(depth_uint8, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = cv2.magnitude(sobelx, sobely)
    
    # Normalize gradient magnitude
    gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Combine original depth with gradient information
    depth_gradient = cv2.addWeighted(depth_bilateral, 0.7, gradient_mag, 0.3, 0)
    gradient_output = os.path.join(output_dir, f"{base_name}_depth_gradient.png")
    cv2.imwrite(gradient_output, depth_gradient)
    print(f"[✓] Gradient-enhanced depth map saved to {gradient_output}")
    
    # 10. Create hybrid enhancement combining edge and multi-scale approaches
    # This works well for architectural features
    hybrid_depth = cv2.addWeighted(depth_multiscale, 0.6, depth_with_edges, 0.4, 0)
    hybrid_output = os.path.join(output_dir, f"{base_name}_depth_hybrid.png")
    cv2.imwrite(hybrid_output, hybrid_depth)
    print(f"[✓] Hybrid enhanced depth map saved to {hybrid_output}")
    
    # 11. Create colored visualizations with different colormaps
    # MODIFIED: Use OpenCV to create colored visualizations instead of matplotlib
    # to ensure the exact same dimensions as the original image
    colormaps = ['turbo', 'plasma', 'viridis']
    for cmap_name in colormaps:
        # Create a colored depth map using OpenCV's COLORMAP
        if cmap_name == 'turbo':
            cmap = cv2.COLORMAP_TURBO
        elif cmap_name == 'plasma':
            cmap = cv2.COLORMAP_PLASMA
        elif cmap_name == 'viridis':
            cmap = cv2.COLORMAP_VIRIDIS
        else:
            cmap = cv2.COLORMAP_JET
            
        # Apply colormap directly to hybrid_depth
        colored_depth = cv2.applyColorMap(hybrid_depth, cmap)
        
        # Make sure dimensions match the original image (resize if needed)
        if colored_depth.shape[:2] != (original_height, original_width):
            colored_depth = cv2.resize(colored_depth, (original_width, original_height), 
                                      interpolation=cv2.INTER_LINEAR)
        
        # Save the colored depth map
        colored_output = os.path.join(output_dir, f"{base_name}_depth_{cmap_name}.png")
        cv2.imwrite(colored_output, colored_depth)
        print(f"[✓] {cmap_name.capitalize()} colored depth map saved to {colored_output}")
    
    # Return the best enhanced depth map for further processing
    return hybrid_depth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="jcsmr.jpg", help="Path to input image")
    parser.add_argument("--depth", type=str, default=None, help="Path to existing depth map (optional)")
    parser.add_argument("--output-dir", type=str, default="depth/depth-anything-large-hf_6", help="Output directory for enhanced depth maps")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or generate depth map
    if args.depth and os.path.exists(args.depth):
        print(f"Loading existing depth map from {args.depth}")
        if args.depth.endswith('.npy'):
            depth_raw = np.load(args.depth)
        else:
            depth_raw = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
            if len(depth_raw.shape) > 2:  # Convert to grayscale if needed
                depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
    else:
        print("Generating new depth map...")
        # Initialize depth estimator
        depth_estimator = DepthEstimator(device="cuda" if torch.cuda.is_available() else "cpu")
        depth_raw = depth_estimator.estimate_depth(args.image)
        
        # Save raw depth data as NumPy file
        raw_output = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.image))[0]}_depth_raw.npy")
        np.save(raw_output, depth_raw)
        print(f"[✓] Raw depth data saved to {raw_output}")
    
    # Apply architectural enhancements
    enhanced_depth = enhance_architectural_depth(depth_raw, args.image, args.output_dir)
    
    print("\nDepth enhancement completed! The recommended depth map for your architectural image is:")
    print(f"{os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.image))[0])}_depth_hybrid.png")
    print("\nYou can use this enhanced depth map for your style transfer or other downstream tasks.")

# Depth estimator class as in your original code
class DepthEstimator:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = None
        
    def load_model(self):
        print("Using device:", self.device)
        self.model = pipeline(
            task="depth-estimation",
            model="LiheYoung/depth-anything-large-hf",
            device=0 if self.device == "cuda" else -1
        )
        
    def estimate_depth(self, image_path):
        if self.model is None:
            self.load_model()
            
        # Load image
        print(f"Processing image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        
        # Get depth map
        output = self.model(image)
        depth = output["depth"]
        
        # Convert to numpy array
        depth_np = np.array(depth)
        
        # Print depth map statistics
        print(f"Depth map shape: {depth_np.shape}")
        print(f"Depth range: min={np.min(depth_np):.4f}, max={np.max(depth_np):.4f}")
        
        return depth_np

if __name__ == "__main__":
    main()