import numpy as np
import cv2
from PIL import Image, ImageEnhance

def remove_horizontal_lines(image, strength=0.8):
    """
    Simple but effective function to remove horizontal line artifacts
    
    Args:
        image: PIL Image or numpy array
        strength: Filter strength (0-1)
        
    Returns:
        Processed image
    """
    import numpy as np
    import cv2
    from PIL import Image, ImageEnhance
    
    # Check input type
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        # Save original size
        original_size = image.size
        # Convert to numpy array
        img_np = np.array(image)
        # Check if normalization is needed
        is_uint8 = img_np.dtype == np.uint8
    else:
        img_np = image.copy()
        is_uint8 = img_np.max() > 1.0
        if is_uint8:
            img_np = img_np.astype(np.float32) / 255.0
    
    # Ensure we're working with float
    if is_uint8:
        img_float = img_np.astype(np.float32) / 255.0
    else:
        img_float = img_np.astype(np.float32)
    
    # Convert to grayscale if needed for processing
    if len(img_float.shape) == 3:
        # Process each channel separately
        result = np.zeros_like(img_float)
        
        for c in range(img_float.shape[2]):
            channel = img_float[:, :, c]
            
            # Step 1: Apply horizontal median filter
            # Create a horizontal kernel for median blur
            ksize = 5  # Must be odd
            horizontal_median = cv2.medianBlur(channel, ksize)
            
            # Step 2: Apply bilateral filter to preserve edges while smoothing
            bilateral = cv2.bilateralFilter(channel, d=5, sigmaColor=0.1, sigmaSpace=5)
            
            # Step 3: Apply directed filter to reduce horizontal lines
            # Create a horizontal kernel
            kernel_len = 9
            kernel = np.zeros((kernel_len, kernel_len))
            # Set middle row to 1s
            kernel[kernel_len//2, :] = np.ones(kernel_len) / kernel_len
            # Apply the filter
            horizontal_filtered = cv2.filter2D(channel, -1, kernel)
            
            # Step 4: Blend the filters based on strength parameter
            blended = channel * (1 - strength) + \
                      (horizontal_median * 0.4 + bilateral * 0.3 + horizontal_filtered * 0.3) * strength
            
            # Store result
            result[:, :, c] = blended
    else:
        # If grayscale
        # Step 1: Apply horizontal median filter
        ksize = 5  # Must be odd
        horizontal_median = cv2.medianBlur(img_float, ksize)
        
        # Step 2: Apply bilateral filter
        bilateral = cv2.bilateralFilter(img_float, d=5, sigmaColor=0.1, sigmaSpace=5)
        
        # Step 3: Apply directed filter
        kernel_len = 9
        kernel = np.zeros((kernel_len, kernel_len))
        kernel[kernel_len//2, :] = np.ones(kernel_len) / kernel_len
        horizontal_filtered = cv2.filter2D(img_float, -1, kernel)
        
        # Step 4: Blend the filters
        result = img_float * (1 - strength) + \
                (horizontal_median * 0.4 + bilateral * 0.3 + horizontal_filtered * 0.3) * strength
    
    # Convert back to the original format
    if is_pil:
        # Convert back to uint8
        result_uint8 = (result * 255).astype(np.uint8)
        # Convert back to PIL
        result_pil = Image.fromarray(result_uint8)
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(result_pil)
        result_pil = enhancer.enhance(1.1)
        # Return
        return result_pil
    else:
        # Return numpy array in the original range
        if is_uint8:
            return (result * 255).astype(np.uint8)
        return result