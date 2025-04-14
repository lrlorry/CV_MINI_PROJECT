import torch

# Lab颜色空间处理类
class LabColorProcessor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # Pre-computed matrices for conversion
        self.register_buffer = lambda name, tensor: setattr(self, name, tensor)
        
        # sRGB to XYZ matrix (D65 reference white)
        self.register_buffer('rgb_to_xyz_matrix', torch.tensor([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ], device=device))
        
        # XYZ to sRGB matrix
        self.register_buffer('xyz_to_rgb_matrix', torch.tensor([
            [ 3.24045484, -1.53713885, -0.49853154],
            [-0.96926639,  1.87596750,  0.04155593],
            [ 0.05564664, -0.20402591,  1.05722647]
        ], device=device))
        
        # D65 white point
        self.register_buffer('xyz_white', torch.tensor([0.95047, 1.0, 1.08883], device=device).view(1, 3, 1, 1))
    
    def rgb_to_lab(self, rgb):
        """Convert from sRGB to CIE Lab
        Input:
            rgb: tensor in range [0, 1]
        Output:
            lab: L in range [0, 100], a and b in range [-128, 127]
        """
        # Ensure input has correct shape and range
        if rgb.dim() == 3:  # If no batch dimension
            rgb = rgb.unsqueeze(0)
        
        rgb = torch.clamp(rgb, 0.0, 1.0)
        
        # sRGB to linear RGB
        rgb_linear = torch.where(
            rgb > 0.04045,
            ((rgb + 0.055) / 1.055) ** 2.4,
            rgb / 12.92
        )
        
        # RGB to XYZ (matrix multiplication)
        batch_size, channels, height, width = rgb_linear.shape
        rgb_reshaped = rgb_linear.permute(0, 2, 3, 1).reshape(-1, 3)
        xyz = torch.matmul(rgb_reshaped, self.rgb_to_xyz_matrix.t()).reshape(batch_size, height, width, 3).permute(0, 3, 1, 2)
        
        # XYZ to Lab
        # Normalize by white point
        xyz_normalized = xyz / (self.xyz_white + 1e-10)  # Avoid division by zero
        
        # Apply the nonlinear transformation
        f = torch.where(
            xyz_normalized > 0.008856,
            xyz_normalized ** (1/3),
            7.787 * xyz_normalized + 16/116
        )
        
        x, y, z = f[:, 0:1], f[:, 1:2], f[:, 2:3]
        
        # Compute Lab components
        L = (116 * y) - 16
        a = 500 * (x - y)
        b = 200 * (y - z)
        
        # Concatenate channels
        lab = torch.cat([L, a, b], dim=1)
        
        return lab
    
    def lab_to_rgb(self, lab):
        """Convert from CIE Lab to sRGB
        Input:
            lab: L in range [0, 100], a and b in range [-128, 127]
        Output:
            rgb: tensor in range [0, 1]
        """
        # Extract Lab channels
        L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
        
        # Compute f(Y), f(X), f(Z)
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        
        # Convert f to XYZ
        xyz = torch.zeros_like(lab)
        
        # For each component, apply the correct transformation
        # For f > 0.206893
        xyz_from_f = lambda f: torch.where(f > 0.206893, f ** 3, (f - 16/116) / 7.787)
        
        # Apply transformation
        x = xyz_from_f(fx)
        y = xyz_from_f(fy)
        z = xyz_from_f(fz)
        
        # Concatenate and multiply by white point
        xyz = torch.cat([x, y, z], dim=1) * self.xyz_white
        
        # XYZ to RGB (matrix multiplication)
        batch_size, channels, height, width = xyz.shape
        xyz_reshaped = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
        rgb_linear = torch.matmul(xyz_reshaped, self.xyz_to_rgb_matrix.t()).reshape(batch_size, height, width, 3).permute(0, 3, 1, 2)
        
        # Clamp values to avoid negatives
        rgb_linear = torch.clamp(rgb_linear, 0.0, None)
        
        # Linear RGB to sRGB
        rgb = torch.where(
            rgb_linear > 0.0031308,
            1.055 * (rgb_linear ** (1/2.4)) - 0.055,
            12.92 * rgb_linear
        )
        
        # Ensure output is in range [0, 1]
        rgb = torch.clamp(rgb, 0.0, 1.0)
        
        return rgb
    
    def process_ab_channels(self, input_rgb, output_rgb):
        """
        Keep L channel from input image, use a and b channels from output image
        """
        # Ensure inputs are on the right device
        self.device = input_rgb.device
        
        # Convert to [0,1] range for processing
        input_01 = torch.clamp((input_rgb.clone() + 1) / 2.0, 0.0, 1.0)
        output_01 = torch.clamp((output_rgb.clone() + 1) / 2.0, 0.0, 1.0)
        
        try:
            # Convert to Lab
            input_lab = self.rgb_to_lab(input_01)
            output_lab = self.rgb_to_lab(output_01)
            
            # Combine input's L channel and output's a, b channels
            combined_lab = torch.cat([
                input_lab[:, 0:1],  # L channel from input
                output_lab[:, 1:3]  # a and b channels from output
            ], dim=1)
            
            # Convert back to RGB
            combined_rgb = self.lab_to_rgb(combined_lab)
            
            # Convert back to [-1,1] range
            result = combined_rgb * 2.0 - 1.0
            
            return result
            
        except Exception as e:
            print(f"Error in Lab processing: {e}")
            # In case of error, return original output
            return output_rgb