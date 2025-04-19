import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights

class VGGPerceptualLoss(nn.Module):
    """
    VGG16-based perceptual loss
    """
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        
        # Load VGG model with trainable parameters set to False
        try:
            # Use torchvision's VGG implementation
            self.vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features
            self.has_vgg = True
            
            # Freeze VGG parameters
            for param in self.vgg.parameters():
                param.requires_grad = False
                
            # Move to appropriate device
            self.vgg.eval()  # Set to evaluation mode
            
            # Define feature layers
            self.feature_layers = ['3', '8', '15', '22']  # Corresponding to specific relu layers
        except Exception as e:
            print(f"Error loading VGG model: {e}")
            self.has_vgg = False
            # Simple fallback network
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Register buffers for normalization to move with model to correct device
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x, y):
        # Make sure inputs have gradients if needed for training
        if not x.requires_grad:
            print("Warning: Input x to VGGPerceptualLoss doesn't require gradients")
        
        # Normalize to [0,1] range - using inplace operations can break gradient flow
        x_normalized = (x.clone() + 1.0) / 2.0
        y_normalized = (y.clone() + 1.0) / 2.0
        
        # Apply ImageNet normalization
        x_vgg = (x_normalized - self.mean) / self.std
        y_vgg = (y_normalized - self.mean) / self.std
        
        # Calculate perceptual loss
        if self.has_vgg:
            loss = 0.0
            
            # Extract features at different layers
            for i, layer in enumerate(self.vgg):
                x_vgg = layer(x_vgg)
                y_vgg = layer(y_vgg)
                
                # Add loss at desired feature layers
                if str(i) in self.feature_layers:
                    loss += F.mse_loss(x_vgg, y_vgg)
        else:
            # Use fallback if VGG isn't available
            x1 = F.relu(self.conv1(x_vgg))
            y1 = F.relu(self.conv1(y_vgg))
            loss = F.mse_loss(x1, y1)
            
            x2 = F.relu(self.conv2(x1))
            y2 = F.relu(self.conv2(y1))
            loss += F.mse_loss(x2, y2)
            
            x3 = F.relu(self.conv3(x2))
            y3 = F.relu(self.conv3(y2))
            loss += F.mse_loss(x3, y3)
        
        return loss
    
class EnhancedStyleLoss(nn.Module):
    def __init__(self):
        super(EnhancedStyleLoss, self).__init__()
        # Keep your existing VGG setup
        try:
            vgg = models.vgg19(weights=VGG16_Weights.DEFAULT).features
            self.has_vgg = True
            self.vgg = vgg
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        except:
            self.has_vgg = False
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    
    def _gram_loss(self, x, y):
        # Your existing Gram matrix code
        b, c, h, w = x.size()
        x_flat = x.view(b, c, -1)
        y_flat = y.view(b, c, -1)
        
        x_gram = torch.bmm(x_flat, x_flat.transpose(1, 2)) / (c * h * w)
        y_gram = torch.bmm(y_flat, y_flat.transpose(1, 2)) / (c * h * w)
        
        return F.mse_loss(x_gram, y_gram)
    
    def _color_histogram_loss(self, x, y, bins=64):
        # New color histogram comparison
        loss = 0
        for i in range(3):  # For each RGB channel
            x_hist = torch.histc(x[:,i,:,:].flatten(), bins=bins, min=0, max=1)
            y_hist = torch.histc(y[:,i,:,:].flatten(), bins=bins, min=0, max=1)
            
            # Normalize histograms
            x_hist = x_hist / (x_hist.sum() + 1e-10)
            y_hist = y_hist / (y_hist.sum() + 1e-10)
            
            # Compare histograms (Wasserstein distance)
            loss += torch.abs(torch.cumsum(x_hist, dim=0) - torch.cumsum(y_hist, dim=0)).sum()
        
        return loss / 3.0
    
    def _gradient_consistency_loss(self, x):
        # New gradient consistency component
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        
        grad_loss = 0
        for c in range(3):
            channel = x[:,c:c+1,:,:]
            grad_x = F.conv2d(channel, sobel_x, padding=1)
            grad_y = F.conv2d(channel, sobel_y, padding=1)
            
            # Encourage small but non-zero gradients (avoiding banding)
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
            # Penalize both very large and very small gradients
            grad_loss += torch.mean((1.0 - torch.exp(-grad_mag * 3)) * torch.exp(-grad_mag * 3))
        
        return grad_loss / 3.0
    
    def forward(self, x, y, include_extras=True):
        # Convert from [-1,1] to [0,1]
        x = (x + 1) / 2
        y = (y + 1) / 2
        
        # Original style loss (Gram matrix-based)
        style_loss = 0.0
        
        if self.has_vgg:
            # Your existing VGG feature extraction
            x_features = []
            y_features = []
            for i, layer in enumerate(self.vgg[:10]):
                x = layer(x)
                y = layer(y)
                if i in [1, 3, 6, 8]:
                    x_features.append(x)
                    y_features.append(y)
        else:
            # Your existing fallback
            x1 = F.relu(self.conv1(x))
            y1 = F.relu(self.conv1(y))
            x2 = F.relu(self.conv2(x1))
            y2 = F.relu(self.conv2(y1))
            x_features = [x1, x2]
            y_features = [y1, y2]
        
        # Original Gram matrix loss
        for xf, yf in zip(x_features, y_features):
            style_loss += self._gram_loss(xf, yf)
        
        # If we want the enhanced version, add the new components
        if include_extras:
            color_loss = self._color_histogram_loss(x, y) * 0.5
            gradient_loss = self._gradient_consistency_loss(x) * 0.2
            return style_loss + color_loss + gradient_loss
        else:
            # Original behavior
            return style_loss
        
class CombinedLoss(nn.Module):
    """
    Combined loss: L1 loss + VGG perceptual loss
    Fixed to preserve gradients
    """
    def __init__(self, lambda_l1=1.0, lambda_perceptual=0.5):
        super(CombinedLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        
    def forward(self, output, target):
        # Ensure output has gradients
        if not output.requires_grad:
            print("Warning: Output to CombinedLoss doesn't require gradients - enabling gradients")
            output.requires_grad_(True)
        
        # Compute L1 loss
        l1 = self.l1_loss(output, target)
        
        # Compute perceptual loss
        perceptual = self.perceptual_loss(output, target)
        
        # Calculate total loss
        total_loss = self.lambda_l1 * l1 + self.lambda_perceptual * perceptual
        
        # Return loss value and components dictionary
        return total_loss, {
            'l1': l1.item(),
            'perceptual': perceptual.item(),
            'total': total_loss.item()
        }