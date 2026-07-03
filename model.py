import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on circular ROI"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv1(x_concat))
        return x * attention

class ColorSensitiveBlock(nn.Module):
    """Specialized block for capturing subtle color variations"""
    def __init__(self, in_channels, out_channels):
        super(ColorSensitiveBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels//4, 1)
        self.conv3x3 = nn.Conv2d(out_channels//4, out_channels//2, 3, padding=1)
        self.conv5x5 = nn.Conv2d(out_channels//4, out_channels//4, 5, padding=2)
        self.conv1x1_final = nn.Conv2d(out_channels//4 + out_channels//2, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x1)
        x3 = self.conv5x5(x1)
        x_concat = torch.cat([x2, x3], dim=1)
        out = self.conv1x1_final(x_concat)
        out = self.bn(out)
        return self.relu(out)

class UricAcidRegressor(nn.Module):
    """
    Specialized CNN for uric acid concentration regression from colorimetric images
    """
    def __init__(self, num_classes=1, dropout_rate=0.3):
        super(UricAcidRegressor, self).__init__()
        
        # Use EfficientNet-B0 as backbone for efficient feature extraction
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Remove the classifier
        self.backbone.classifier = nn.Identity()
        
        # Get feature dimensions
        backbone_features = 1280  # EfficientNet-B0 output features
        
        # Color-sensitive preprocessing layers
        self.color_enhance = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ColorSensitiveBlock(16, 32),
            nn.MaxPool2d(2),
            ColorSensitiveBlock(32, 64),
            nn.MaxPool2d(2)
        )
        
        # Spatial attention for ROI focus
        self.spatial_attention = SpatialAttention()
        
        # Additional feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Conv2d(backbone_features, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Color enhancement features processing
        self.color_features = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Regression head with multiple paths
        self.regressor = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Color enhancement path
        color_features = self.color_enhance(x)
        color_features = self.spatial_attention(color_features)
        color_features = self.color_features(color_features)
        color_features = color_features.view(color_features.size(0), -1)
        
        # Main backbone path
        backbone_features = self.backbone.features(x)
        backbone_features = self.spatial_attention(backbone_features)
        backbone_features = self.feature_processor(backbone_features)
        backbone_features = backbone_features.view(backbone_features.size(0), -1)
        
        # Combine features
        combined_features = torch.cat([backbone_features, color_features], dim=1)
        
        # Regression prediction
        output = self.regressor(combined_features)
        
        return output

# Model factory function
def create_model(num_classes=1, dropout_rate=0.3):
    """
    Create and return the uric acid regression model
    
    Args:
        num_classes (int): Number of output classes (1 for regression)
        dropout_rate (float): Dropout rate for regularization
    
    Returns:
        UricAcidRegressor: The model instance
    """
    model = UricAcidRegressor(num_classes=num_classes, dropout_rate=dropout_rate)
    return model

if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_model()
    test_input = torch.randn(2, 3, 430, 430)  # Batch of 2 images
    output = model(test_input)
    print(f"Model output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")