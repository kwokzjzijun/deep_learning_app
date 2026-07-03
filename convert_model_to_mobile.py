import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# 纯PyTorch实现的EfficientNet-B0（不依赖torchvision）
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConv, self).__init__()
        self.stride = stride
        self.use_residual = in_channels == out_channels and stride == 1
        
        hidden_dim = int(round(in_channels * expand_ratio))
        
        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        else:
            self.expand_conv = nn.Identity()
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride, 
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        # Pointwise convolution
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        result = self.expand_conv(x)
        result = self.depthwise_conv(result)
        result = self.pointwise_conv(result)
        
        if self.use_residual:
            result = result + x
            
        return result

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetB0, self).__init__()
        
        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            
            # Stage 1
            MBConv(32, 16, 3, 1, 1),
            
            # Stage 2
            MBConv(16, 24, 3, 2, 6),
            MBConv(24, 24, 3, 1, 6),
            
            # Stage 3
            MBConv(24, 40, 5, 2, 6),
            MBConv(40, 40, 5, 1, 6),
            
            # Stage 4
            MBConv(40, 80, 3, 2, 6),
            MBConv(80, 80, 3, 1, 6),
            MBConv(80, 80, 3, 1, 6),
            
            # Stage 5
            MBConv(80, 112, 5, 1, 6),
            MBConv(112, 112, 5, 1, 6),
            MBConv(112, 112, 5, 1, 6),
            
            # Stage 6
            MBConv(112, 192, 5, 2, 6),
            MBConv(192, 192, 5, 1, 6),
            MBConv(192, 192, 5, 1, 6),
            MBConv(192, 192, 5, 1, 6),
            
            # Stage 7
            MBConv(192, 320, 3, 1, 6),
            
            # Final convolution
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Linear(1280, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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
    """Specialized CNN for uric acid concentration regression"""
    def __init__(self, num_classes=1, dropout_rate=0.3):
        super(UricAcidRegressor, self).__init__()
        
        # 使用EfficientNet-B0作为backbone
        self.backbone = EfficientNetB0(num_classes=1280)
        self.backbone.classifier = nn.Identity()  # 移除最后的分类层
        
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
            nn.Conv2d(1280, 512, 3, padding=1),
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

def create_model(num_classes=1, dropout_rate=0.3):
    """Create and return the uric acid regression model"""
    model = UricAcidRegressor(num_classes=num_classes, dropout_rate=dropout_rate)
    return model

def convert_model_to_mobile():
    """将PyTorch模型转换为PyTorch Mobile格式"""
    
    print("开始模型转换...")
    
    # 检查模型文件是否存在
    model_path = 'best_uric_acid_model.pth'
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        print("请确保 best_uric_acid_model.pth 文件在当前目录中")
        return
    
    try:
        # 创建模型实例
        print("创建模型实例...")
        model = create_model(num_classes=1, dropout_rate=0.3)
        
        # 加载训练好的权重
        print("加载模型权重...")
        state_dict = torch.load(model_path, map_location='cpu')
        
        # 尝试加载权重，如果结构不匹配则使用strict=False
        try:
            model.load_state_dict(state_dict)
            print("✅ 模型权重加载成功")
        except Exception as e:
            print(f"⚠️ 模型结构不完全匹配，尝试部分加载: {e}")
            model.load_state_dict(state_dict, strict=False)
            print("✅ 模型权重部分加载成功")
        
        model.eval()
        
        # 创建示例输入（用于追踪）
        print("创建示例输入...")
        example_input = torch.rand(1, 3, 224, 224)
        
        # 测试前向传播
        print("测试模型前向传播...")
        with torch.no_grad():
            test_output = model(example_input)
            print(f"✅ 测试输出形状: {test_output.shape}")
        
        # 使用TorchScript追踪模型
        print("开始TorchScript追踪...")
        traced_script_module = torch.jit.trace(model, example_input)
        
        # 保存为mobile格式
        traced_script_module.save("uric_acid_model_mobile.pt")
        print("✅ 标准TorchScript模型已保存: uric_acid_model_mobile.pt")
        
        # 尝试优化模型用于移动端
        try:
            print("开始移动端优化...")
            from torch.utils.mobile_optimizer import optimize_for_mobile
            optimized_model = optimize_for_mobile(traced_script_module)
            optimized_model._save_for_lite_interpreter("uric_acid_model_mobile_optimized.ptl")
            print("✅ 优化的移动端模型已保存: uric_acid_model_mobile_optimized.ptl")
        except Exception as e:
            print(f"⚠️ 移动端优化失败，使用标准模型: {e}")
            print("✅ 使用标准模型: uric_acid_model_mobile.pt")
        
        print("\n🎉 模型转换完成！")
        print("生成的文件:")
        if os.path.exists("uric_acid_model_mobile.pt"):
            print("- uric_acid_model_mobile.pt: 标准TorchScript模型")
        if os.path.exists("uric_acid_model_mobile_optimized.ptl"):
            print("- uric_acid_model_mobile_optimized.ptl: 优化的移动端模型")
        
    except Exception as e:
        print(f"❌ 模型转换失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert_model_to_mobile()
