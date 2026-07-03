# 错误排查指南 - 快速解决运行问题

## 🔴 运行前检查清单
```bash
# 1. 检查Python环境
python --version  # 需要Python 3.7+

# 2. 检查PyTorch
python -c "import torch; print(torch.__version__)"

# 3. 检查模型文件
ls -la best_uric_acid_model.pth

# 4. 检查model.py
grep "create_model" model.py
```

## 🟡 模型转换阶段错误

### 错误：ImportError: cannot import name 'create_model'
```python
# 临时解决方案 - 直接在convert_model_to_mobile.py中定义模型
import torch
import torch.nn as nn
from torchvision import models

def create_model(num_classes=1, dropout_rate=0.5):
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, num_classes)
    )
    return model
```

### 错误：RuntimeError: Error loading state dict
```python
# 检查模型结构是否匹配
# 在convert_model_to_mobile.py中添加
try:
    model.load_state_dict(torch.load('best_uric_acid_model.pth', map_location='cpu'))
except:
    # 尝试只加载部分权重
    state_dict = torch.load('best_uric_acid_model.pth', map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
```

### 错误：ModuleNotFoundError: No module named 'torch.utils.mobile_optimizer'
```bash
# 安装必要的包
pip install torch torchvision

# 如果还是失败，使用简化版本（不优化）
# 注释掉优化相关代码，只保存.pt文件
```

## 🟠 Android Studio阶段错误

### 错误：Gradle sync failed
```gradle
// 在 app/build.gradle.kts 中添加仓库
repositories {
    google()
    mavenCentral()
    maven { url = uri("https://maven.aliyun.com/repository/public") }  // 添加阿里云镜像
}
```

### 错误：Could not find pytorch_android_lite
```gradle
// 使用旧版本或替代版本
implementation("org.pytorch:pytorch_android:1.12.2")
implementation("org.pytorch:pytorch_android_torchvision:1.12.2")
```

### 错误：Duplicate class found
```gradle
// 在 gradle.properties 添加
android.useAndroidX=true
android.enableJetifier=true
```

## 🔵 运行时错误

### 错误：java.io.FileNotFoundException: uric_acid_model_mobile_optimized.ptl
```java
// 在 UricAcidProcessor.java 中修改
// 尝试不同的文件名
String[] possibleNames = {
    "uric_acid_model_mobile_optimized.ptl",
    "uric_acid_model_mobile.pt",
    "model.pt"
};

for (String name : possibleNames) {
    try {
        String modelPath = assetFilePath(context, name);
        module = LiteModuleLoader.load(modelPath);
        break;
    } catch (Exception e) {
        continue;
    }
}
```

### 错误：RuntimeError: forward() Expected 4D tensor
```java
// 确保输入维度正确
// 在 UricAcidProcessor.java 中检查
Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
    bitmap,
    NORM_MEAN,
    NORM_STD
);

// 添加调试信息
Log.d(TAG, "Tensor shape: " + Arrays.toString(inputTensor.shape()));
```

### 错误：java.lang.UnsatisfiedLinkError
```xml
<!-- 在 AndroidManifest.xml 的 application 标签中添加 -->
<application
    android:extractNativeLibs="true"
    ...>
```

## 🟢 快速修复脚本

创建 `quick_fix.sh`：
```bash
#!/bin/bash

echo "🔧 开始快速修复..."

# 1. 创建assets目录
mkdir -p app/src/main/assets

# 2. 尝试转换模型
if [ -f "best_uric_acid_model.pth" ]; then
    echo "✅ 找到模型文件"
    python convert_model_to_mobile.py
    
    # 3. 复制模型文件
    for file in *.pt *.ptl; do
        if [ -f "$file" ]; then
            cp "$file" app/src/main/assets/
            echo "✅ 已复制 $file"
        fi
    done
else
    echo "❌ 未找到 best_uric_acid_model.pth"
fi

# 4. 修复权限
chmod +x gradlew

echo "🎉 修复完成！"
```

## 💡 调试技巧

### 1. 启用详细日志
```java
// 在 MainActivity.java onCreate() 中添加
if (BuildConfig.DEBUG) {
    Log.d("MainActivity", "Debug mode enabled");
}
```

### 2. 查看Logcat
```bash
# 过滤相关日志
adb logcat | grep -E "UricAcidProcessor|MainActivity"
```

### 3. 测试模型加载
```java
// 添加测试方法
private void testModelLoading() {
    try {
        // 尝试加载空输入
        float[] dummyInput = new float[1 * 3 * 224 * 224];
        Tensor testTensor = Tensor.fromBlob(dummyInput, new long[]{1, 3, 224, 224});
        Log.d(TAG, "测试输入创建成功");
    } catch (Exception e) {
        Log.e(TAG, "测试失败: " + e.getMessage());
    }
}
```

## 📞 紧急联系方式

如果以上方法都无法解决，请：
1. 截图完整的错误信息
2. 提供 `adb logcat` 输出
3. 说明执行了哪些步骤
4. 检查Android Studio版本（推荐使用最新稳定版）

## 备用方案

如果PyTorch Mobile始终无法工作，考虑：
1. 使用TensorFlow Lite替代
2. 使用ONNX Runtime Mobile
3. 将推理移到服务器端（API方式）

---
**记住**：大部分问题都是因为文件路径或依赖版本不匹配导致的！
