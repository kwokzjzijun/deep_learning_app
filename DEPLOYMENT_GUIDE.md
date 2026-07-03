# PyTorch深度学习模型Android部署指南

## 项目概述
本项目实现了将Python训练的PyTorch尿酸浓度预测模型部署到Android应用的完整流程。

## 实现功能
1. ✅ 从相册选择图片并显示
2. ✅ 使用PyTorch Mobile进行模型推理
3. ✅ 实时显示预测结果
4. ✅ 完整的权限管理（支持Android 13+）

## 部署步骤

### 步骤1：转换PyTorch模型
在项目根目录运行Python脚本将.pth模型转换为移动端格式：

```bash
python convert_model_to_mobile.py
```

这将生成两个文件：
- `uric_acid_model_mobile.pt` - 标准TorchScript模型
- `uric_acid_model_mobile_optimized.ptl` - 优化的移动端模型（推荐使用）

### 步骤2：将模型文件添加到Android项目
1. 在 `app/src/main/` 目录下创建 `assets` 文件夹
2. 将 `uric_acid_model_mobile_optimized.ptl` 复制到 `assets` 文件夹中

```bash
mkdir -p app/src/main/assets
cp uric_acid_model_mobile_optimized.ptl app/src/main/assets/
```

### 步骤3：同步Gradle依赖
在Android Studio中：
1. 点击 "Sync Project with Gradle Files" 按钮
2. 等待PyTorch Mobile依赖下载完成

### 步骤4：运行应用
1. 连接Android设备或启动模拟器
2. 点击运行按钮部署应用

## 文件结构说明

### 核心文件
- **MainActivity.java** - 主界面逻辑，处理用户交互
- **UricAcidProcessor.java** - 图像处理和模型推理核心类
- **activity_main.xml** - 主界面布局
- **AndroidManifest.xml** - 应用配置和权限声明
- **build.gradle.kts** - 项目依赖配置

### 关键修改
1. **布局文件** (`activity_main.xml`)
   - 纵向布局设计
   - ImageView显示选中图片
   - TextView显示预测结果
   - 两个操作按钮（选择图片、开始处理）

2. **权限配置** (`AndroidManifest.xml`)
   - Android 13以下：READ_EXTERNAL_STORAGE
   - Android 13及以上：READ_MEDIA_IMAGES

3. **依赖配置** (`build.gradle.kts`)
   ```kotlin
   implementation("org.pytorch:pytorch_android_lite:1.13.1")
   implementation("org.pytorch:pytorch_android_torchvision_lite:1.13.1")
   ```

## 模型处理流程

### 1. 图像预处理
- 调整大小：224x224像素
- 归一化：使用ImageNet标准（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）

### 2. 模型推理
- 加载优化的PyTorch Mobile模型
- 执行前向传播
- 获取预测的尿酸浓度值（ppm）

### 3. 结果展示
- 显示预测浓度值
- 根据浓度范围给出状态提示
- 显示模型准确率范围

## 注意事项

### 性能优化
- 使用优化的模型文件（.ptl）而非标准模型（.pt）
- 在后台线程执行推理避免UI阻塞
- 图像预处理使用固定尺寸减少计算量

### 兼容性
- 最低Android版本：API 27 (Android 8.1)
- 目标Android版本：API 34 (Android 14)
- 支持ARM和x86架构

### 调试建议
1. 检查模型文件是否正确放置在assets文件夹
2. 确认权限是否正确授予
3. 查看Logcat中的"UricAcidProcessor"标签获取调试信息

## 可能的问题及解决方案

### 问题1：模型加载失败
**解决方案**：
- 确认模型文件存在于 `app/src/main/assets/` 目录
- 检查模型文件名是否正确
- 验证模型转换是否成功

### 问题2：预测结果异常
**解决方案**：
- 确认图像预处理参数与训练时一致
- 检查模型输入维度是否正确（1x3x224x224）
- 验证归一化参数是否正确

### 问题3：权限请求失败
**解决方案**：
- 在设置中手动授予应用存储权限
- 确认AndroidManifest.xml中声明了正确的权限
- 针对不同Android版本使用相应的权限

## 扩展功能建议
1. 添加相机拍照功能
2. 实现批量图片处理
3. 添加历史记录功能
4. 实现结果分享功能
5. 添加图表可视化展示

## 技术栈
- **Android开发**：Java, Android SDK
- **深度学习框架**：PyTorch Mobile
- **图像处理**：TensorImageUtils
- **UI框架**：Material Design Components

## 更新日志
- v1.0 - 初始版本，实现基础预测功能
  - 图片选择和显示
  - 模型推理
  - 结果展示
  - 权限管理
