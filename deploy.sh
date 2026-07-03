#!/bin/bash

echo "🚀 尿酸浓度预测Android应用部署脚本"
echo "=================================="

# 检查模型文件是否存在
if [ ! -f "best_uric_acid_model.pth" ]; then
    echo "❌ 错误：找不到 best_uric_acid_model.pth 文件"
    echo "请确保模型文件在当前目录中"
    exit 1
fi

echo "✅ 找到模型文件: best_uric_acid_model.pth"

# 转换模型
echo "🔄 开始转换模型..."
python convert_model_to_mobile.py

if [ $? -ne 0 ]; then
    echo "❌ 模型转换失败"
    exit 1
fi

# 创建assets目录
echo "📁 创建assets目录..."
mkdir -p app/src/main/assets

# 复制模型文件
echo "📋 复制模型文件到assets目录..."
cp uric_acid_model_mobile_optimized.ptl app/src/main/assets/

if [ ! -f "app/src/main/assets/uric_acid_model_mobile_optimized.ptl" ]; then
    echo "❌ 模型文件复制失败"
    exit 1
fi

echo "✅ 模型文件已复制到: app/src/main/assets/uric_acid_model_mobile_optimized.ptl"

# 检查文件大小
file_size=$(ls -lh app/src/main/assets/uric_acid_model_mobile_optimized.ptl | awk '{print $5}')
echo "📊 模型文件大小: $file_size"

echo ""
echo "🎉 部署完成！"
echo "=================================="
echo "下一步操作："
echo "1. 在Android Studio中打开项目"
echo "2. 点击 'Sync Project with Gradle Files'"
echo "3. 连接Android设备或启动模拟器"
echo "4. 点击运行按钮部署应用"
echo ""
echo "📱 应用功能："
echo "- 从相册选择图片"
echo "- 使用深度学习模型预测尿酸浓度"
echo "- 显示详细预测结果"
echo ""
echo "🔧 如果遇到问题，请查看 ERROR_TROUBLESHOOTING.md"
