package com.example.deep_learning_app;

import android.content.Context;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.provider.OpenableColumns;
import android.util.Log;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class UricAcidProcessor {
    private static final String TAG = "UricAcidProcessor";
    private static final int MODEL_IMAGE_SIZE = 224;
    private static final int MODEL_CHANNELS = 3;
    private static final String[] MODEL_ASSET_CANDIDATES = new String[]{
            "uric_acid_current_mobile.ptl",
            "best_convnext_2_0425_mobile.ptl",
            "best_aug_again_ema_mobile.ptl",
            "uric_acid_model_mobile_optimized.ptl"
    };
    private Module module;
    private PyObject preprocessModule;
    private String loadedAssetName = "unknown";
    private Context context;
    private PcRoiRepository pcRoiRepository;

    public UricAcidProcessor(Context context) {
        this.context = context;
        loadModel();
        preprocessModule = loadPreprocessModule();
        pcRoiRepository = new PcRoiRepository(context);
    }
    
    /**
     * 加载PyTorch模型
     */
    private void loadModel() {
        IOException lastIoError = null;
        RuntimeException lastRuntimeError = null;
        for (String assetName : MODEL_ASSET_CANDIDATES) {
            try {
                String modelPath = assetFilePath(context, assetName);
                module = LiteModuleLoader.load(modelPath);
                loadedAssetName = assetName;
                Log.d(TAG, "模型加载成功: " + assetName);
                return;
            } catch (IOException e) {
                lastIoError = e;
            } catch (RuntimeException e) {
                lastRuntimeError = e;
                Log.w(TAG, "模型加载失败(" + assetName + "): " + e.getMessage());
            }
        }
        if (lastRuntimeError != null) {
            Log.e(TAG, "模型加载失败: Lite运行时或模型不兼容", lastRuntimeError);
        } else if (lastIoError != null) {
            Log.e(TAG, "模型加载失败: " + lastIoError.getMessage(), lastIoError);
        } else {
            Log.e(TAG, "模型加载失败: 无可用模型或模型格式不兼容");
        }
    }

    private PyObject loadPreprocessModule() {
        try {
            if (!Python.isStarted()) {
                Python.start(new AndroidPlatform(context));
            }
            PyObject module = Python.getInstance().getModule("uric_preprocess");
            Log.i(TAG, "Uric acid PIL preprocessing module loaded");
            return module;
        } catch (RuntimeException e) {
            Log.e(TAG, "Uric acid preprocessing module unavailable", e);
            return null;
        }
    }
    
    /**
     * 预测尿酸浓度并返回可视化结果
     * @param imageUri 图片URI
     * @return 预测结果对象
     */
    public PredictionResult predictUricAcid(Uri imageUri) {
        if (module == null) {
            return new PredictionResult("错误：模型未加载", null);
        }
        
        try {
            String imageStem = resolveImageStem(imageUri);
            PcRoiRepository.PcRoiSet pcRoiSet = pcRoiRepository.loadForImageStem(imageStem);
            if (pcRoiSet != null && !pcRoiSet.getCircleCrops().isEmpty()) {
                return predictPcRoiSet(pcRoiSet);
            }

            // 1. 加载原始图像
            Bitmap originalBitmap = loadOriginalImage(imageUri);
            if (originalBitmap == null) {
                return new PredictionResult("错误：无法加载图像", null);
            }
            Log.i(TAG, "Prediction source bitmap: " + originalBitmap.getWidth() + "x" + originalBitmap.getHeight());
            
            float predictedPPM = predictBitmapPpm(originalBitmap);
            Log.i(TAG, String.format("Prediction ppm=%.6f, model=%s", predictedPPM, loadedAssetName));
            
            // 6. 返回原始图像；预测不添加电脑端不存在的额外覆盖层
            Bitmap visualizedBitmap = originalBitmap;
            
            // 7. 格式化输出结果
            String result = formatPredictionResult(predictedPPM);
            List<Float> ppmValues = new ArrayList<>();
            ppmValues.add(predictedPPM);

            return new PredictionResult(
                    result,
                    visualizedBitmap,
                    ppmValues,
                    loadedAssetName,
                    "Direct image",
                    imageStem,
                    ""
            );
            
        } catch (Exception e) {
            Log.e(TAG, "预测失败: " + e.getMessage());
            e.printStackTrace();
            return new PredictionResult("错误：预测失败 - " + e.getMessage(), null);
        }
    }

    private PredictionResult predictPcRoiSet(PcRoiRepository.PcRoiSet pcRoiSet) throws IOException {
        List<Float> ppms = new ArrayList<>();
        List<Bitmap> crops = pcRoiSet.getCircleCrops();
        for (int i = 0; i < crops.size(); i++) {
            Bitmap crop = crops.get(i);
            Log.i(TAG, String.format(
                    Locale.US,
                    "Prediction PC ROI source: stem=%s, roi=%03d, bitmap=%dx%d",
                    pcRoiSet.getStem(),
                    i + 1,
                    crop.getWidth(),
                    crop.getHeight()
            ));
            float ppm = predictBitmapPpm(crop);
            ppms.add(ppm);
            Log.i(TAG, String.format(
                    Locale.US,
                    "Prediction PC ROI ppm=%.6f, stem=%s, roi=%03d, model=%s",
                    ppm,
                    pcRoiSet.getStem(),
                    i + 1,
                    loadedAssetName
            ));
        }

        StringBuilder result = new StringBuilder();
        result.append("===== 尿酸浓度预测结果 =====\n\n");
        result.append("输入来源: PC-step2-source\n");
        result.append("原图: ").append(pcRoiSet.getDisplayName()).append("\n");
        result.append("文件夹: ").append(pcRoiSet.getSourceFolder()).append("\n\n");
        for (int i = 0; i < ppms.size(); i++) {
            result.append(String.format(Locale.US, "ROI %03d: %.1f ppm\n", i + 1, ppms.get(i)));
        }
        result.append("\n模型: ").append(loadedAssetName);

        Bitmap visual = crops.isEmpty() ? null : crops.get(0);
        return new PredictionResult(
                result.toString(),
                visual,
                ppms,
                loadedAssetName,
                "PC-step2-source",
                pcRoiSet.getDisplayName(),
                pcRoiSet.getSourceFolder()
        );
    }

    private float predictBitmapPpm(Bitmap bitmap) throws IOException {
        Tensor inputTensor = preprocessImageForInference(bitmap);
        IValue output = module.forward(IValue.from(inputTensor));
        return output.toTensor().getDataAsFloatArray()[0];
    }
    
    /**
     * 加载原始图像（保持原始尺寸）
     */
    private Bitmap loadOriginalImage(Uri imageUri) {
        try {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            return BitmapOrientationUtils.decodeBitmap(context, imageUri, options);
        } catch (Exception e) {
            Log.e(TAG, "原始图像加载失败: " + e.getMessage());
            return null;
        }
    }

    private String resolveImageStem(Uri uri) {
        if (uri == null) {
            return "";
        }

        String name = null;
        try {
            Cursor cursor = context.getContentResolver().query(
                    uri,
                    new String[]{OpenableColumns.DISPLAY_NAME},
                    null,
                    null,
                    null
            );
            if (cursor != null) {
                try {
                    if (cursor.moveToFirst()) {
                        int idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                        if (idx >= 0) {
                            name = cursor.getString(idx);
                        }
                    }
                } finally {
                    cursor.close();
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Failed to resolve prediction display name", e);
        }

        if (name == null || name.trim().isEmpty()) {
            name = uri.getLastPathSegment();
        }
        if (name == null || name.trim().isEmpty()) {
            return "";
        }

        int dot = name.lastIndexOf('.');
        String stem = dot > 0 ? name.substring(0, dot) : name;
        stem = stem.replaceAll("[^a-zA-Z0-9._() -]", "_");
        return stem.trim();
    }
    
    private Tensor preprocessImageForInference(Bitmap originalBitmap) throws IOException {
        if (preprocessModule == null) {
            throw new IOException("PIL预处理模块未加载，无法保证与电脑端一致");
        }

        File workDir = new File(context.getCacheDir(), "uric_preprocess");
        if (!workDir.exists() && !workDir.mkdirs()) {
            throw new IOException("无法创建预处理缓存目录");
        }

        long stamp = System.nanoTime();
        File imageFile = new File(workDir, "input_" + stamp + ".png");
        File tensorFile = new File(workDir, "tensor_" + stamp + ".bin");

        try {
            try (FileOutputStream os = new FileOutputStream(imageFile)) {
                if (!originalBitmap.compress(Bitmap.CompressFormat.PNG, 100, os)) {
                    throw new IOException("预处理临时PNG写入失败");
                }
            }

            PyObject result = preprocessModule.callAttr(
                    "preprocess_to_tensor_file",
                    imageFile.getAbsolutePath(),
                    tensorFile.getAbsolutePath(),
                    MODEL_IMAGE_SIZE
            );
            String outPath = result.toJava(String.class);
            if (outPath == null || outPath.trim().isEmpty()) {
                throw new IOException("PIL预处理失败");
            }

            int expectedFloats = MODEL_CHANNELS * MODEL_IMAGE_SIZE * MODEL_IMAGE_SIZE;
            float[] data = readLittleEndianFloatFile(new File(outPath), expectedFloats);
            Log.i(TAG, String.format(
                    "PIL preprocessing done: tensor=%d floats, checksum=%.6f",
                    data.length,
                    tensorChecksum(data)
            ));
            return Tensor.fromBlob(data, new long[]{1, MODEL_CHANNELS, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE});
        } finally {
            if (imageFile.exists() && !imageFile.delete()) {
                Log.w(TAG, "Failed to delete temp image: " + imageFile.getAbsolutePath());
            }
            if (tensorFile.exists() && !tensorFile.delete()) {
                Log.w(TAG, "Failed to delete temp tensor: " + tensorFile.getAbsolutePath());
            }
        }
    }

    private float[] readLittleEndianFloatFile(File file, int expectedFloats) throws IOException {
        long expectedBytes = (long) expectedFloats * 4L;
        if (!file.exists() || file.length() != expectedBytes) {
            throw new IOException("预处理tensor尺寸不正确: " + file.getAbsolutePath());
        }

        byte[] bytes = new byte[(int) expectedBytes];
        int offset = 0;
        try (FileInputStream is = new FileInputStream(file)) {
            while (offset < bytes.length) {
                int read = is.read(bytes, offset, bytes.length - offset);
                if (read < 0) {
                    break;
                }
                offset += read;
            }
        }
        if (offset != bytes.length) {
            throw new IOException("预处理tensor读取不完整");
        }

        float[] data = new float[expectedFloats];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(data);
        return data;
    }

    private float tensorChecksum(float[] data) {
        float sum = 0f;
        int step = Math.max(1, data.length / 1024);
        for (int i = 0; i < data.length; i += step) {
            sum += data[i];
        }
        return sum;
    }
    
    /**
     * 格式化预测结果
     */
    private String formatPredictionResult(float predictedPPM) {
        StringBuilder result = new StringBuilder();
        result.append("===== 尿酸浓度预测结果 =====\n\n");
        result.append(String.format("预测浓度: %.1f ppm\n", predictedPPM));
        
        // 添加浓度范围判断
        if (predictedPPM < 50) {
            result.append("浓度等级: 低\n");
            result.append("状态: ✓ 正常范围");
        } else if (predictedPPM < 70) {
            result.append("浓度等级: 中等\n");
            result.append("状态: ⚠ 需要注意");
        } else {
            result.append("浓度等级: 高\n");
            result.append("状态: ✗ 偏高");
        }
        
        result.append("\n\n该预测基于深度学习模型: \n");
        result.append(loadedAssetName).append("\n");
        result.append("\n模型准确率: ±5-10 ppm");
        
        return result.toString();
    }
    
    /**
     * 从assets文件夹复制文件到应用私有目录
     */
    private static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file, false)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
    
    /**
     * 预测结果类，包含文本结果和可视化图像
     */
    public static class PredictionResult {
        private final String textResult;
        private final Bitmap visualBitmap;
        private final List<Float> ppmValues;
        private final String modelAssetName;
        private final String sourceLabel;
        private final String sourceName;
        private final String sourceFolder;

        public PredictionResult(String textResult, Bitmap visualBitmap) {
            this(textResult, visualBitmap, new ArrayList<Float>(), "unknown", "unknown", "", "");
        }

        public PredictionResult(
                String textResult,
                Bitmap visualBitmap,
                List<Float> ppmValues,
                String modelAssetName,
                String sourceLabel,
                String sourceName,
                String sourceFolder
        ) {
            this.textResult = textResult;
            this.visualBitmap = visualBitmap;
            this.ppmValues = new ArrayList<>(ppmValues);
            this.modelAssetName = modelAssetName == null ? "unknown" : modelAssetName;
            this.sourceLabel = sourceLabel == null ? "unknown" : sourceLabel;
            this.sourceName = sourceName == null ? "" : sourceName;
            this.sourceFolder = sourceFolder == null ? "" : sourceFolder;
        }

        public String getTextResult() {
            return textResult;
        }

        public Bitmap getVisualBitmap() {
            return visualBitmap;
        }

        public List<Float> getPpmValues() {
            return new ArrayList<>(ppmValues);
        }

        public String getModelAssetName() {
            return modelAssetName;
        }

        public String getSourceLabel() {
            return sourceLabel;
        }

        public String getSourceName() {
            return sourceName;
        }

        public String getSourceFolder() {
            return sourceFolder;
        }
    }
}
