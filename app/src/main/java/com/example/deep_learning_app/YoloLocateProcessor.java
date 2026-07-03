package com.example.deep_learning_app;

import android.content.ContentResolver;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.net.Uri;
import android.util.Log;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Random;

public class YoloLocateProcessor {
    private static final String TAG = "YoloLocateProcessor";
    private static final String[] MODEL_ASSET_CANDIDATES = new String[]{
            "yolo_locate_current_mobile.ptl",
            "best_YOLO11n_seg_0420_mobile.ptl",
            "yolo_best_mobile.ptl",
            "best.ptl"
    };

    private static final int INPUT_SIZE = 640;
    private static final int MAX_BOXES_AFTER_NMS = 128;
    private static final int SHAPE_CHECK_SAMPLE_ROWS = 24;

    private static final float CONF_THRESHOLD = 0.25f;
    private static final float NMS_THRESHOLD = 0.45f;
    private static final int ROI_CLASS_ID = 0;

    private final Context context;
    private final Module yoloModule;
    private final PyObject roiRefineModule;
    private final String loadedAssetName;
    private final Random random = new Random();

    public YoloLocateProcessor(Context context) throws IOException {
        this.context = context.getApplicationContext();

        String loadedAsset = null;
        Module loadedModule = null;
        IOException lastIoException = null;
        RuntimeException lastRuntimeException = null;

        for (String assetName : MODEL_ASSET_CANDIDATES) {
            if (!isSupportedLocateModel(assetName)) {
                continue;
            }

            try {
                String modelPath = assetFilePath(this.context, assetName);
                loadedModule = loadModuleSafely(modelPath, assetName);
                loadedAsset = assetName;
                break;
            } catch (IOException e) {
                lastIoException = e;
            } catch (RuntimeException e) {
                lastRuntimeException = e;
                Log.w(TAG, "YOLO model load failed for " + assetName + ": " + e.getMessage(), e);
            }
        }

        if (loadedModule == null) {
            if (lastRuntimeException != null) {
                throw new IOException("YOLO模型加载失败（模型格式或算子不兼容）", lastRuntimeException);
            }
            if (lastIoException != null) {
                throw new IOException("未找到可用的YOLO模型资产文件", lastIoException);
            }
            throw new IOException("未找到可用的YOLO模型资产文件");
        }

        this.yoloModule = loadedModule;
        this.loadedAssetName = loadedAsset;
        this.roiRefineModule = loadRoiRefineModule();
        Log.i(TAG, "YOLO model loaded: " + loadedAsset);
    }

    private PyObject loadRoiRefineModule() {
        try {
            if (!Python.isStarted()) {
                Python.start(new AndroidPlatform(context));
            }
            PyObject module = Python.getInstance().getModule("roi_refine");
            Log.i(TAG, "ROI refine module loaded");
            return module;
        } catch (RuntimeException e) {
            Log.w(TAG, "ROI refine module unavailable; refined circle crops will be skipped", e);
            return null;
        }
    }

    private static boolean isSupportedLocateModel(String assetName) {
        return assetName != null && assetName.toLowerCase(Locale.US).endsWith(".ptl");
    }

    private static Module loadModuleSafely(String modelPath, String assetName) {
        if (!assetName.toLowerCase(Locale.US).endsWith(".ptl")) {
            throw new IllegalArgumentException("Locate仅支持.ptl模型: " + assetName);
        }

        try {
            return LiteModuleLoader.load(modelPath);
        } catch (UnsatisfiedLinkError e) {
            throw new RuntimeException("PyTorch原生库加载失败，请检查jniLibs冲突", e);
        } catch (RuntimeException e) {
            throw new RuntimeException("Lite模型加载失败: " + assetName, e);
        }
    }

    public LocateOutput run(Uri imageUri) throws IOException {
        Bitmap original = loadBitmapFromUri(imageUri);
        if (original == null) {
            throw new IOException("无法读取定位图片");
        }
        return run(original);
    }

    public LocateOutput run(Bitmap originalBitmap) throws IOException {
        if (originalBitmap == null) {
            return new LocateOutput(null, null, new ArrayList<Bitmap>(), new ArrayList<Bitmap>(), 0);
        }

        LetterboxResult letterbox = createLetterbox(originalBitmap, INPUT_SIZE);
        Tensor inputTensor = bitmapToYoloInputTensor(letterbox.bitmap);

        final IValue output;
        try {
            output = yoloModule.forward(IValue.from(inputTensor));
        } catch (RuntimeException e) {
            Log.e(TAG, "YOLO forward failed", e);
            throw new IOException("YOLO推理失败（可能是模型算子与Lite运行时不兼容）", e);
        }

        ModelOutputInfo outputInfo = extractModelOutputInfo(output);
        if (outputInfo == null || outputInfo.detectionTensor == null) {
            Log.e(TAG, "YOLO output tensor is null or unsupported");
            return new LocateOutput(originalBitmap, null, new ArrayList<Bitmap>(), new ArrayList<Bitmap>(), 0);
        }

        List<DetectionBox> boxes;
        try {
            boxes = parseDetections(
                    outputInfo.detectionTensor,
                    outputInfo.protoChannels,
                    letterbox.scale,
                    letterbox.padX,
                    letterbox.padY,
                    originalBitmap.getWidth(),
                    originalBitmap.getHeight()
            );
        } catch (RuntimeException e) {
            Log.e(TAG, "Detection parse failed", e);
            throw new IOException("YOLO结果解析失败（输出格式不兼容）", e);
        }

        Bitmap overlayBitmap = drawOverlay(originalBitmap, boxes);
        List<Bitmap> rectCrops = new ArrayList<>();
        List<Bitmap> circleCrops = new ArrayList<>();
        for (int i = 0; i < boxes.size(); i++) {
            DetectionBox box = boxes.get(i);
            RefinedCrop refined = buildRefinedCircleCrop(
                    originalBitmap,
                    box,
                    outputInfo.protoTensor,
                    letterbox.scale,
                    letterbox.padX,
                    letterbox.padY
            );
            if (refined == null || refined.circleCrop == null || refined.rectCrop == null) {
                Log.w(TAG, String.format(
                        Locale.US,
                        "ROI %03d skipped: refined circle crop failed, detectRect=%dx%d, box=[%d,%d,%d,%d], conf=%.4f, model=%s",
                        i + 1,
                        box.right - box.left,
                        box.bottom - box.top,
                        box.left,
                        box.top,
                        box.right,
                        box.bottom,
                        box.conf,
                        loadedAssetName
                ));
                continue;
            }
            Bitmap rect = refined.rectCrop;
            Bitmap circle = refined.circleCrop;
            rectCrops.add(rect);
            if (circle.getWidth() != circle.getHeight()) {
                Log.w(TAG, String.format(
                        Locale.US,
                        "ROI %03d skipped: refined crop is not square, circle=%dx%d, rect=%dx%d",
                        i + 1,
                        circle.getWidth(),
                        circle.getHeight(),
                        rect.getWidth(),
                        rect.getHeight()
                ));
                continue;
            }
            Log.i(TAG, String.format(
                    Locale.US,
                    "ROI %03d refined circle accepted: rect=%dx%d, circle=%dx%d, box=[%d,%d,%d,%d], conf=%.4f, model=%s",
                    i + 1,
                    rect.getWidth(),
                    rect.getHeight(),
                    circle.getWidth(),
                    circle.getHeight(),
                    box.left,
                    box.top,
                    box.right,
                    box.bottom,
                    box.conf,
                    loadedAssetName
            ));
            circleCrops.add(circle);
        }

        Bitmap randomCircle = null;
        if (!circleCrops.isEmpty()) {
            randomCircle = circleCrops.get(random.nextInt(circleCrops.size()));
        }

        return new LocateOutput(
                overlayBitmap,
                randomCircle,
                rectCrops,
                circleCrops,
                circleCrops.size()
        );
    }

    private Tensor bitmapToYoloInputTensor(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        float[] data = new float[3 * width * height];
        int rOffset = 0;
        int gOffset = width * height;
        int bOffset = 2 * width * height;

        for (int i = 0; i < pixels.length; i++) {
            int px = pixels[i];
            float r = ((px >> 16) & 0xFF) / 255f;
            float g = ((px >> 8) & 0xFF) / 255f;
            float b = (px & 0xFF) / 255f;
            data[rOffset + i] = r;
            data[gOffset + i] = g;
            data[bOffset + i] = b;
        }

        return Tensor.fromBlob(data, new long[]{1, 3, height, width});
    }

    private Bitmap loadBitmapFromUri(Uri imageUri) throws IOException {
        ContentResolver resolver = context.getContentResolver();
        int orientation = BitmapOrientationUtils.readExifOrientation(context, imageUri);

        BitmapFactory.Options boundOptions = new BitmapFactory.Options();
        boundOptions.inJustDecodeBounds = true;
        try (InputStream is = resolver.openInputStream(imageUri)) {
            if (is == null) {
                throw new IOException("无法打开图片流");
            }
            BitmapFactory.decodeStream(is, null, boundOptions);
        }

        if (boundOptions.outWidth <= 0 || boundOptions.outHeight <= 0) {
            throw new IOException("无法读取图片尺寸信息");
        }

        Log.i(
                TAG,
                "Locate source image (original URI) size: "
                        + boundOptions.outWidth + "x" + boundOptions.outHeight
                        + ". YOLO uses source bitmap, not ImageView preview."
        );

        BitmapFactory.Options decodeOptions = new BitmapFactory.Options();
        decodeOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;
        decodeOptions.inSampleSize = 1;

        try {
            try (InputStream is = resolver.openInputStream(imageUri)) {
                if (is == null) {
                    throw new IOException("无法打开图片流");
                }
                Bitmap bitmap = BitmapFactory.decodeStream(is, null, decodeOptions);
                if (bitmap == null) {
                    throw new IOException("图片解码失败");
                }
                bitmap = BitmapOrientationUtils.applyExifOrientation(bitmap, orientation);
                Log.i(
                        TAG,
                        "Locate source bitmap after EXIF orientation: "
                                + bitmap.getWidth() + "x" + bitmap.getHeight()
                                + ", orientation=" + orientation
                );
                return bitmap;
            }
        } catch (OutOfMemoryError oom) {
            throw new IOException(
                    "原图分辨率过高导致内存不足，无法按原图处理。请先压缩图片后重试。",
                    oom
            );
        }
    }

    private ModelOutputInfo extractModelOutputInfo(IValue output) {
        if (output == null) {
            return null;
        }

        ModelOutputInfo info = new ModelOutputInfo();
        collectOutputInfo(output, info);
        return info.detectionTensor != null ? info : null;
    }

    private void collectOutputInfo(IValue value, ModelOutputInfo info) {
        if (value == null) {
            return;
        }

        if (value.isTensor()) {
            considerOutputTensor(value.toTensor(), info);
            return;
        }

        if (value.isTensorList()) {
            Tensor[] tensors = value.toTensorList();
            for (Tensor tensor : tensors) {
                considerOutputTensor(tensor, info);
            }
            return;
        }

        if (value.isTuple()) {
            for (IValue item : value.toTuple()) {
                collectOutputInfo(item, info);
            }
        }
    }

    private void considerOutputTensor(Tensor tensor, ModelOutputInfo info) {
        if (tensor == null) {
            return;
        }

        long[] shape = tensor.shape();
        if (info.detectionTensor == null && (shape.length == 3 || shape.length == 2 || shape.length == 1)) {
            info.detectionTensor = tensor;
        }

        if (shape.length == 4 && shape[0] == 1 && shape[1] > 0) {
            info.protoChannels = (int) shape[1];
            info.protoTensor = tensor;
        }
    }

    private List<DetectionBox> parseDetections(
            Tensor detectionTensor,
            Integer protoChannels,
            float scale,
            int padX,
            int padY,
            int origW,
            int origH
    ) {
        long[] shape = detectionTensor.shape();
        float[] data = detectionTensor.getDataAsFloatArray();

        Log.d(
                TAG,
                "YOLO output shape: " + formatShape(shape)
                        + ", values=" + data.length
                        + ", protoChannels=" + (protoChannels == null ? "none" : protoChannels)
        );

        if (!isSupportedOutputShape(shape)) {
            Log.e(TAG, "Unsupported detection tensor shape: " + formatShape(shape));
            return new ArrayList<>();
        }

        float maxScoreHint = estimateMaxScore(shape, data, protoChannels);
        float[] thresholds = new float[]{CONF_THRESHOLD, 0.15f, 0.08f, 0.03f, 0.01f, 0.005f, 0.001f};
        List<DetectionBox> boxes = new ArrayList<>();
        float usedThreshold = thresholds[0];

        for (float threshold : thresholds) {
            boxes = parseDetectionsWithThreshold(
                    shape,
                    data,
                    protoChannels,
                    scale,
                    padX,
                    padY,
                    origW,
                    origH,
                    threshold
            );
            usedThreshold = threshold;
            if (!boxes.isEmpty()) {
                break;
            }
        }

        if (boxes.isEmpty()) {
            DetectionBox best = tryBestScoreFallback(
                    shape,
                    data,
                    protoChannels,
                    scale,
                    padX,
                    padY,
                    origW,
                    origH
            );
            if (best != null) {
                boxes.add(best);
                Log.w(
                        TAG,
                        "Using best-score fallback box, conf="
                                + String.format(Locale.US, "%.4f", best.conf)
                                + ", maxScoreHint="
                                + String.format(Locale.US, "%.4f", maxScoreHint)
                );
            }
        }

        if (boxes.isEmpty()) {
            Log.w(
                    TAG,
                    "No ROI detected after threshold fallback, maxScoreHint="
                            + String.format(Locale.US, "%.4f", maxScoreHint)
            );
            return boxes;
        }

        if (usedThreshold < CONF_THRESHOLD) {
            Log.w(TAG, "ROI detected with fallback threshold=" + usedThreshold);
        }

        List<DetectionBox> deduped = applyNms(boxes, NMS_THRESHOLD, MAX_BOXES_AFTER_NMS);

        // Keep the same ordering semantics as Deal_YOLO_Result.py: sort by box center (x then y).
        Collections.sort(deduped, new Comparator<DetectionBox>() {
            @Override
            public int compare(DetectionBox a, DetectionBox b) {
                float acx = (a.left + a.right) / 2f;
                float bcx = (b.left + b.right) / 2f;
                int cmpX = Float.compare(acx, bcx);
                if (cmpX != 0) {
                    return cmpX;
                }
                float acy = (a.top + a.bottom) / 2f;
                float bcy = (b.top + b.bottom) / 2f;
                return Float.compare(acy, bcy);
            }
        });

        return deduped;
    }

    private DetectionBox tryBestScoreFallback(
            long[] shape,
            float[] data,
            Integer protoChannels,
            float scale,
            int padX,
            int padY,
            int origW,
            int origH
    ) {
        List<DetectionBox> relaxed = parseDetectionsWithThreshold(
                shape,
                data,
                protoChannels,
                scale,
                padX,
                padY,
                origW,
                origH,
                -1f
        );
        if (relaxed.isEmpty()) {
            return null;
        }
        DetectionBox best = relaxed.get(0);
        for (int i = 1; i < relaxed.size(); i++) {
            DetectionBox box = relaxed.get(i);
            if (box.conf > best.conf) {
                best = box;
            }
        }
        return best;
    }

    private boolean isSupportedOutputShape(long[] shape) {
        if (shape.length == 3 && shape[0] == 1) {
            int dim1 = (int) shape[1];
            int dim2 = (int) shape[2];
            return (isLikelyChannelsFirst(dim1, dim2) || dim2 >= 5 || dim1 >= 5);
        }
        if (shape.length == 2) {
            return shape[1] >= 5;
        }
        return shape.length == 1 && shape[0] % 6 == 0;
    }

    private List<DetectionBox> parseDetectionsWithThreshold(
            long[] shape,
            float[] data,
            Integer protoChannels,
            float scale,
            int padX,
            int padY,
            int origW,
            int origH,
            float confThreshold
    ) {
        List<DetectionBox> boxes = new ArrayList<>();

        if (shape.length == 3 && shape[0] == 1) {
            int dim1 = (int) shape[1];
            int dim2 = (int) shape[2];
            if (isLikelyChannelsFirst(dim1, dim2)) {
                parseChannelsFirstFormat(
                        data, dim1, dim2, protoChannels, scale, padX, padY, origW, origH, boxes, confThreshold
                );
            } else if (dim2 >= 5) {
                parseRowsFormat(
                        data, dim1, dim2, protoChannels, scale, padX, padY, origW, origH, boxes, confThreshold
                );
            }
            return boxes;
        }

        if (shape.length == 2) {
            int rows = (int) shape[0];
            int cols = (int) shape[1];
            parseRowsFormat(
                    data, rows, cols, protoChannels, scale, padX, padY, origW, origH, boxes, confThreshold
            );
            return boxes;
        }

        int rows = (int) (shape[0] / 6);
        parseRowsFormat(
                data, rows, 6, protoChannels, scale, padX, padY, origW, origH, boxes, confThreshold
        );
        return boxes;
    }

    private float estimateMaxScore(long[] shape, float[] data, Integer protoChannels) {
        if (shape.length == 3 && shape[0] == 1) {
            int dim1 = (int) shape[1];
            int dim2 = (int) shape[2];
            if (isLikelyChannelsFirst(dim1, dim2) && dim1 >= 5 && dim2 > 0) {
                float max = 0f;
                int classCount = inferRawClassCount(dim1, protoChannels);
                for (int c = 0; c < classCount; c++) {
                    int channelOffset = (4 + c) * dim2;
                    int upper = Math.min(data.length, channelOffset + dim2);
                    for (int i = channelOffset; i < upper; i++) {
                        float score = normalizeScore(data[i]);
                        if (Float.isFinite(score) && score > max) {
                            max = score;
                        }
                    }
                }
                return max;
            }
        }
        float max = 0f;
        for (float v : data) {
            float score = normalizeScore(v);
            if (Float.isFinite(score) && score > max) {
                max = score;
            }
        }
        return max;
    }

    private boolean isLikelyChannelsFirst(int dim1, int dim2) {
        // Typical raw YOLO export: [1, C, N], where N (candidate count) is much larger than C.
        if (dim1 > 0 && dim2 > dim1 && dim2 >= 128) {
            return true;
        }
        // Backward-compatible heuristic for older small-channel exports.
        return dim1 > 0 && dim1 <= 16 && dim2 >= 32;
    }

    private void parseRowsFormat(
            float[] data,
            int rows,
            int cols,
            Integer protoChannels,
            float scale,
            int padX,
            int padY,
            int origW,
            int origH,
            List<DetectionBox> boxes,
            float confThreshold
    ) {
        if (cols < 5) {
            Log.e(TAG, "Detection tensor columns < 5, cols=" + cols);
            return;
        }

        int expected = rows * cols;
        int safeLength = Math.min(data.length, expected);

        boolean useNmsFormat;
        if (cols < 6) {
            useNmsFormat = false;
        } else if (cols <= 8) {
            useNmsFormat = looksLikeNmsRows(data, rows, cols, safeLength);
        } else {
            useNmsFormat = false;
        }

        for (int i = 0; i < rows; i++) {
            int base = i * cols;
            if (base + cols > safeLength) {
                break;
            }

            if (useNmsFormat) {
                addCandidateBox(
                        boxes,
                        data[base],
                        data[base + 1],
                        data[base + 2],
                        data[base + 3],
                        data[base + 4],
                        Math.round(data[base + 5]),
                        null,
                        scale,
                        padX,
                        padY,
                        origW,
                        origH,
                        confThreshold
                );
                continue;
            }

            parseRawYoloRow(
                    data, base, cols, protoChannels, scale, padX, padY, origW, origH, boxes, confThreshold
            );
        }
    }

    private boolean looksLikeNmsRows(float[] data, int rows, int cols, int safeLength) {
        int sampleRows = Math.min(rows, SHAPE_CHECK_SAMPLE_ROWS);
        int validSamples = 0;
        int nmsLike = 0;

        for (int i = 0; i < sampleRows; i++) {
            int base = i * cols;
            if (base + 6 > safeLength) {
                break;
            }

            float x1 = data[base];
            float y1 = data[base + 1];
            float x2 = data[base + 2];
            float y2 = data[base + 3];
            float conf = data[base + 4];
            float cls = data[base + 5];

            if (!Float.isFinite(x1) || !Float.isFinite(y1)
                    || !Float.isFinite(x2) || !Float.isFinite(y2)
                    || !Float.isFinite(conf) || !Float.isFinite(cls)) {
                continue;
            }

            validSamples++;
            boolean cornersLike = x2 >= x1 && y2 >= y1;
            boolean confLike = conf >= -0.05f && conf <= 1.5f;
            boolean clsLike = Math.abs(cls - Math.round(cls)) < 1e-3f;
            if (cornersLike && confLike && clsLike) {
                nmsLike++;
            }
        }

        if (validSamples == 0) {
            return false;
        }
        return nmsLike * 2 >= validSamples;
    }

    private void parseRawYoloRow(
            float[] data,
            int base,
            int cols,
            Integer protoChannels,
            float scale,
            int padX,
            int padY,
            int origW,
            int origH,
            List<DetectionBox> boxes,
            float confThreshold
    ) {
        float cx = data[base];
        float cy = data[base + 1];
        float w = Math.max(0f, data[base + 2]);
        float h = Math.max(0f, data[base + 3]);

        int classId = 0;
        int classCount = inferRawClassCount(cols, protoChannels);
        ClassScore bestClass = getBestClassScoreFromRow(data, base + 4, classCount);
        classId = bestClass.classId;
        float classScore = bestClass.score;
        float[] maskCoefficients = extractMaskCoefficientsFromRow(data, base, cols, classCount, protoChannels);

        if (!Float.isFinite(classScore)) {
            return;
        }

        float conf = normalizeScore(classScore);

        addCandidateBox(
                boxes,
                cx - (w / 2f),
                cy - (h / 2f),
                cx + (w / 2f),
                cy + (h / 2f),
                conf,
                classId,
                maskCoefficients,
                scale,
                padX,
                padY,
                origW,
                origH,
                confThreshold
        );
    }

    private void parseChannelsFirstFormat(
            float[] data,
            int numChannels,
            int numCandidates,
            Integer protoChannels,
            float scale,
            int padX,
            int padY,
            int origW,
            int origH,
            List<DetectionBox> boxes,
            float confThreshold
    ) {
        if (numChannels < 5 || numCandidates <= 0) {
            Log.e(TAG, "Invalid channels-first shape CxN = " + numChannels + "x" + numCandidates);
            return;
        }

        int requiredLength = numChannels * numCandidates;
        if (data.length < requiredLength) {
            Log.e(TAG, "Unexpected tensor data length, required=" + requiredLength + ", actual=" + data.length);
            return;
        }

        int classCount = inferRawClassCount(numChannels, protoChannels);
        for (int i = 0; i < numCandidates; i++) {
            float cx = data[i];
            float cy = data[numCandidates + i];
            float w = Math.max(0f, data[(2 * numCandidates) + i]);
            float h = Math.max(0f, data[(3 * numCandidates) + i]);

            int classId = 0;
            ClassScore bestClass = getBestClassScoreFromChannels(data, i, numCandidates, 4, classCount);
            classId = bestClass.classId;
            float classScore = bestClass.score;
            float[] maskCoefficients = extractMaskCoefficientsFromChannels(
                    data, i, numCandidates, 4 + classCount, protoChannels
            );

            float conf = normalizeScore(classScore);

            addCandidateBox(
                    boxes,
                    cx - (w / 2f),
                    cy - (h / 2f),
                    cx + (w / 2f),
                    cy + (h / 2f),
                    conf,
                    classId,
                    maskCoefficients,
                    scale,
                    padX,
                    padY,
                    origW,
                    origH,
                    confThreshold
            );
        }
    }

    private ClassScore getBestClassScoreFromRow(float[] row, int start, int length) {
        if (length <= 0) {
            return new ClassScore(0, 1f);
        }

        float bestScore = Float.NEGATIVE_INFINITY;
        int bestClass = 0;
        for (int i = 0; i < length; i++) {
            float score = row[start + i];
            if (score > bestScore) {
                bestScore = score;
                bestClass = i;
            }
        }
        return new ClassScore(bestClass, bestScore);
    }

    private ClassScore getBestClassScoreFromChannels(
            float[] data,
            int candidateIdx,
            int numCandidates,
            int classStartChannel,
            int classCount
    ) {
        float bestScore = Float.NEGATIVE_INFINITY;
        int bestClass = 0;

        for (int c = classStartChannel; c < (classStartChannel + classCount); c++) {
            float score = data[(c * numCandidates) + candidateIdx];
            if (score > bestScore) {
                bestScore = score;
                bestClass = c - classStartChannel;
            }
        }
        return new ClassScore(bestClass, bestScore);
    }

    private int inferRawClassCount(int totalChannels, Integer protoChannels) {
        int classCount = totalChannels - 4;
        if (protoChannels != null && protoChannels > 0 && totalChannels > (4 + protoChannels)) {
            int inferred = totalChannels - 4 - protoChannels;
            if (inferred > 0) {
                classCount = inferred;
            }
        }
        return Math.max(1, classCount);
    }

    private float[] extractMaskCoefficientsFromRow(
            float[] data,
            int base,
            int cols,
            int classCount,
            Integer protoChannels
    ) {
        if (protoChannels == null || protoChannels <= 0) {
            return null;
        }
        int start = base + 4 + classCount;
        int end = start + protoChannels;
        if (end > base + cols || end > data.length) {
            return null;
        }
        float[] coeffs = new float[protoChannels];
        System.arraycopy(data, start, coeffs, 0, protoChannels);
        return coeffs;
    }

    private float[] extractMaskCoefficientsFromChannels(
            float[] data,
            int candidateIdx,
            int numCandidates,
            int coeffStartChannel,
            Integer protoChannels
    ) {
        if (protoChannels == null || protoChannels <= 0) {
            return null;
        }
        float[] coeffs = new float[protoChannels];
        for (int c = 0; c < protoChannels; c++) {
            coeffs[c] = data[((coeffStartChannel + c) * numCandidates) + candidateIdx];
        }
        return coeffs;
    }

    private void addCandidateBox(
            List<DetectionBox> boxes,
            float x1,
            float y1,
            float x2,
            float y2,
            float conf,
            int cls,
            float[] maskCoefficients,
            float scale,
            int padX,
            int padY,
            int origW,
            int origH,
            float confThreshold
    ) {
        if (!Float.isFinite(x1) || !Float.isFinite(y1)
                || !Float.isFinite(x2) || !Float.isFinite(y2)
                || !Float.isFinite(conf)) {
            return;
        }

        conf = normalizeScore(conf);
        if (conf < confThreshold || cls != ROI_CLASS_ID) {
            return;
        }

        float minX = Math.min(x1, x2);
        float minY = Math.min(y1, y2);
        float maxX = Math.max(x1, x2);
        float maxY = Math.max(y1, y2);

        float ox1 = (minX - padX) / scale;
        float oy1 = (minY - padY) / scale;
        float ox2 = (maxX - padX) / scale;
        float oy2 = (maxY - padY) / scale;

        // Same clamp logic as Deal_YOLO_Result.py
        int left = clamp((int) Math.floor(ox1), 0, origW - 1);
        int top = clamp((int) Math.floor(oy1), 0, origH - 1);
        int right = clamp((int) Math.ceil(ox2), 1, origW);
        int bottom = clamp((int) Math.ceil(oy2), 1, origH);

        if (right <= left || bottom <= top) {
            return;
        }

        boxes.add(new DetectionBox(left, top, right, bottom, minX, minY, maxX, maxY, conf, maskCoefficients));
    }

    private float normalizeScore(float score) {
        if (!Float.isFinite(score)) {
            return Float.NaN;
        }
        if (score >= 0f && score <= 1f) {
            return score;
        }
        float clipped = Math.max(-30f, Math.min(30f, score));
        return (float) (1.0 / (1.0 + Math.exp(-clipped)));
    }

    private List<DetectionBox> applyNms(List<DetectionBox> input, float iouThreshold, int maxKeep) {
        if (input.isEmpty()) {
            return input;
        }

        List<DetectionBox> sorted = new ArrayList<>(input);
        Collections.sort(sorted, new Comparator<DetectionBox>() {
            @Override
            public int compare(DetectionBox a, DetectionBox b) {
                return Float.compare(b.conf, a.conf);
            }
        });

        List<DetectionBox> kept = new ArrayList<>();
        boolean[] suppressed = new boolean[sorted.size()];

        for (int i = 0; i < sorted.size(); i++) {
            if (suppressed[i]) {
                continue;
            }

            DetectionBox current = sorted.get(i);
            kept.add(current);
            if (kept.size() >= maxKeep) {
                break;
            }

            for (int j = i + 1; j < sorted.size(); j++) {
                if (suppressed[j]) {
                    continue;
                }
                if (iou(current, sorted.get(j)) > iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }

        return kept;
    }

    private float iou(DetectionBox a, DetectionBox b) {
        int interLeft = Math.max(a.left, b.left);
        int interTop = Math.max(a.top, b.top);
        int interRight = Math.min(a.right, b.right);
        int interBottom = Math.min(a.bottom, b.bottom);

        int interW = interRight - interLeft;
        int interH = interBottom - interTop;
        if (interW <= 0 || interH <= 0) {
            return 0f;
        }

        float interArea = interW * interH;
        float areaA = (a.right - a.left) * (a.bottom - a.top);
        float areaB = (b.right - b.left) * (b.bottom - b.top);
        float union = areaA + areaB - interArea;
        if (union <= 0f) {
            return 0f;
        }
        return interArea / union;
    }

    private Bitmap drawOverlay(Bitmap originalBitmap, List<DetectionBox> boxes) {
        Bitmap overlay = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(overlay);

        Paint boxPaint = new Paint();
        boxPaint.setAntiAlias(true);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(Math.max(3f, Math.min(8f, originalBitmap.getWidth() / 220f)));
        boxPaint.setColor(Color.rgb(0, 255, 0));

        Paint textPaint = new Paint();
        textPaint.setAntiAlias(true);
        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setColor(Color.rgb(0, 255, 0));
        textPaint.setTextSize(Math.max(20f, originalBitmap.getWidth() / 38f));

        for (int i = 0; i < boxes.size(); i++) {
            DetectionBox box = boxes.get(i);
            canvas.drawRect(box.left, box.top, box.right, box.bottom, boxPaint);
            String label = String.format(Locale.US, "ROI%02d %.2f", i + 1, box.conf);
            canvas.drawText(label, box.left, Math.max(24, box.top - 8), textPaint);
        }

        return overlay;
    }

    private Bitmap cropRect(Bitmap source, Rect rect) {
        int width = rect.width();
        int height = rect.height();
        if (width <= 0 || height <= 0) {
            return null;
        }

        try {
            return Bitmap.createBitmap(source, rect.left, rect.top, width, height);
        } catch (Exception e) {
            Log.w(TAG, "Crop rect failed: " + e.getMessage(), e);
            return null;
        }
    }

    private RefinedCrop buildRefinedCircleCrop(
            Bitmap source,
            DetectionBox box,
            Tensor protoTensor,
            float scale,
            int padX,
            int padY
    ) {
        if (source == null || protoTensor == null || box.maskCoefficients == null || roiRefineModule == null) {
            return null;
        }

        LocalMaskCrop maskCrop = buildLocalMaskCrop(source, box, protoTensor, scale, padX, padY);
        if (maskCrop == null || maskCrop.rectCrop == null || maskCrop.maskBitmap == null) {
            return null;
        }

        File workDir = new File(context.getCacheDir(), "roi_refine");
        if (!workDir.exists() && !workDir.mkdirs()) {
            Log.w(TAG, "Failed to create roi_refine cache directory");
            return null;
        }

        File roiFile = null;
        File maskFile = null;
        File outFile = null;
        try {
            roiFile = File.createTempFile("roi_", ".png", workDir);
            maskFile = File.createTempFile("mask_", ".png", workDir);
            outFile = File.createTempFile("circle_", ".png", workDir);

            saveBitmapPng(maskCrop.rectCrop, roiFile);
            saveBitmapPng(maskCrop.maskBitmap, maskFile);

            PyObject result = roiRefineModule.callAttr(
                    "refine_circle_crop",
                    roiFile.getAbsolutePath(),
                    maskFile.getAbsolutePath(),
                    outFile.getAbsolutePath()
            );
            if (result == null) {
                return null;
            }

            String outPath = result.toJava(String.class);
            if (outPath == null || outPath.isEmpty()) {
                return null;
            }

            Bitmap refined = BitmapFactory.decodeFile(outPath);
            if (refined == null) {
                Log.w(TAG, "Python refine returned empty bitmap: " + outPath);
            }
            return new RefinedCrop(maskCrop.rectCrop, refined);
        } catch (Exception e) {
            Log.w(TAG, "Python circle refinement failed: " + e.getMessage(), e);
            return null;
        } finally {
            if (roiFile != null) {
                roiFile.delete();
            }
            if (maskFile != null) {
                maskFile.delete();
            }
            if (outFile != null) {
                outFile.delete();
            }
        }
    }

    private void saveBitmapPng(Bitmap bitmap, File file) throws IOException {
        try (OutputStream os = new FileOutputStream(file, false)) {
            if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, os)) {
                throw new IOException("Bitmap PNG write failed: " + file.getAbsolutePath());
            }
            os.flush();
        }
    }

    private LocalMaskCrop buildLocalMaskCrop(
            Bitmap source,
            DetectionBox box,
            Tensor protoTensor,
            float scale,
            int padX,
            int padY
    ) {
        Rect sourceRect = new Rect(box.left, box.top, box.right, box.bottom);
        Bitmap rectCrop = cropRect(source, sourceRect);
        if (rectCrop == null) {
            return null;
        }

        Bitmap localMask = buildLocalMaskBitmap(sourceRect, box, protoTensor, scale, padX, padY);
        if (localMask == null) {
            return null;
        }

        Log.i(TAG, String.format(
                Locale.US,
                "Local mask bbox crop: rect=[%d,%d,%d,%d], size=%dx%d",
                sourceRect.left,
                sourceRect.top,
                sourceRect.right,
                sourceRect.bottom,
                rectCrop.getWidth(),
                rectCrop.getHeight()
        ));
        return new LocalMaskCrop(rectCrop, localMask);
    }

    private Bitmap buildLocalMaskBitmap(
            Rect region,
            DetectionBox box,
            Tensor protoTensor,
            float scale,
            int padX,
            int padY
    ) {
        long[] protoShape = protoTensor.shape();
        if (protoShape.length != 4 || protoShape[0] != 1) {
            return null;
        }

        int protoChannels = (int) protoShape[1];
        int protoH = (int) protoShape[2];
        int protoW = (int) protoShape[3];
        if (box.maskCoefficients.length != protoChannels) {
            return null;
        }

        int width = region.width();
        int height = region.height();
        if (width <= 0 || height <= 0) {
            return null;
        }

        float[] protoData = protoTensor.getDataAsFloatArray();
        float[] lowResMask = buildProtoMask(
                protoData,
                protoChannels,
                protoH,
                protoW,
                box
        );

        int[] pixels = new int[width * height];
        int fgCount = 0;
        for (int y = 0; y < height; y++) {
            float origY = region.top + y + 0.5f;
            float inputY = origY * scale + padY;
            float protoY = (inputY * protoH / (float) INPUT_SIZE) - 0.5f;
            for (int x = 0; x < width; x++) {
                float origX = region.left + x + 0.5f;
                float inputX = origX * scale + padX;
                float protoX = (inputX * protoW / (float) INPUT_SIZE) - 0.5f;

                float logit = bilinearMaskSample(lowResMask, protoH, protoW, protoX, protoY);
                boolean fg = logit > 0f;
                if (fg) {
                    pixels[y * width + x] = Color.WHITE;
                    fgCount++;
                } else {
                    pixels[y * width + x] = Color.BLACK;
                }
            }
        }

        if (fgCount < 16) {
            Log.w(TAG, String.format(
                    Locale.US,
                    "Local mask too small after rebuild: fg=%d, contextRect=[%d,%d,%d,%d], inputBox=[%.2f,%.2f,%.2f,%.2f], proto=%dx%d",
                    fgCount,
                    region.left,
                    region.top,
                    region.right,
                    region.bottom,
                    box.inputLeft,
                    box.inputTop,
                    box.inputRight,
                    box.inputBottom,
                    protoW,
                    protoH
            ));
            return null;
        }

        Log.i(TAG, String.format(
                Locale.US,
                "Local mask rebuilt: fg=%d/%d, contextRect=[%d,%d,%d,%d], inputBox=[%.2f,%.2f,%.2f,%.2f], proto=%dx%d",
                fgCount,
                width * height,
                region.left,
                region.top,
                region.right,
                region.bottom,
                box.inputLeft,
                box.inputTop,
                box.inputRight,
                box.inputBottom,
                protoW,
                protoH
        ));

        Bitmap maskBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        maskBitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return maskBitmap;
    }

    private float[] buildProtoMask(
            float[] protoData,
            int channels,
            int protoH,
            int protoW,
            DetectionBox box
    ) {
        float[] mask = new float[protoH * protoW];
        float widthRatio = protoW / (float) INPUT_SIZE;
        float heightRatio = protoH / (float) INPUT_SIZE;
        float cropLeft = box.inputLeft * widthRatio;
        float cropRight = box.inputRight * widthRatio;
        float cropTop = box.inputTop * heightRatio;
        float cropBottom = box.inputBottom * heightRatio;

        for (int y = 0; y < protoH; y++) {
            boolean inY = y >= cropTop && y < cropBottom;
            int row = y * protoW;
            for (int x = 0; x < protoW; x++) {
                if (!inY || x < cropLeft || x >= cropRight) {
                    mask[row + x] = 0f;
                    continue;
                }

                float logit = 0f;
                for (int c = 0; c < channels; c++) {
                    logit += box.maskCoefficients[c] * protoValue(protoData, channels, protoH, protoW, c, x, y);
                }
                mask[row + x] = logit;
            }
        }
        return mask;
    }

    private float bilinearMaskSample(float[] mask, int height, int width, float x, float y) {
        float clampedX = Math.max(0f, Math.min(width - 1.001f, x));
        float clampedY = Math.max(0f, Math.min(height - 1.001f, y));

        int x0 = (int) Math.floor(clampedX);
        int y0 = (int) Math.floor(clampedY);
        int x1 = Math.min(width - 1, x0 + 1);
        int y1 = Math.min(height - 1, y0 + 1);

        float dx = clampedX - x0;
        float dy = clampedY - y0;

        float v00 = mask[y0 * width + x0];
        float v01 = mask[y0 * width + x1];
        float v10 = mask[y1 * width + x0];
        float v11 = mask[y1 * width + x1];

        float top = v00 + dx * (v01 - v00);
        float bottom = v10 + dx * (v11 - v10);
        return top + dy * (bottom - top);
    }

    private float bilinearProtoSample(
            float[] protoData,
            int channels,
            int protoH,
            int protoW,
            int channel,
            float x,
            float y
    ) {
        float clampedX = Math.max(0f, Math.min(protoW - 1.001f, x));
        float clampedY = Math.max(0f, Math.min(protoH - 1.001f, y));

        int x0 = (int) Math.floor(clampedX);
        int y0 = (int) Math.floor(clampedY);
        int x1 = Math.min(protoW - 1, x0 + 1);
        int y1 = Math.min(protoH - 1, y0 + 1);

        float dx = clampedX - x0;
        float dy = clampedY - y0;

        float v00 = protoValue(protoData, channels, protoH, protoW, channel, x0, y0);
        float v01 = protoValue(protoData, channels, protoH, protoW, channel, x1, y0);
        float v10 = protoValue(protoData, channels, protoH, protoW, channel, x0, y1);
        float v11 = protoValue(protoData, channels, protoH, protoW, channel, x1, y1);

        float top = v00 + dx * (v01 - v00);
        float bottom = v10 + dx * (v11 - v10);
        return top + dy * (bottom - top);
    }

    private float protoValue(
            float[] protoData,
            int channels,
            int protoH,
            int protoW,
            int channel,
            int x,
            int y
    ) {
        int index = ((channel * protoH) + y) * protoW + x;
        if (index < 0 || index >= protoData.length) {
            return 0f;
        }
        return protoData[index];
    }

    private float sigmoid(float value) {
        float clipped = Math.max(-30f, Math.min(30f, value));
        return (float) (1.0 / (1.0 + Math.exp(-clipped)));
    }

    private LetterboxResult createLetterbox(Bitmap source, int targetSize) {
        int srcW = source.getWidth();
        int srcH = source.getHeight();
        float scale = Math.min((float) targetSize / srcW, (float) targetSize / srcH);

        int newW = Math.max(1, Math.round(srcW * scale));
        int newH = Math.max(1, Math.round(srcH * scale));
        int padX = (targetSize - newW) / 2;
        int padY = (targetSize - newH) / 2;

        Bitmap resized = Bitmap.createScaledBitmap(source, newW, newH, true);
        Bitmap padded = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(padded);
        canvas.drawColor(Color.BLACK);
        canvas.drawBitmap(resized, padX, padY, null);

        return new LetterboxResult(padded, scale, padX, padY);
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private String formatShape(long[] shape) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(shape[i]);
        }
        sb.append("]");
        return sb.toString();
    }

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
        }
        return file.getAbsolutePath();
    }

    private static class ClassScore {
        final int classId;
        final float score;

        ClassScore(int classId, float score) {
            this.classId = classId;
            this.score = score;
        }
    }

    private static class ModelOutputInfo {
        Tensor detectionTensor;
        Integer protoChannels;
        Tensor protoTensor;
    }

    private static class LocalMaskCrop {
        final Bitmap rectCrop;
        final Bitmap maskBitmap;

        LocalMaskCrop(Bitmap rectCrop, Bitmap maskBitmap) {
            this.rectCrop = rectCrop;
            this.maskBitmap = maskBitmap;
        }
    }

    private static class RefinedCrop {
        final Bitmap rectCrop;
        final Bitmap circleCrop;

        RefinedCrop(Bitmap rectCrop, Bitmap circleCrop) {
            this.rectCrop = rectCrop;
            this.circleCrop = circleCrop;
        }
    }

    private static class DetectionBox {
        final int left;
        final int top;
        final int right;
        final int bottom;
        final float inputLeft;
        final float inputTop;
        final float inputRight;
        final float inputBottom;
        final float conf;
        final float[] maskCoefficients;

        DetectionBox(
                int left,
                int top,
                int right,
                int bottom,
                float inputLeft,
                float inputTop,
                float inputRight,
                float inputBottom,
                float conf,
                float[] maskCoefficients
        ) {
            this.left = left;
            this.top = top;
            this.right = right;
            this.bottom = bottom;
            this.inputLeft = inputLeft;
            this.inputTop = inputTop;
            this.inputRight = inputRight;
            this.inputBottom = inputBottom;
            this.conf = conf;
            this.maskCoefficients = maskCoefficients;
        }
    }

    private static class LetterboxResult {
        final Bitmap bitmap;
        final float scale;
        final int padX;
        final int padY;

        LetterboxResult(Bitmap bitmap, float scale, int padX, int padY) {
            this.bitmap = bitmap;
            this.scale = scale;
            this.padX = padX;
            this.padY = padY;
        }
    }

    public static class LocateOutput {
        private final Bitmap overlayBitmap;
        private final Bitmap randomCircleBitmap;
        private final List<Bitmap> rectCrops;
        private final List<Bitmap> circleCrops;
        private final int roiCount;

        LocateOutput(
                Bitmap overlayBitmap,
                Bitmap randomCircleBitmap,
                List<Bitmap> rectCrops,
                List<Bitmap> circleCrops,
                int roiCount
        ) {
            this.overlayBitmap = overlayBitmap;
            this.randomCircleBitmap = randomCircleBitmap;
            this.rectCrops = rectCrops;
            this.circleCrops = circleCrops;
            this.roiCount = roiCount;
        }

        public Bitmap getOverlayBitmap() {
            return overlayBitmap;
        }

        public Bitmap getRandomCircleBitmap() {
            return randomCircleBitmap;
        }

        public List<Bitmap> getRectCrops() {
            return rectCrops;
        }

        public List<Bitmap> getCircleCrops() {
            return circleCrops;
        }

        public int getRoiCount() {
            return roiCount;
        }
    }
}
