package com.example.deep_learning_app;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.provider.OpenableColumns;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class Locate {
    private static final String TAG = "Locate";
    private static final String YOLO_ENGINE_TAG = "YOLOv8.3-locate";
    private static final String PC_ROI_ENGINE_TAG = "PC-step2-source";

    private final Context context;
    private final PcRoiRepository pcRoiRepository;
    private YoloLocateProcessor yoloProcessor;

    public Locate(Context context) throws IOException {
        this.context = context.getApplicationContext();
        this.pcRoiRepository = new PcRoiRepository(this.context);
    }

    public LocateResult run(Uri imageUri) throws IOException {
        long stamp = System.currentTimeMillis();
        String imageStem = resolveImageStem(imageUri, stamp);
        PcRoiRepository.PcRoiSet pcRoiSet = pcRoiRepository.loadForImageStem(imageStem);
        if (pcRoiSet != null && !pcRoiSet.getCircleCrops().isEmpty()) {
            return savePcRoiResult(imageUri, imageStem, pcRoiSet);
        }

        if (yoloProcessor == null) {
            yoloProcessor = new YoloLocateProcessor(this.context);
        }
        YoloLocateProcessor.LocateOutput output = yoloProcessor.run(imageUri);
        if (output.getRoiCount() <= 0) {
            throw new IOException("YOLO 未检测到 ROI，请更换图片重试");
        }

        String albumName = buildAlbumName();
        int savedCount = 0;

        List<Bitmap> circleCrops = output.getCircleCrops();
        for (int i = 0; i < circleCrops.size(); i++) {
            String file = String.format(Locale.US, "%s_roi_%03d_circle.png", imageStem, i + 1);
            if (trySaveBitmap(circleCrops.get(i), file, albumName)) {
                savedCount++;
            }
        }

        String summary = String.format(
                Locale.US,
                "%s done | ROI=%d | Saved=%d | Album=%s",
                YOLO_ENGINE_TAG,
                output.getRoiCount(),
                savedCount,
                albumName
        );

        return new LocateResult(
                output.getOverlayBitmap(),
                output.getRandomCircleBitmap(),
                output.getRoiCount(),
                savedCount,
                albumName,
                summary
        );
    }

    private LocateResult savePcRoiResult(
            Uri imageUri,
            String imageStem,
            PcRoiRepository.PcRoiSet pcRoiSet
    ) {
        String albumName = buildAlbumName();
        int savedCount = 0;

        List<Bitmap> circleCrops = pcRoiSet.getCircleCrops();
        for (int i = 0; i < circleCrops.size(); i++) {
            String file = String.format(Locale.US, "%s_roi_%03d_circle.png", imageStem, i + 1);
            if (trySaveBitmap(circleCrops.get(i), file, albumName)) {
                savedCount++;
            }
        }

        Bitmap overlay = loadOriginalBitmap(imageUri);
        Bitmap firstCircle = circleCrops.isEmpty() ? null : circleCrops.get(0);
        String summary = String.format(
                Locale.US,
                "%s done | ROI=%d | Saved=%d | Album=%s | Folder=%s",
                PC_ROI_ENGINE_TAG,
                circleCrops.size(),
                savedCount,
                albumName,
                pcRoiSet.getSourceFolder()
        );
        Log.i(TAG, String.format(
                Locale.US,
                "PC ROI locate finished, stem=%s, rois=%d, saved=%d",
                pcRoiSet.getStem(),
                circleCrops.size(),
                savedCount
        ));

        return new LocateResult(
                overlay,
                firstCircle,
                circleCrops.size(),
                savedCount,
                albumName,
                summary
        );
    }

    private boolean trySaveBitmap(Bitmap bitmap, String filename, String albumName) {
        try {
            saveBitmapToAlbum(bitmap, filename, albumName);
            return true;
        } catch (IOException e) {
            Log.e(TAG, "Save failed: " + filename + ", album=" + albumName, e);
            return false;
        }
    }

    private String buildAlbumName() {
        String date = new SimpleDateFormat("yyyyMMdd", Locale.US).format(new Date());
        return "YOLO-" + date + "-CutROI";
    }

    private String resolveImageStem(Uri uri, long fallbackStamp) {
        String fallback = "image_" + fallbackStamp;
        if (uri == null) {
            return fallback;
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
            Log.w(TAG, "Failed to resolve display name from uri", e);
        }

        if (name == null || name.trim().isEmpty()) {
            name = uri.getLastPathSegment();
        }
        if (name == null || name.trim().isEmpty()) {
            return fallback;
        }

        int dot = name.lastIndexOf('.');
        String stem = dot > 0 ? name.substring(0, dot) : name;
        stem = stem.replaceAll("[^a-zA-Z0-9._() -]", "_").trim();
        if (stem.isEmpty()) {
            return fallback;
        }
        return stem;
    }

    private Bitmap loadOriginalBitmap(Uri imageUri) {
        try {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            return BitmapOrientationUtils.decodeBitmap(context, imageUri, options);
        } catch (Exception e) {
            Log.w(TAG, "Failed to load original preview for PC ROI result", e);
            return null;
        }
    }

    private Uri saveBitmapToAlbum(Bitmap bitmap, String filename, String albumName) throws IOException {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ContentResolver resolver = context.getContentResolver();
            ContentValues values = new ContentValues();
            values.put(MediaStore.MediaColumns.DISPLAY_NAME, filename);
            values.put(MediaStore.MediaColumns.MIME_TYPE, "image/png");
            values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/" + albumName);
            Uri uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
            if (uri == null) {
                throw new IOException("Failed to create MediaStore record");
            }
            try (OutputStream os = resolver.openOutputStream(uri)) {
                if (os == null || !bitmap.compress(Bitmap.CompressFormat.PNG, 100, os)) {
                    throw new IOException("Failed to write bitmap to album");
                }
                os.flush();
            }
            return uri;
        }

        File baseDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        File albumDir = new File(baseDir, albumName);
        if (!albumDir.exists() && !albumDir.mkdirs()) {
            throw new IOException("Failed to create album directory");
        }

        File outFile = new File(albumDir, filename);
        try (FileOutputStream fos = new FileOutputStream(outFile)) {
            if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)) {
                throw new IOException("Failed to save bitmap");
            }
            fos.flush();
        }

        MediaScannerConnection.scanFile(
                context,
                new String[]{outFile.getAbsolutePath()},
                new String[]{"image/png"},
                null
        );
        return Uri.fromFile(outFile);
    }

    public static class LocateResult {
        private final Bitmap overlayBitmap;
        private final Bitmap randomCircleBitmap;
        private final int roiCount;
        private final int savedCount;
        private final String albumName;
        private final String summary;

        public LocateResult(
                Bitmap overlayBitmap,
                Bitmap randomCircleBitmap,
                int roiCount,
                int savedCount,
                String albumName,
                String summary
        ) {
            this.overlayBitmap = overlayBitmap;
            this.randomCircleBitmap = randomCircleBitmap;
            this.roiCount = roiCount;
            this.savedCount = savedCount;
            this.albumName = albumName;
            this.summary = summary;
        }

        public Bitmap getOverlayBitmap() {
            return overlayBitmap;
        }

        public Bitmap getRandomCircleBitmap() {
            return randomCircleBitmap;
        }

        public int getRoiCount() {
            return roiCount;
        }

        public int getSavedCount() {
            return savedCount;
        }

        public String getAlbumName() {
            return albumName;
        }

        public String getSummary() {
            return summary;
        }
    }
}
