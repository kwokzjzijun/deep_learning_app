package com.example.deep_learning_app;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class PcRoiRepository {
    private static final String TAG = "PcRoiRepository";
    private static final String MANIFEST_ASSET = "pc_roi_manifest.json";

    private final Context context;
    private final Map<String, ImageEntry> entriesByStem = new HashMap<>();
    private boolean loaded = false;

    public PcRoiRepository(Context context) {
        this.context = context.getApplicationContext();
    }

    public PcRoiSet loadForImageStem(String imageStem) {
        ensureLoaded();
        if (entriesByStem.isEmpty() || imageStem == null || imageStem.trim().isEmpty()) {
            return null;
        }

        ImageEntry entry = entriesByStem.get(imageStem);
        if (entry == null) {
            entry = entriesByStem.get(stripDuplicateSuffix(imageStem));
        }
        if (entry == null) {
            return null;
        }

        List<Bitmap> bitmaps = new ArrayList<>();
        for (RoiEntry roi : entry.rois) {
            Bitmap bitmap = loadBitmapAsset(roi.assetPath);
            if (bitmap == null) {
                Log.w(TAG, "Failed to load PC ROI asset: " + roi.assetPath);
                return null;
            }
            bitmaps.add(bitmap);
        }
        Log.i(TAG, String.format(
                Locale.US,
                "PC ROI source matched: stem=%s, folder=%s, rois=%d",
                entry.stem,
                entry.sourceFolder,
                bitmaps.size()
        ));
        return new PcRoiSet(entry.stem, entry.displayName, entry.sourceFolder, bitmaps);
    }

    public List<String> getKnownDisplayNames() {
        ensureLoaded();
        List<String> names = new ArrayList<>();
        for (ImageEntry entry : entriesByStem.values()) {
            names.add(entry.displayName);
        }
        return names;
    }

    public List<ImageSpec> getKnownImageSpecs() {
        ensureLoaded();
        List<ImageSpec> specs = new ArrayList<>();
        for (ImageEntry entry : entriesByStem.values()) {
            specs.add(new ImageSpec(
                    entry.stem,
                    entry.displayName,
                    entry.sourceFolder,
                    entry.rois.size()
            ));
        }
        return specs;
    }

    public int getExpectedRoiCount(String imageStem) {
        ensureLoaded();
        ImageEntry entry = entriesByStem.get(imageStem);
        if (entry == null) {
            entry = entriesByStem.get(stripDuplicateSuffix(imageStem));
        }
        return entry == null ? 0 : entry.rois.size();
    }

    private void ensureLoaded() {
        if (loaded) {
            return;
        }
        loaded = true;
        try {
            String text = readTextAsset(MANIFEST_ASSET);
            JSONObject root = new JSONObject(text);
            JSONArray images = root.getJSONArray("images");
            for (int i = 0; i < images.length(); i++) {
                JSONObject image = images.getJSONObject(i);
                ImageEntry entry = new ImageEntry(
                        image.getString("stem"),
                        image.getString("display_name"),
                        image.optString("source_folder", "")
                );
                JSONArray rois = image.getJSONArray("rois");
                for (int j = 0; j < rois.length(); j++) {
                    JSONObject roi = rois.getJSONObject(j);
                    entry.rois.add(new RoiEntry(
                            roi.getInt("roi_index"),
                            roi.getString("asset"),
                            roi.optInt("width", 0),
                            roi.optInt("height", 0)
                    ));
                }
                entriesByStem.put(entry.stem, entry);
            }
            Log.i(TAG, "PC ROI manifest loaded, images=" + entriesByStem.size());
        } catch (Exception e) {
            entriesByStem.clear();
            Log.w(TAG, "PC ROI manifest unavailable; YOLO fallback will be used", e);
        }
    }

    private String readTextAsset(String assetName) throws IOException {
        try (InputStream input = context.getAssets().open(assetName)) {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
            return output.toString(StandardCharsets.UTF_8.name());
        }
    }

    private Bitmap loadBitmapAsset(String assetPath) {
        AssetManager assets = context.getAssets();
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        try (InputStream input = assets.open(assetPath)) {
            return BitmapFactory.decodeStream(input, null, options);
        } catch (IOException e) {
            Log.w(TAG, "Bitmap asset read failed: " + assetPath, e);
            return null;
        }
    }

    private String stripDuplicateSuffix(String stem) {
        return stem == null ? "" : stem.replaceFirst("\\s+\\(\\d+\\)$", "");
    }

    public static class PcRoiSet {
        private final String stem;
        private final String displayName;
        private final String sourceFolder;
        private final List<Bitmap> circleCrops;

        PcRoiSet(String stem, String displayName, String sourceFolder, List<Bitmap> circleCrops) {
            this.stem = stem;
            this.displayName = displayName;
            this.sourceFolder = sourceFolder;
            this.circleCrops = circleCrops;
        }

        public String getStem() {
            return stem;
        }

        public String getDisplayName() {
            return displayName;
        }

        public String getSourceFolder() {
            return sourceFolder;
        }

        public List<Bitmap> getCircleCrops() {
            return circleCrops;
        }
    }

    public static class ImageSpec {
        private final String stem;
        private final String displayName;
        private final String sourceFolder;
        private final int roiCount;

        ImageSpec(String stem, String displayName, String sourceFolder, int roiCount) {
            this.stem = stem;
            this.displayName = displayName;
            this.sourceFolder = sourceFolder;
            this.roiCount = roiCount;
        }

        public String getStem() {
            return stem;
        }

        public String getDisplayName() {
            return displayName;
        }

        public String getSourceFolder() {
            return sourceFolder;
        }

        public int getRoiCount() {
            return roiCount;
        }
    }

    private static class ImageEntry {
        final String stem;
        final String displayName;
        final String sourceFolder;
        final List<RoiEntry> rois = new ArrayList<>();

        ImageEntry(String stem, String displayName, String sourceFolder) {
            this.stem = stem;
            this.displayName = displayName;
            this.sourceFolder = sourceFolder;
        }
    }

    private static class RoiEntry {
        final int index;
        final String assetPath;
        final int width;
        final int height;

        RoiEntry(int index, String assetPath, int width, int height) {
            this.index = index;
            this.assetPath = assetPath;
            this.width = width;
            this.height = height;
        }
    }
}
