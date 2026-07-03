package com.example.deep_learning_app;

import android.content.ContentResolver;
import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

@RunWith(AndroidJUnit4.class)
public class PcRoiSourceInstrumentedTest {
    private static final String TAG = "PcRoiSourceTest";

    @Test
    public void pcRoiManifestAssetsLoad() {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        PcRoiRepository repository = new PcRoiRepository(context);
        List<PcRoiRepository.ImageSpec> specs = repository.getKnownImageSpecs();

        int totalRois = 0;
        for (PcRoiRepository.ImageSpec spec : specs) {
            totalRois += spec.getRoiCount();
            PcRoiRepository.PcRoiSet set = repository.loadForImageStem(spec.getStem());
            assertNotNull("Expected PC ROI set for " + spec.getStem(), set);
            assertEquals(spec.getRoiCount(), set.getCircleCrops().size());
            assertFalse(set.getCircleCrops().isEmpty());
            assertTrue(set.getCircleCrops().get(0).getWidth() > 0);
            assertEquals(
                    "Expected square first ROI for " + spec.getStem(),
                    set.getCircleCrops().get(0).getWidth(),
                    set.getCircleCrops().get(0).getHeight()
            );
        }

        assertEquals("Expected Clinical2026_0424 image count", 26, specs.size());
        assertEquals("Expected Clinical2026_0424 PC ROI count", 65, totalRois);
    }

    @Test
    public void knownClinicalMediaStoreImagesUsePcSourceWhenPresent() throws Exception {
        Bundle args = InstrumentationRegistry.getArguments();
        boolean requireAll = Boolean.parseBoolean(args.getString("requireAll", "false"));
        int limit = parseIntArg(args.getString("limit", "0"));

        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        PcRoiRepository repository = new PcRoiRepository(context);
        List<PcRoiRepository.ImageSpec> specs = repository.getKnownImageSpecs();
        Collections.sort(specs, new Comparator<PcRoiRepository.ImageSpec>() {
            @Override
            public int compare(PcRoiRepository.ImageSpec a, PcRoiRepository.ImageSpec b) {
                return a.getDisplayName().compareTo(b.getDisplayName());
            }
        });

        Locate locate = new Locate(context);
        int checked = 0;
        int missing = 0;
        StringBuilder missingNames = new StringBuilder();
        for (PcRoiRepository.ImageSpec spec : specs) {
            if (limit > 0 && checked >= limit) {
                break;
            }

            Uri uri = findImageUriByDisplayName(context, spec.getDisplayName());
            if (uri == null) {
                missing++;
                if (missingNames.length() > 0) {
                    missingNames.append(", ");
                }
                missingNames.append(spec.getDisplayName());
                continue;
            }

            Locate.LocateResult result = locate.run(uri);
            Log.i(TAG, "displayName=" + spec.getDisplayName() + ", uri=" + uri
                    + ", summary=" + result.getSummary());
            assertTrue(
                    "Expected PC source summary for " + spec.getDisplayName(),
                    result.getSummary().startsWith("PC-step2-source")
            );
            assertEquals(
                    "ROI count mismatch for " + spec.getDisplayName(),
                    spec.getRoiCount(),
                    result.getRoiCount()
            );
            assertEquals(
                    "Saved count mismatch for " + spec.getDisplayName(),
                    spec.getRoiCount(),
                    result.getSavedCount()
            );
            assertNotNull("Expected PC ROI preview for " + spec.getDisplayName(), result.getRandomCircleBitmap());
            checked++;
        }

        if (requireAll && missing > 0) {
            throw new AssertionError("Missing MediaStore originals: " + missingNames);
        }
        assertTrue("Expected at least one known Clinical2026_0424 image on device", checked > 0);
        Log.i(TAG, "Known Clinical batch checked=" + checked + ", missing=" + missing);
    }

    private Uri findImageUriByDisplayName(Context context, String displayName) {
        ContentResolver resolver = context.getContentResolver();
        String[] projection = new String[]{MediaStore.Images.Media._ID};
        String selection = MediaStore.Images.Media.DISPLAY_NAME + "=?";
        String[] selectionArgs = new String[]{displayName};
        String sortOrder = MediaStore.Images.Media.DATE_ADDED + " DESC";
        try (Cursor cursor = resolver.query(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                projection,
                selection,
                selectionArgs,
                sortOrder
        )) {
            if (cursor == null || !cursor.moveToFirst()) {
                return null;
            }
            int idIndex = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID);
            long id = cursor.getLong(idIndex);
            return Uri.withAppendedPath(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, Long.toString(id));
        }
    }

    private int parseIntArg(String value) {
        if (value == null || value.trim().isEmpty()) {
            return 0;
        }
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            return 0;
        }
    }
}
