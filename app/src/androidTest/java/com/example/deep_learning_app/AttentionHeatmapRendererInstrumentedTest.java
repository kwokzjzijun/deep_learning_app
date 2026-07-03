package com.example.deep_learning_app;

import android.graphics.Bitmap;
import android.graphics.Color;

import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

@RunWith(AndroidJUnit4.class)
public class AttentionHeatmapRendererInstrumentedTest {
    @Test
    public void renderCreatesHeatmapPreviewWithoutChangingSourceDimensions() {
        Bitmap source = Bitmap.createBitmap(32, 32, Bitmap.Config.ARGB_8888);
        source.eraseColor(Color.BLACK);
        for (int y = 10; y < 22; y++) {
            for (int x = 8; x < 24; x++) {
                source.setPixel(x, y, Color.rgb(90, 190, 70));
            }
        }

        Bitmap heatmap = AttentionHeatmapRenderer.render(source);

        assertNotNull(heatmap);
        assertEquals(source.getWidth(), heatmap.getWidth());
        assertEquals(source.getHeight(), heatmap.getHeight());
        assertNotEquals(source.getPixel(16, 16), heatmap.getPixel(16, 16));
        assertTrue(Color.red(heatmap.getPixel(16, 16)) > Color.red(heatmap.getPixel(1, 1)));
    }
}
