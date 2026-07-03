package com.example.deep_learning_app;

import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.junit.Test;
import org.junit.runner.RunWith;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

@RunWith(AndroidJUnit4.class)
public class PredictionPipelineInstrumentedTest {
    private static final String TAG = "PredictionPipelineTest";
    private static final String DEFAULT_PREDICT_URIS =
            "content://media/external/images/media/1000010783,"
                    + "content://media/external/images/media/1000010784";

    @Test
    public void predictMediaUrisWithConvNext() {
        Bundle args = InstrumentationRegistry.getArguments();
        String uriArgs = args.getString("predictUris", DEFAULT_PREDICT_URIS);
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        UricAcidProcessor processor = new UricAcidProcessor(context);

        String[] uris = uriArgs.split(",");
        int tested = 0;
        for (String rawUri : uris) {
            String trimmed = rawUri.trim();
            if (trimmed.isEmpty()) {
                continue;
            }

            UricAcidProcessor.PredictionResult result =
                    processor.predictUricAcid(Uri.parse(trimmed));
            String text = result.getTextResult();
            Bitmap visual = result.getVisualBitmap();

            Log.i(TAG, "uri=" + trimmed + ", result=" + text.replace('\n', ' '));
            assertFalse("Prediction failed for " + trimmed, text.startsWith("错误"));
            assertNotNull("Expected visual bitmap for " + trimmed, visual);
            assertTrue("Expected positive image width", visual.getWidth() > 0);
            assertTrue("Expected positive image height", visual.getHeight() > 0);
            tested++;
        }

        assertTrue("Expected at least one prediction URI", tested > 0);
    }
}
