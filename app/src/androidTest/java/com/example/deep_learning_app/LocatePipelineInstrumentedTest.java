package com.example.deep_learning_app;

import android.content.Context;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.junit.Test;
import org.junit.runner.RunWith;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

@RunWith(AndroidJUnit4.class)
public class LocatePipelineInstrumentedTest {
    private static final String TAG = "LocatePipelineTest";
    private static final String DEFAULT_LOCATE_URI =
            "content://media/external/images/media/1000010739";

    @Test
    public void locateMediaUriProducesSavedCircleRois() throws Exception {
        Bundle args = InstrumentationRegistry.getArguments();
        String uriArg = args.getString("locateUri", DEFAULT_LOCATE_URI);
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();

        Locate locate = new Locate(context);
        Locate.LocateResult result = locate.run(Uri.parse(uriArg));

        Log.i(TAG, "uri=" + uriArg + ", summary=" + result.getSummary());
        assertTrue("Expected at least one refined circle ROI", result.getRoiCount() > 0);
        assertTrue("Expected at least one saved refined circle ROI", result.getSavedCount() > 0);
        assertNotNull("Expected random circle preview bitmap", result.getRandomCircleBitmap());
        assertTrue(
                "Expected square random circle preview bitmap",
                result.getRandomCircleBitmap().getWidth() == result.getRandomCircleBitmap().getHeight()
        );
    }
}
