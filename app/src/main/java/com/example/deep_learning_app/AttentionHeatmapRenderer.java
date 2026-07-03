package com.example.deep_learning_app;

import android.graphics.Bitmap;
import android.graphics.Color;

final class AttentionHeatmapRenderer {
    private static final int MAX_PREVIEW_EDGE = 720;

    private AttentionHeatmapRenderer() {
    }

    static Bitmap render(Bitmap sourceBitmap) {
        if (sourceBitmap == null || sourceBitmap.getWidth() <= 0 || sourceBitmap.getHeight() <= 0) {
            return null;
        }

        Bitmap source = createPreviewSource(sourceBitmap);
        int width = source.getWidth();
        int height = source.getHeight();
        int[] sourcePixels = new int[width * height];
        int[] outputPixels = new int[sourcePixels.length];
        float[] attentionScores = new float[sourcePixels.length];
        source.getPixels(sourcePixels, 0, width, 0, 0, width, height);

        float maxScore = 0.0f;
        float centerX = (width - 1) / 2.0f;
        float centerY = (height - 1) / 2.0f;
        float maxDistance = Math.max(1.0f, (float) Math.sqrt(centerX * centerX + centerY * centerY));

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                int pixel = sourcePixels[index];
                int alpha = Color.alpha(pixel);
                if (alpha == 0) {
                    continue;
                }

                int red = Color.red(pixel);
                int green = Color.green(pixel);
                int blue = Color.blue(pixel);
                int maxChannel = Math.max(red, Math.max(green, blue));
                int minChannel = Math.min(red, Math.min(green, blue));
                float brightness = maxChannel / 255.0f;
                if (brightness < 0.04f) {
                    continue;
                }

                float saturation = maxChannel == 0 ? 0.0f : (maxChannel - minChannel) / (float) maxChannel;
                float warmSignal = Math.max(0, red - Math.min(green, blue)) / 255.0f;
                float stripColorSignal = Math.max(0, green - blue) / 255.0f;
                float dx = x - centerX;
                float dy = y - centerY;
                float distance = (float) Math.sqrt(dx * dx + dy * dy);
                float centerBias = 0.55f + 0.45f * (1.0f - Math.min(1.0f, distance / maxDistance));

                float score = (
                        saturation * 0.68f
                                + warmSignal * 0.22f
                                + stripColorSignal * 0.10f
                ) * (0.35f + 0.65f * (float) Math.sqrt(brightness)) * centerBias;
                attentionScores[index] = score;
                if (score > maxScore) {
                    maxScore = score;
                }
            }
        }

        if (maxScore < 0.01f) {
            maxScore = fillBrightnessFallback(sourcePixels, attentionScores, width, height);
        }

        for (int i = 0; i < sourcePixels.length; i++) {
            int pixel = sourcePixels[i];
            float normalized = maxScore <= 0.0f ? 0.0f : clamp(attentionScores[i] / maxScore);
            int heatColor = heatColor(normalized);
            float overlayAlpha = normalized < 0.05f ? 0.08f : 0.22f + normalized * 0.58f;
            outputPixels[i] = blendDimmedBaseWithHeat(pixel, heatColor, overlayAlpha);
        }

        Bitmap output = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        output.setPixels(outputPixels, 0, width, 0, 0, width, height);
        return output;
    }

    private static Bitmap createPreviewSource(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int maxEdge = Math.max(width, height);
        if (maxEdge > MAX_PREVIEW_EDGE) {
            float scale = MAX_PREVIEW_EDGE / (float) maxEdge;
            int targetWidth = Math.max(1, Math.round(width * scale));
            int targetHeight = Math.max(1, Math.round(height * scale));
            return Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
                    .copy(Bitmap.Config.ARGB_8888, false);
        }
        return bitmap.copy(Bitmap.Config.ARGB_8888, false);
    }

    private static float fillBrightnessFallback(int[] pixels, float[] scores, int width, int height) {
        float maxScore = 0.0f;
        float centerX = (width - 1) / 2.0f;
        float centerY = (height - 1) / 2.0f;
        float maxDistance = Math.max(1.0f, (float) Math.sqrt(centerX * centerX + centerY * centerY));

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                int pixel = pixels[index];
                int maxChannel = Math.max(Color.red(pixel), Math.max(Color.green(pixel), Color.blue(pixel)));
                float brightness = maxChannel / 255.0f;
                float dx = x - centerX;
                float dy = y - centerY;
                float distance = (float) Math.sqrt(dx * dx + dy * dy);
                float centerBias = 1.0f - Math.min(1.0f, distance / maxDistance);
                float score = brightness * (0.35f + centerBias * 0.65f);
                scores[index] = score;
                if (score > maxScore) {
                    maxScore = score;
                }
            }
        }
        return maxScore;
    }

    private static int blendDimmedBaseWithHeat(int sourcePixel, int heatColor, float heatAlpha) {
        int alpha = Color.alpha(sourcePixel);
        int baseRed = Math.round(Color.red(sourcePixel) * 0.62f);
        int baseGreen = Math.round(Color.green(sourcePixel) * 0.62f);
        int baseBlue = Math.round(Color.blue(sourcePixel) * 0.62f);

        int red = blendChannel(baseRed, Color.red(heatColor), heatAlpha);
        int green = blendChannel(baseGreen, Color.green(heatColor), heatAlpha);
        int blue = blendChannel(baseBlue, Color.blue(heatColor), heatAlpha);
        return Color.argb(alpha, red, green, blue);
    }

    private static int blendChannel(int base, int overlay, float overlayAlpha) {
        return Math.round(base * (1.0f - overlayAlpha) + overlay * overlayAlpha);
    }

    private static int heatColor(float value) {
        float clamped = clamp(value);
        if (clamped < 0.33f) {
            float t = clamped / 0.33f;
            return interpolateColor(40, 80, 210, 0, 210, 255, t);
        }
        if (clamped < 0.66f) {
            float t = (clamped - 0.33f) / 0.33f;
            return interpolateColor(0, 210, 255, 255, 232, 0, t);
        }
        float t = (clamped - 0.66f) / 0.34f;
        return interpolateColor(255, 232, 0, 230, 38, 38, t);
    }

    private static int interpolateColor(
            int startRed,
            int startGreen,
            int startBlue,
            int endRed,
            int endGreen,
            int endBlue,
            float value
    ) {
        float clamped = clamp(value);
        int red = Math.round(startRed + (endRed - startRed) * clamped);
        int green = Math.round(startGreen + (endGreen - startGreen) * clamped);
        int blue = Math.round(startBlue + (endBlue - startBlue) * clamped);
        return Color.rgb(red, green, blue);
    }

    private static float clamp(float value) {
        return Math.max(0.0f, Math.min(1.0f, value));
    }
}
