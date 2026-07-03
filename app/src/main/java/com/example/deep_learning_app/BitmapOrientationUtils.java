package com.example.deep_learning_app;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;

final class BitmapOrientationUtils {
    private static final String TAG = "BitmapOrientationUtils";

    private BitmapOrientationUtils() {
    }

    static Bitmap decodeBitmap(Context context, Uri imageUri, BitmapFactory.Options options) throws IOException {
        int orientation = readExifOrientation(context, imageUri);
        try (InputStream is = context.getContentResolver().openInputStream(imageUri)) {
            if (is == null) {
                throw new IOException("无法打开图片流");
            }
            Bitmap bitmap = BitmapFactory.decodeStream(is, null, options);
            if (bitmap == null) {
                throw new IOException("图片解码失败");
            }
            return applyExifOrientation(bitmap, orientation);
        }
    }

    static int readExifOrientation(Context context, Uri imageUri) {
        try (InputStream is = context.getContentResolver().openInputStream(imageUri)) {
            if (is == null) {
                return ExifInterface.ORIENTATION_NORMAL;
            }
            ExifInterface exif = new ExifInterface(is);
            return exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
            );
        } catch (Exception e) {
            Log.w(TAG, "Failed to read EXIF orientation, using normal orientation", e);
            return ExifInterface.ORIENTATION_NORMAL;
        }
    }

    static Bitmap applyExifOrientation(Bitmap bitmap, int orientation) {
        Matrix matrix = matrixForOrientation(orientation);
        if (matrix == null) {
            return bitmap;
        }

        Bitmap transformed = Bitmap.createBitmap(
                bitmap,
                0,
                0,
                bitmap.getWidth(),
                bitmap.getHeight(),
                matrix,
                true
        );
        if (transformed != bitmap) {
            bitmap.recycle();
        }
        return transformed;
    }

    private static Matrix matrixForOrientation(int orientation) {
        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                matrix.setScale(-1, 1);
                return matrix;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.setRotate(180);
                return matrix;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                matrix.setRotate(180);
                matrix.postScale(-1, 1);
                return matrix;
            case ExifInterface.ORIENTATION_TRANSPOSE:
                matrix.setRotate(90);
                matrix.postScale(-1, 1);
                return matrix;
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.setRotate(90);
                return matrix;
            case ExifInterface.ORIENTATION_TRANSVERSE:
                matrix.setRotate(-90);
                matrix.postScale(-1, 1);
                return matrix;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.setRotate(-90);
                return matrix;
            case ExifInterface.ORIENTATION_NORMAL:
            case ExifInterface.ORIENTATION_UNDEFINED:
            default:
                return null;
        }
    }
}
