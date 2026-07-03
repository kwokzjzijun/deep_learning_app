package com.example.deep_learning_app;

import android.Manifest;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    private static final int PICK_IMAGE_FOR_LOCATE_REQUEST = 1;
    private static final int PICK_IMAGE_FOR_PREDICT_REQUEST = 2;
    private static final int PERMISSION_REQUEST_CODE = 100;
    private static final double PPM_TO_UMOL_L = 5.945d;
    private static final double FEMALE_LOW_UMOL_L = 89.0d;
    private static final double FEMALE_HIGH_UMOL_L = 357.0d;
    private static final double MALE_LOW_UMOL_L = 150.0d;
    private static final double MALE_HIGH_UMOL_L = 416.0d;
    private static final String MODEL_DISPLAY_NAME = "ConvNeXtCCAM";
    private static final String MODEL_VERSION = "v0612 mobile";


    private ImageView imageView1;
    private ImageView imageView2;
    private ImageView imageView3;
    private ScrollView scrollViewClinicalReport;
    private TextView textViewClinicalReport;
    private TextView textViewOutput;
    private View progressPanel;
    private TextView textViewProgressStatus;
    private Button buttonLocateGallery;
    private Button buttonLocate;
    private Button buttonGallery;
    private Button buttonProcess;
    private Button buttonReset;

    private Uri locateImageUri;
    private Uri predictImageUri;

    private UricAcidProcessor processor;
    private Locate locateProcessor;
    private String activeRecordId = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initViews();
        setupButtonListeners();
        checkAndRequestPermissions();
    }

    private void initViews() {
        imageView1 = findViewById(R.id.imageView1);
        imageView2 = findViewById(R.id.imageView2);
        imageView3 = findViewById(R.id.imageView3);
        scrollViewClinicalReport = findViewById(R.id.scrollViewClinicalReport);
        textViewClinicalReport = findViewById(R.id.textViewClinicalReport);
        textViewOutput = findViewById(R.id.textViewOutput);
        progressPanel = findViewById(R.id.progressPanel);
        textViewProgressStatus = findViewById(R.id.textViewProgressStatus);

        buttonLocateGallery = findViewById(R.id.buttonLocateGallery);
        buttonLocate = findViewById(R.id.buttonLocate);
        buttonGallery = findViewById(R.id.buttonGallery);
        buttonProcess = findViewById(R.id.buttonProcess);
        buttonReset = findViewById(R.id.buttonReset);

        buttonLocate.setEnabled(false);
        buttonProcess.setEnabled(false);
        buttonReset.setEnabled(false);
    }

    private void setupButtonListeners() {
        buttonLocateGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openGallery(PICK_IMAGE_FOR_LOCATE_REQUEST);
            }
        });

        buttonLocate.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                runLocate();
            }
        });

        buttonGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openGallery(PICK_IMAGE_FOR_PREDICT_REQUEST);
            }
        });

        buttonProcess.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                processImage();
            }
        });
        
        buttonReset.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                resetInterface();
            }
        });
    }

    private void checkAndRequestPermissions() {
        List<String> permissions = new ArrayList<>();

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES)
                    != PackageManager.PERMISSION_GRANTED) {
                permissions.add(Manifest.permission.READ_MEDIA_IMAGES);
            }
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                permissions.add(Manifest.permission.READ_EXTERNAL_STORAGE);
            }
        }

        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                permissions.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
            }
        }

        if (!permissions.isEmpty()) {
            ActivityCompat.requestPermissions(this, permissions.toArray(new String[0]), PERMISSION_REQUEST_CODE);
        }
    }

    private void openGallery(int requestCode) {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        intent.setType("image/*");
        try {
            startActivityForResult(intent, requestCode);
        } catch (ActivityNotFoundException e) {
            Log.e(TAG, "No gallery app available", e);
            Toast.makeText(this, "未找到可用相册应用", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode != RESULT_OK || data == null || data.getData() == null) {
            return;
        }

        Uri uri = data.getData();
        Bitmap previewBitmap = loadPreviewBitmap(uri);
        if (previewBitmap != null) {
            imageView1.setImageBitmap(previewBitmap);
        } else {
            imageView1.setImageURI(uri);
        }
        buttonReset.setEnabled(true);
        Log.i(TAG, "Image selected, requestCode=" + requestCode + ", uri=" + uri);

        if (requestCode == PICK_IMAGE_FOR_LOCATE_REQUEST) {
            locateImageUri = uri;
            buttonLocate.setEnabled(true);
            showClinicalReport(false);
            textViewOutput.setText(R.string.status_locate_image_loaded);
        } else if (requestCode == PICK_IMAGE_FOR_PREDICT_REQUEST) {
            predictImageUri = uri;
            buttonProcess.setEnabled(true);
            imageView2.setImageResource(R.drawable.ic_launcher_background);
            setProgressVisible(false, "");
            setClinicalReportText("SalivaUAconcentration\nAwaiting analysis");
            showClinicalReport(true);
            textViewOutput.setText(R.string.status_predict_image_loaded);
        }
    }

    private Bitmap loadPreviewBitmap(Uri uri) {
        try {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            return BitmapOrientationUtils.decodeBitmap(this, uri, options);
        } catch (Exception e) {
            Log.w(TAG, "Preview decode failed, falling back to ImageView URI decode", e);
            return null;
        }
    }

    private void runLocate() {
        if (locateImageUri == null) {
            Toast.makeText(this, "请先使用“定位选图”选择图片", Toast.LENGTH_SHORT).show();
            return;
        }

        Log.i(TAG, "Locate started, uri=" + locateImageUri);
        textViewOutput.setText(R.string.status_locate_running);
        showClinicalReport(false);
        buttonLocate.setEnabled(false);

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    if (locateProcessor == null) {
                        locateProcessor = new Locate(MainActivity.this);
                    }
                    final Locate.LocateResult result = locateProcessor.run(locateImageUri);
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            if (result.getOverlayBitmap() != null) {
                                imageView2.setImageBitmap(result.getOverlayBitmap());
                            } else {
                                imageView2.setImageResource(R.drawable.ic_launcher_background);
                            }
                            if (result.getRandomCircleBitmap() != null) {
                                imageView3.setImageBitmap(result.getRandomCircleBitmap());
                            } else {
                                imageView3.setImageResource(R.drawable.ic_launcher_background);
                            }
                            showClinicalReport(false);
                            textViewOutput.setText(result.getSummary());
                            buttonLocate.setEnabled(true);
                            Log.i(TAG, "Locate finished, roi=" + result.getRoiCount() + ", saved=" + result.getSavedCount());
                        }
                    });
                } catch (Exception e) {
                    Log.e(TAG, "Locate failed", e);
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            textViewOutput.setText(getString(R.string.status_locate_failed, e.getMessage()));
                            showClinicalReport(false);
                            buttonLocate.setEnabled(true);
                        }
                    });
                }
            }
        }).start();
    }

    private void processImage() {
        if (predictImageUri == null) {
            Toast.makeText(this, "请先点击“选择图片”选择用于预测的图像", Toast.LENGTH_SHORT).show();
            return;
        }

        activeRecordId = createRecordId();
        setProgressVisible(true, "Running ConvNeXtCCAM inference...");
        setClinicalReportText("SalivaUAconcentration\nCalculating...");
        showClinicalReport(true);
        textViewOutput.setText(formatPendingModelSummary());
        buttonProcess.setEnabled(false);

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    if (processor == null) {
                        processor = new UricAcidProcessor(MainActivity.this);
                    }
                    final UricAcidProcessor.PredictionResult result = processor.predictUricAcid(predictImageUri);
                    final Bitmap attentionBitmap = renderAttentionPreview(result.getVisualBitmap());
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            setProgressVisible(false, "");
                            showClinicalReport(true);
                            textViewOutput.setText(formatModelSummary(result));
                            setClinicalReportText(formatClinicalReport(result));
                            if (attentionBitmap != null) {
                                imageView2.setImageBitmap(attentionBitmap);
                            } else if (result.getVisualBitmap() != null) {
                                imageView2.setImageBitmap(result.getVisualBitmap());
                            }
                            buttonProcess.setEnabled(true);
                        }
                    });
                } catch (Exception e) {
                    Log.e(TAG, "Predict failed", e);
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            setProgressVisible(false, "");
                            showClinicalReport(true);
                            textViewOutput.setText("处理失败: " + e.getMessage());
                            setClinicalReportText("SalivaUAconcentration\nAnalysis unavailable");
                            buttonProcess.setEnabled(true);
                        }
                    });
                }
            }
        }).start();
    }

    private void setProgressVisible(boolean visible, String message) {
        progressPanel.setVisibility(visible ? View.VISIBLE : View.GONE);
        imageView2.setVisibility(visible ? View.INVISIBLE : View.VISIBLE);
        if (message != null && !message.trim().isEmpty()) {
            textViewProgressStatus.setText(message);
        }
    }

    private void showClinicalReport(boolean visible) {
        scrollViewClinicalReport.setVisibility(visible ? View.VISIBLE : View.GONE);
        imageView3.setVisibility(visible ? View.GONE : View.VISIBLE);
    }

    private void setClinicalReportText(String reportText) {
        textViewClinicalReport.setText(reportText);
        scrollViewClinicalReport.post(new Runnable() {
            @Override
            public void run() {
                scrollViewClinicalReport.scrollTo(0, 0);
            }
        });
    }

    private Bitmap renderAttentionPreview(Bitmap sourceBitmap) {
        if (sourceBitmap == null) {
            return null;
        }
        try {
            return AttentionHeatmapRenderer.render(sourceBitmap);
        } catch (RuntimeException e) {
            Log.w(TAG, "Attention preview rendering failed; using source bitmap", e);
            return sourceBitmap;
        }
    }

    private String formatPendingModelSummary() {
        return "Model: " + MODEL_DISPLAY_NAME + "\n"
                + "Version: " + MODEL_VERSION + "\n"
                + "Record: " + activeRecordId + "\n"
                + "ParamsUsed: input=224x224, preprocess=PIL/ImageNet, unit=ppm*5.945";
    }

    private String formatModelSummary(UricAcidProcessor.PredictionResult result) {
        StringBuilder summary = new StringBuilder();
        summary.append("Model: ").append(MODEL_DISPLAY_NAME).append("\n");
        summary.append("Version: ").append(MODEL_VERSION).append("\n");
        summary.append("Record: ").append(activeRecordId).append("\n");
        summary.append("Asset: ").append(result.getModelAssetName()).append("\n");
        summary.append("Source: ").append(result.getSourceLabel());
        if (!result.getSourceName().isEmpty()) {
            summary.append(" | ").append(result.getSourceName());
        }
        summary.append("\nParamsUsed: input=224x224, preprocess=PIL/ImageNet, output=mean ROI ppm");
        return summary.toString();
    }

    private String formatClinicalReport(UricAcidProcessor.PredictionResult result) {
        List<Float> ppmValues = result.getPpmValues();
        if (ppmValues.isEmpty()) {
            return "SalivaUAconcentration\nNo numeric prediction available";
        }

        double meanPpm = calculateMeanPpm(ppmValues);
        double meanUmol = meanPpm * PPM_TO_UMOL_L;
        StringBuilder report = new StringBuilder();
        report.append("SalivaUAconcentration\n");
        report.append(String.format(Locale.US, "%.1f ppm  |  %.1f umol/L\n", meanPpm, meanUmol));
        report.append(String.format(Locale.US, "ROI count: %d\n", ppmValues.size()));
        if (ppmValues.size() > 1) {
            report.append("ROI ppm detail\n");
            for (int i = 0; i < ppmValues.size(); i++) {
                report.append(String.format(Locale.US, "ROI %03d: %.1f ppm\n", i + 1, ppmValues.get(i)));
            }
        }
        report.append("\nReference: Male 150-416 umol/L; Female 89-357 umol/L\n");
        report.append("Report: ").append(formatReferenceReport(meanUmol));
        return report.toString();
    }

    private double calculateMeanPpm(List<Float> ppmValues) {
        double total = 0.0d;
        for (Float value : ppmValues) {
            total += value;
        }
        return total / ppmValues.size();
    }

    private String formatReferenceReport(double uaUmolL) {
        if (uaUmolL < FEMALE_LOW_UMOL_L) {
            return "Needs attention - below adult reference range.";
        }
        if (uaUmolL < MALE_LOW_UMOL_L) {
            return "Keep monitoring - within female range, below male reference.";
        }
        if (uaUmolL <= FEMALE_HIGH_UMOL_L) {
            return "Keep it up - within adult reference overlap.";
        }
        if (uaUmolL <= MALE_HIGH_UMOL_L) {
            return "Needs attention - above female reference, within male range.";
        }
        return "Needs attention - above adult reference range.";
    }

    private String createRecordId() {
        return "UA-" + new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date());
    }

    private void resetInterface() {
        imageView1.setImageResource(R.drawable.ic_launcher_background);
        imageView2.setImageResource(R.drawable.ic_launcher_background);
        imageView3.setImageResource(R.drawable.ic_launcher_background);
        setProgressVisible(false, "");
        setClinicalReportText("SalivaUAconcentration\nAwaiting analysis");
        showClinicalReport(false);
        textViewOutput.setText(R.string.status_idle);

        buttonLocate.setEnabled(false);
        buttonProcess.setEnabled(false);
        buttonReset.setEnabled(false);

        locateImageUri = null;
        predictImageUri = null;
        activeRecordId = "";

        Toast.makeText(this, "已重置，请重新选择图片", Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == PERMISSION_REQUEST_CODE) {
            boolean allGranted = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }
            if (allGranted) {
                Toast.makeText(this, "权限已授予", Toast.LENGTH_SHORT).show();
            } else {
                showPermissionDeniedDialog();
            }
        }
    }

    private void showPermissionDeniedDialog() {
        new AlertDialog.Builder(this)
                .setTitle("需要存储权限")
                .setMessage("该应用需要图片读写权限来读取相册并保存 Locate 结果。请在设置中授予权限。")
                .setPositiveButton("去设置", (dialog, which) -> {
                    Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
                    Uri uri = Uri.fromParts("package", getPackageName(), null);
                    intent.setData(uri);
                    startActivity(intent);
                })
                .setNegativeButton("取消", (dialog, which) -> {
                    Toast.makeText(this, "没有权限将无法选择图片或保存 Locate 结果", Toast.LENGTH_SHORT).show();
                })
                .show();
    }
}
