# Uric Acid Strip Analyzer Android App

An Android application for locating uric-acid test-strip regions of interest and running on-device concentration prediction with PyTorch Lite models.

The app is designed around a desktop-to-mobile computer-vision workflow:

1. Select a source image from the phone album.
2. Locate or load a matching region of interest.
3. Preprocess the ROI with the same normalization used by the desktop pipeline.
4. Run a bundled PyTorch Lite regression model.
5. Display the predicted uric-acid concentration in ppm.

This repository contains the Android app, model conversion/deployment helpers, embedded mobile model assets, and the small Python preprocessing modules used through Chaquopy.

## Status

- Android Java app with image selection, ROI localization, preprocessing, and prediction.
- PyTorch Lite regression and YOLO localization model assets are bundled under `app/src/main/assets/`.
- Known desktop ROI assets can be loaded through `PcRoiRepository` for desktop-aligned predictions.
- Unknown images fall back to the mobile YOLO localization path.

## Important Note

This project is a research and engineering prototype for uric-acid test-strip analysis. It is not a certified medical device and should not be used as the sole basis for diagnosis or treatment decisions.

## Repository Layout

```text
deep_learning_app/
├── app/
│   ├── build.gradle.kts
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── assets/
│       │   ├── uric_acid_current_mobile.ptl
│       │   ├── yolo_locate_current_mobile.ptl
│       │   ├── pc_roi_manifest.json
│       │   └── pc_rois/
│       ├── java/com/example/deep_learning_app/
│       │   ├── MainActivity.java
│       │   ├── WelcomeActivity.java
│       │   ├── Locate.java
│       │   ├── YoloLocateProcessor.java
│       │   ├── PcRoiRepository.java
│       │   ├── UricAcidProcessor.java
│       │   ├── AttentionHeatmapRenderer.java
│       │   └── BitmapOrientationUtils.java
│       ├── python/
│       │   ├── uric_preprocess.py
│       │   └── roi_refine.py
│       └── res/
├── gradle/
├── build.gradle.kts
├── settings.gradle.kts
├── convert_model_to_mobile.py
├── deploy.sh
├── deploy_updated_mobile_models.sh
├── DEPLOYMENT_GUIDE.md
├── ERROR_TROUBLESHOOTING.md
└── PROJECT_HANDOVER.md
```

## Requirements

- Android Studio with Android SDK 34.
- JDK compatible with the Android Gradle plugin used by this project.
- Python 3.10 for Chaquopy runtime compatibility.
- Git LFS for model files (`*.pt`, `*.pth`, `*.ptl`).

Install Git LFS before cloning if you want the bundled model assets:

```bash
git lfs install
```

## Build

From the repository root:

```bash
./gradlew :app:assembleDebug
```

For instrumented test APK generation:

```bash
./gradlew :app:assembleDebugAndroidTest
```

Android Studio can also open the project root directly.

## Model Assets

The application expects stable model aliases in `app/src/main/assets/`:

- `uric_acid_current_mobile.ptl`
- `yolo_locate_current_mobile.ptl`

Several historical or comparison model assets are also kept in the assets folder for reproducibility. Because some model files are larger than GitHub's normal 100 MB file limit, this repository uses Git LFS for `*.pt`, `*.pth`, and `*.ptl` files.

If model assets are missing after cloning, run:

```bash
git lfs pull
```

## Useful Scripts

- `convert_model_to_mobile.py`: convert desktop PyTorch checkpoints to mobile-compatible assets.
- `deploy.sh`: legacy one-command deployment helper.
- `deploy_updated_mobile_models.sh`: update selected regression or YOLO model assets.
- `check_16kb_pagesize.sh`: check native library page-size compatibility.
- `logcat_locate.sh`: collect Android logs related to localization.

## App Flow

1. Launch the app.
2. Select an image from the device album.
3. The app checks whether the source image has known desktop ROI assets.
4. If known, embedded PC ROI assets are used for desktop-aligned prediction.
5. If unknown, the mobile YOLO fallback attempts localization.
6. The selected or generated ROI is preprocessed and passed to the PyTorch Lite regression model.

## Development Notes

- `local.properties`, Gradle build outputs, IDE state, and Python cache files are intentionally ignored.
- Keep Android asset aliases stable when swapping models.
- Large model files should stay in Git LFS, not normal Git blobs.
- Rebuilding an APK is not enough for phone validation; install it on the device before testing behavior.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
