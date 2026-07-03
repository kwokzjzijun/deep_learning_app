plugins {
    alias(libs.plugins.android.application)
    id("com.chaquo.python")
}

android {
    namespace = "com.example.deep_learning_app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.deep_learning_app"
        minSdk = 27
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    buildFeatures {
        viewBinding = true
    }

    packaging {
        jniLibs {
            // Chaquopy + PyTorch Lite can both include libc++_shared.so.
            pickFirsts += setOf(
                "lib/arm64-v8a/libc++_shared.so",
                "lib/armeabi-v7a/libc++_shared.so",
                "lib/x86/libc++_shared.so",
                "lib/x86_64/libc++_shared.so"
            )
        }
    }

}

chaquopy {
    defaultConfig {
        version = "3.10"
        pip {
            install("numpy==1.26.2")
            install("Pillow==11.0.0")
        }
    }
}

dependencies {

    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.constraintlayout)
    implementation(libs.navigation.fragment)
    implementation(libs.navigation.ui)
    
    // PyTorch Mobile dependencies
    implementation("org.pytorch:pytorch_android_lite:2.1.0")
    implementation("org.pytorch:pytorch_android_torchvision_lite:2.1.0")
    
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
}
