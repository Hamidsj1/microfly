plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.example.microfly"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.microfly"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        // Filament ships native libs per ABI; keep only what phones need.
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") }
    }

    // Signing values are injected from CI secrets via environment variables.
    // If they are missing (local build / fork PR) release falls back to debug signing.
    val ksPath: String? = System.getenv("KEYSTORE_PATH")
    signingConfigs {
        create("release") {
            if (ksPath != null) {
                storeFile = file(ksPath)
                storePassword = System.getenv("KEYSTORE_PASSWORD")
                keyAlias = System.getenv("KEY_ALIAS")
                keyPassword = System.getenv("KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = if (ksPath != null) signingConfigs.getByName("release")
                            else signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }

    // Never compress GLB / Filament material / KTX files: they are memory-mapped.
    androidResources { noCompress += listOf("glb", "gltf", "filamat", "ktx", "bin") }
}

dependencies {
    implementation("io.github.sceneview:sceneview:2.2.1")   // Filament-based, replaces Sceneform

    implementation(platform("androidx.compose:compose-bom:2024.10.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.core:core-ktx:1.13.1")
}
