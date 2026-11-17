# MicroFly

Steps:
1. Create a GitHub repo and push this folder to `main`.
2. Actions -> "Build APK" -> download `microfly-debug` / `microfly-release` from Artifacts.
3. Optional signing secrets: KEYSTORE_BASE64, KEYSTORE_PASSWORD, KEY_ALIAS, KEY_PASSWORD.

The workflow installs Gradle 8.9 itself, so no gradlew / wrapper files are needed.
`app/src/main/assets/models/fly.glb` is a simple procedural fly (nodes Wing_L / Wing_R, +Z forward).
Replace it with a detailed model if you like (keep the wing node names).
