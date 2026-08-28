# AndroidChat

Minimal local-first PySide6 chat client for AgentRouter/Anthropic-compatible models.

## Android build

GitHub Actions builds an ARM64 debug APK automatically on pushes to `main`, and the workflow can also be started manually from the Actions tab.

The APK does **not** contain an API key. On first launch it asks for the AgentRouter API key and stores it in the app-private data directory on the device.

Persistent chat state and files are also stored in the app-private data directory.

### Downloading the APK

Open **Actions → Build Android APK → latest successful run → Artifacts → AndroidChat-apk**.

The Android packaging is based on Qt for Python's `pyside6-android-deploy` tool and targets `arm64-v8a`.
