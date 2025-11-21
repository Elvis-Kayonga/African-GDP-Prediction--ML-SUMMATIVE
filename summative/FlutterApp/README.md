# African GDP Growth Predictor - Flutter App

A mobile application to predict GDP growth rates for African countries based on economic indicators.

## Features

- 📊 Predict GDP growth rate using 8 economic indicators
- 🌍 Supports 18 African countries
- ✅ Input validation with range constraints
- 📱 Clean, organized, and user-friendly interface
- 🔄 Real-time API integration

## Setup Instructions

### Prerequisites

1. **Flutter SDK** installed (version 3.0.0 or higher)
   - Download from: https://flutter.dev/docs/get-started/install
   - Verify installation: `flutter doctor`

2. **Android Studio** or **VS Code** with Flutter extension

3. **Android Emulator** or **Physical Device** for testing

### Installation Steps

1. **Navigate to the Flutter app directory:**
   ```bash
   cd summative/FlutterApp
   ```

2. **Install dependencies:**
   ```bash
   flutter pub get
   ```

3. **Update API URL:**
   - Open `lib/main.dart`
   - Find line 56: `final String apiUrl = 'YOUR_RENDER_URL_HERE/predict';`
   - Replace `YOUR_RENDER_URL_HERE` with your actual Render API URL
   - Example: `https://your-app-name.onrender.com/predict`

4. **Run the app:**
   ```bash
   flutter run
   ```

### For Android

```bash
flutter run -d android
```

### For iOS (Mac only)

```bash
flutter run -d ios
```

### For Web (Testing only)

```bash
flutter run -d chrome
```

## App Structure

```
FlutterApp/
├── lib/
│   └── main.dart          # Main application code
├── pubspec.yaml           # Dependencies
└── README.md             # This file
```

## Input Fields

The app requires the following 8 inputs:

1. **Country** - Select from dropdown (18 African countries)
2. **Year** - Integer (2000-2050)
3. **Inflation Rate** - Float (0-100%)
4. **Unemployment Rate** - Float (0-100%)
5. **FDI** - Float (0-50000 million USD)
6. **Trade Balance** - Float (-50000 to 50000 million USD)
7. **Government Debt** - Float (0-200% of GDP)
8. **Internet Penetration** - Float (0-100%)

## Example Input

```
Country: Nigeria
Year: 2024
Inflation Rate: 12.5
Unemployment Rate: 18.0
FDI (Millions USD): 3500
Trade Balance (Millions USD): -800
Government Debt (% GDP): 38.0
Internet Penetration (%): 55.0
```

**Expected Output:** Predicted GDP Growth Rate: ~2.34%

## Troubleshooting

### API Connection Error

1. Verify API URL is correct in `main.dart`
2. Ensure API server is running on Render
3. Check internet connection
4. Test API endpoint in browser: `https://your-api-url.com/health`

### Flutter Issues

```bash
# Clean build
flutter clean

# Get dependencies again
flutter pub get

# Check Flutter installation
flutter doctor
```

### Android Emulator Not Detected

```bash
# List devices
flutter devices

# Start emulator
flutter emulators --launch <emulator_id>
```

## Building APK (Android)

```bash
flutter build apk --release
```

APK location: `build/app/outputs/flutter-apk/app-release.apk`

## Building for iOS

```bash
flutter build ios --release
```

## Dependencies

- `flutter`: Framework
- `http`: HTTP requests to API
- `cupertino_icons`: iOS-style icons

## API Integration

The app communicates with the FastAPI backend using HTTP POST requests:

- **Endpoint:** `POST /predict`
- **Content-Type:** `application/json`
- **Response:** JSON with predicted GDP growth rate

## Author

Created for ML Summative Assignment - African Finance Mission

## License

Educational project - Free to use
