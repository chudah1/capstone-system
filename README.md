# AI- SYSTEM RECORDING APPLICATION

This project is a an implementation of a system to show AI can be used to record sales passively in "corner shops" in Ghana
## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [Environment Variables](#environment-variables)
  - [Firestore Configuration](#firestore-configuration)
  - [Flutter Application](#flutter-application)
- [Running the Flask Application](#running-the-flask-application)
- [Running the Flutter Application](#running-the-flutter-application)
- [Project Structure](#project-structure)

## Prerequisites

Before running the project, ensure you have the following installed:

- [Python 3.8+](https://www.python.org/downloads/)
- [Flutter](https://flutter.dev/docs/get-started/install)

## Setup

### Environment Variables

1. Create a `.env` file in the root directory of the project.
2. Add the following environment variables:

    ```env
    host_machine= "The ip address of a vm for hosting all the files apart from main.dart"
    API_KEY= "An api key from OPEN AI"
    FIREBASE_KEY= "path to the service credentials json file from Firebase"
    ```

    - **host_machine**: The ip address of a vm for hosting all the files apart from main.dart.
    - **API_KEY**: An api key from OPEN AI
    - **FIREBASE_KEY**: The path to your Firebase key JSON file.

### Firestore Configuration

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new Firebase project or use an existing one.
3. Download the service credentials file from the Firebase project settings and place it in the root directory of your project.
4. Ensure the `FIREBASE_KEY` in the `.env` file points to this `key.json` file.

### Flutter Application

1. Create a new Flutter project by running ```flutter create <name of project> ``` either in this directory or another.
2. Ensure `main.dart` exists and is set up to run your Flutter application.
3. Navigate to the lib folder and replace the `main.dart` file with the `main.dart` file from this project.
4. Run `flutter pub get` to install dependencies.

## Running the Flask Application

1. Install the required Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the Flask application:

    ```bash
    flask run
    ```

   The application will start, and you can access it at `http://127.0.0.1:5000/`.

## Running the Flutter Application

1. Navigate to the `flutter_app` directory:

    ```bash
    cd <name of flutter project>
    ```

2. Run the Flutter application:

    ```bash
    flutter run
    ```

   The Flutter app will start on the connected device or emulator.

## Project Structure

```bash
├── .env                # Environment variables
├── main.py              # Flask application entry point
├── requirements.txt    # Python dependencies
├── key.json            # Firestore service account key
├── flutter_app/        # Flutter application directory
│   ├── main.dart       # Flutter application entry point
│   └── ...             # Other Flutter files and directories
└── README.md           # This file
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
