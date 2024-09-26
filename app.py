import os
from flask import Flask, request, render_template
import numpy as np
import librosa
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the saved model
model = load_model("audio_classifier.h5")

# Define parameters
SAMPLE_RATE = 44100  # For example, 44.1 kHz
MAX_TIME_STEPS = 109
N_MELS = 128
DURATION = 10  # Duration of audio clips in seconds

def preprocess_audio(file_path):
    # Load audio file using librosa
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    # Extract Mel spectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Ensure all spectrograms have the same width (time steps)
    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

    # Add channel dimension
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)

    return mel_spectrogram

def predict_single_audio(file_path, model):
    # Preprocess the audio
    mel_spectrogram = preprocess_audio(file_path)

    # Reshape the input data to match the model's expected input shape
    X_test = np.expand_dims(mel_spectrogram, axis=0)  # Add batch dimension

    # Make prediction
    y_pred = model.predict(X_test)

    # Define a threshold for classifying the sample
    threshold = 0.5

    # Classify the sample based on the threshold
    class_labels = ["deepfake", "not deepfake"]
    predicted_label = class_labels[int(y_pred[0, 1] > threshold)]

    return predicted_label

@app.route("/audio", methods=["GET", "POST"])
def audio_detector():
    if request.method == "POST":
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message="No file part")
        
        file = request.files['file']
        
        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', message="No selected file")

        # If file exists and is allowed
        if file:
            # Save the uploaded audio file temporarily
            file_path = "temp_audio.wav"
            file.save(file_path)

            # Predict whether the audio is deepfake or not
            predicted_label = predict_single_audio(file_path, model)

            # Remove the temporary audio file
            os.remove(file_path)

            return render_template('index.html', message="Prediction: " + predicted_label)

    return render_template('index.html', message="Upload an audio file")

if __name__ == "__main__":
    app.run(debug=True)
