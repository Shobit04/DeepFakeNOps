import os
from flask import Flask, render_template, request, jsonify
import google.generativeai as palm
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
from pytube import YouTube

app = Flask(__name__, static_url_path='/static')

# Configure the AI API key
palm_api_key = "AIzaSyBp1wcksd55hbKK2lJtvBoNFWiaWND3QMU"
palm.configure(api_key=palm_api_key)

# Load the audio classification model
audio_model = load_model("audio_classifier.h5")
video_model = tf.keras.models.load_model('deepfakeworking.h5')

# Define parameters for audio preprocessing
SAMPLE_RATE = 44100
MAX_TIME_STEPS = 109
N_MELS = 128
DURATION = 10

def preprocess_audio(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)

    return mel_spectrogram

def predict_single_audio(file_path, model):
    mel_spectrogram = preprocess_audio(file_path)
    X_test = np.expand_dims(mel_spectrogram, axis=0)
    y_pred = model.predict(X_test)

    threshold = 0.5
    class_labels = ["deepfake", "not deepfake"]
    predicted_label = class_labels[int(y_pred[0, 1] > threshold)]

    return predicted_label

def extract_frames(video_path, interval=2):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frame_count = 0
    success = True
    output_frames = []
    while success:
        success, frame = cap.read()
        if frame_count % frame_interval == 0:
            # Append the frame to the output list
            output_frames.append(frame)
        frame_count += 1
    cap.release()
    return output_frames

def predict_frames(image):
    image = np.expand_dims(image, axis=0)
    image = image / 255
    y_pred = video_model.predict(image)
    return y_pred[0][0]

def predict_video(video_path):
    frames = extract_frames(video_path)
    predictions = []
    for frame in frames:
        frame = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_CUBIC)
        prediction = predict_frames(frame)
        predictions.append(prediction)
    avg_prediction = np.mean(predictions)
    return 'real' if avg_prediction >= 0.5 else 'fake'

@app.route('/')
def index():
    return render_template('New.html')

@app.route('/audio', methods=['GET', 'POST'])
def audio_detector():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message="No file part")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', message="No selected file")

        if file:
            file_path = "temp_audio.wav"
            file.save(file_path)
            predicted_label = predict_single_audio(file_path, audio_model)
            os.remove(file_path)
            return render_template('index.html', message="Prediction: " + predicted_label)

    return render_template('index.html')

@app.route('/youtube', methods=['GET', 'POST'])
def youtube():
    if request.method == 'POST':
        url_to_predict = request.form['url']
        prediction = predict_video(url_to_predict)
        return render_template('youtube.html', prediction=prediction)
    return render_template('youtube.html')

@app.route('/video', methods=['GET', 'POST'])
def video():
    if request.method == 'POST':
        if 'videoFile' not in request.files:
            return "No file part"
        
        video_file = request.files['videoFile']
        
        if video_file.filename == '':
            return "No selected file"
        
        if video_file:
            video_path = os.path.join('temp', video_file.filename)
            video_file.save(video_path)
            prediction = predict_video(video_path)
            os.remove(video_path)
            return prediction

    return render_template('video.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = palm.chat(messages=user_input)
    truncated_response = response.last[:100] if len(response.last) > 100 else response.last
    return jsonify({'response': truncated_response})

@app.route('/voice')
def voice():
    return render_template('voice.html')
@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)
@app.route('/FAQ', methods=['GET', 'POST'])
def responder():
    return render_template('FAQ.html')

if __name__ == '__main__':
    app.run(debug=True)
