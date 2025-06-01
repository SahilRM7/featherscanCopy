from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
from django.conf import settings
import json
import librosa
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.layers import Lambda
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from keras._tf_keras.keras.applications.resnet50 import preprocess_input
import keras._tf_keras.keras.backend as K

from bird.models import FAQ
from scan.models import Bird

# ====== Custom Metrics ======
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precisions = precision(y_true, y_pred)
    recalls = recall(y_true, y_pred)
    return 2*((precisions*recalls)/(precisions+recalls+K.epsilon()))

# ====== Load model and labels once ======
MODEL_PATH = os.path.join(settings.BASE_DIR, 'scan', 'final_model_transfer_learning.keras')
LABELS_PATH = os.path.join(settings.BASE_DIR, 'scan', 'class_names.txt')

model = load_model(MODEL_PATH, custom_objects={'f1': f1, 'precision': precision, 'recall': recall})

with open(LABELS_PATH, 'r') as file:
    class_labels = [line.strip() for line in file.readlines()]

# ====== View Function ======
def image_scan(request):
    faqs = FAQ.objects.all()[:4]
    prediction = None
    confidence = None

    if request.method == 'POST' and request.FILES.get('bird_image'):
        image_file = request.FILES['bird_image']
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        file_path = fs.path(filename)

        try:
            img = load_img(file_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            preds = model.predict(img_array)
            predicted_index = np.argmax(preds)
            prediction = class_labels[predicted_index]
            confidence = round(float(np.max(preds)) * 100, 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"
            confidence = None

        # Optional: Remove the uploaded image file after prediction
        if os.path.exists(file_path):
            os.remove(file_path)

    return render(request, 'imgscan.html', {
        'faqs': faqs,
        'prediction': prediction,
        'confidence': confidence,
    })


def audio_scan(request):
    result = None
    faqs = FAQ.objects.all()[:4]
    bird_info = None
    uploaded_audio_url = None

    if request.method == 'POST' and request.FILES.get('audio'):
        audio_file = request.FILES['audio']

        fs = FileSystemStorage()
        filename = fs.save(audio_file.name, audio_file)
        file_path = fs.path(filename)
        uploaded_audio_url = fs.url(filename)

        try:
            # Load label mapping
            with open(os.path.join(settings.BASE_DIR, 'scan', 'prediction.json'), 'r') as f:
                prediction_dict = json.load(f)

            # Preprocess the audio
            audio, sample_rate = librosa.load(file_path)
            mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_features = np.mean(mfccs_features, axis=1)
            mfccs_features = mfccs_features.reshape(1, -1, 1)

            mfccs_tensors = tf.convert_to_tensor(mfccs_features, dtype=tf.float32)

            # Load model and predict
            model = tf.keras.models.load_model(os.path.join(settings.BASE_DIR, 'scan', 'model.h5'))
            prediction = model.predict(mfccs_tensors)

            predicted_label = np.argmax(prediction)
            predicted_class = prediction_dict.get(str(predicted_label), "Unknown")
            confidence = round(float(np.max(prediction)) * 100, 2)

            result = f"{predicted_class} ({confidence}% confidence)"

            # Remove '_sound' and fetch bird info from DB
            clean_name = predicted_class.replace('_sound', '')
            bird_info = Bird.objects.filter(name__iexact=clean_name).first()

        except Exception as e:
            result = f"Error processing audio: {str(e)}"

        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)

    return render(request, 'audscan.html', {
        'result': result,
        'faqs': faqs,
        'bird_info': bird_info,
        'audio_url': uploaded_audio_url,
    })