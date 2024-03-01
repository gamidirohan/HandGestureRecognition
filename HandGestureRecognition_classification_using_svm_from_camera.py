import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import tkinter as tk
from tkinter import messagebox
import pyttsx3

# Function to perform object detection on the frame
def perform_object_detection(frame, svm_model, x_scaler):
    # Preprocess the frame for object detection (adjust input shape accordingly)
    img = cv2.resize(frame, (28, 28))
    img_array = x_scaler.transform(img.reshape(1, -1))

    # Perform object detection on the frame
    predicted_category = svm_model.predict(img_array)[0]
    return predicted_category

# Function to start real-time sign language detection
def start_real_time_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame in the GUI
        cv2.imshow('Live Sign Language Detection', frame)

        # Perform object detection on the frame
        predicted_category = perform_object_detection(frame, svm_model, x_scaler)

        # Speak the predicted category
        text_to_speech(f"Predicted sign language: {Categories[predicted_category]}")

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit the loop
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to start live sign language recognition with GUI
def start_recognition_gui():
    root = tk.Tk()
    root.title("Sign Language Recognition")

    # Button to start recognition
    start_button = tk.Button(root, text="Start Recognition", command=start_real_time_detection)
    start_button.pack(pady=20)

    # Close the GUI on window close
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    
    # Start the GUI event loop
    root.mainloop()

# Function for text-to-speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Paths
train_path = "D:/College/Sem_3/Fundamentals of AI/Project FILES/SVM Final Project/sign_mnist_train.csv"
test_path = "D:/College/Sem_3/Fundamentals of AI/Project FILES/SVM Final Project/sign_mnist_test.csv"
model_path = "svm_model.pkl"

# Load the datasets
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Separate features (pixels) and labels
X_train = train_data.iloc[:, 1:].values  # Features (pixel values)
y_train = train_data.iloc[:, 0].values   # Labels

X_test = test_data.iloc[:, 1:].values   # Features (pixel values)
y_test = test_data.iloc[:, 0].values    # Labels

# Standardize the features (mean=0 and variance=1) for training set
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)

# Use the same scaling for the test set
X_test_scaled = x_scaler.transform(X_test)

# Exclude J and Z from Categories list
Categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c']

# Check if the model file exists
if os.path.exists(model_path):
    # Load the pre-trained model
    with open(model_path, 'rb') as model_file:
        svm_model = pickle.load(model_file)
else:
    # Train the SVM model
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # Save the trained model to a file
    with open(model_path, 'wb') as model_file:
        pickle.dump(svm_model, model_file)

# Start the GUI for live recognition
start_recognition_gui()
