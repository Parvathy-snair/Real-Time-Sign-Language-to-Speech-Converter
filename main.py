import cv2
import time
import numpy as np
import math
import os
import pyttsx3
import tkinter as tk
from PIL import Image, ImageTk
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Initialize text-to-speech
text_speech = pyttsx3.init()

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Load model and initialize variables
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = [
    "A","B","C","D","E","F","G","H","I","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","Y"
]

last_prediction = None
stable_count = 0
threshold = 5
last_print_time = time.time()
alphabet_delay = 1
word_delay = 3
predicted_word = ""

# Create GUI window
root = tk.Tk()
root.title("Sign Language Recognition")
root.geometry("800x600")
root.configure(bg="#f0f0f0")

title_label = tk.Label(
    root,
    text="Sign Language Predictor",
    font=("Arial",20,"bold"),
    bg="#f0f0f0",
    fg="#333"
)
title_label.pack(pady=20)

video_label = tk.Label(root)
video_label.pack(pady=20)

text_label = tk.Label(
    root,
    text="Detected Text:",
    font=("Arial",14),
    bg="#f0f0f0",
    fg="#555"
)
text_label.pack(pady=20)


def update_frame():
    global last_prediction, stable_count, last_print_time, predicted_word

    success, img = cap.read()
    imgOutput = img.copy()

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal+hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        current_prediction = labels[index]

        if current_prediction == last_prediction:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count == threshold and time.time() - last_print_time >= alphabet_delay:
            predicted_word += current_prediction
            text_speech.say(current_prediction)
            text_speech.runAndWait()

            text_label.config(text=f"Detected Text: {predicted_word}")
            last_print_time = time.time()

        last_prediction = current_prediction

        cv2.rectangle(imgOutput,
                      (x-offset, y-offset-50),
                      (x-offset+90, y-offset),
                      (255,0,255),
                      cv2.FILLED)

        cv2.putText(imgOutput,
                    labels[index],
                    (x, y-26),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1.7,
                    (255,255,255),
                    2)

        cv2.rectangle(imgOutput,
                      (x-offset, y-offset),
                      (x+w+offset, y+h+offset),
                      (255,0,255),
                      4)

    if time.time() - last_print_time >= word_delay and predicted_word:
        text_speech.say(f"The word is {predicted_word}")
        text_speech.runAndWait()

        predicted_word = ""
        text_label.config(text="Detected Text:")

    imgRGB = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    imgPIL = Image.fromarray(imgRGB)
    imgTk = ImageTk.PhotoImage(image=imgPIL)

    video_label.imgTk = imgTk
    video_label.configure(image=imgTk)

    root.after(10, update_frame)


def close_program():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()


# Start video
update_frame()

close_button = tk.Button(
    root,
    text="Close",
    font=("Arial",14),
    command=close_program,
    bg="red",
    fg="white"
)
close_button.pack(pady=20)

root.protocol("WM_DELETE_WINDOW", close_program)
root.mainloop()