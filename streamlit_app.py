import streamlit as st
import cv2
import numpy as np
import tempfile
import datetime
import time

st.title("🚗 Accident Detection System")

uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4", "avi"])

if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    car_cascade = cv2.CascadeClassifier('cars.xml')
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    last_detected_time = 0
    display_duration = 3

    while True:
        ret, frame1 = cap.read()
        if not ret:
            break

        ret2, frame2 = cap.read()
        if not ret2:
            break

        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        current_time = time.time()

        for contour in contours:
            if cv2.contourArea(contour) > 5000:
                last_detected_time = current_time

        if current_time - last_detected_time <= display_duration:
            cv2.putText(frame1, "Accident Detected!", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray_frame, 1.1, 6)

        for (x,y,w,h) in cars:
            cv2.rectangle(frame1, (x,y), (x+w,y+h), (0,255,0), 2)

        stframe.image(frame1, channels="BGR")

    cap.release()