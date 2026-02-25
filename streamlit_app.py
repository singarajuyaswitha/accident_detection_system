import streamlit as st
import cv2
import numpy as np
import datetime
import time

st.title("🚗 Accident Detection System (Video Based)")

video_file = "test_video.mp4"

car_cascade = cv2.CascadeClassifier('cars.xml')

cap = cv2.VideoCapture(video_file)

stframe = st.empty()

ret, frame1 = cap.read()
ret, frame2 = cap.read()

last_detected_time = 0
display_duration = 3  # seconds

while ret:

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    accident_detected = False
    current_time = time.time()

    for contour in contours:
        if cv2.contourArea(contour) > 5000:
            accident_detected = True
            last_detected_time = current_time

    # Show accident alert for few seconds
    if current_time - last_detected_time <= display_duration:
        cv2.putText(frame1, "Accident Detected!", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        cv2.putText(frame1, str(datetime.datetime.now()), (20,frame1.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Vehicle detection
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray_frame, 1.1, 6)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame1, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame1, "Vehicle", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    stframe.image(frame1, channels="BGR")

    frame1 = frame2
    ret, frame2 = cap.read()

cap.release()