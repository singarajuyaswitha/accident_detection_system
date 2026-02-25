import cv2
import winsound
import datetime
import time

# Load vehicle detection model
car_cascade = cv2.CascadeClassifier('cars.xml')

# Camera capture
cap = cv2.VideoCapture(0)

# Reduce resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Variables for beep frequency
last_beep_time = 0
frame_count = 0  # for frame skipping

while ret:
    frame_count += 1
    # Process every 3rd frame for speed
    if frame_count % 3 != 0:
        frame1 = frame2
        ret, frame2 = cap.read()
        continue

    # Resize for motion detection
    frame1_small = cv2.resize(frame1, (320, 240))
    frame2_small = cv2.resize(frame2, (320, 240))
    diff = cv2.absdiff(frame1_small, frame2_small)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Motion detection
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    accident_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 3000:  # Lowered for testing
            accident_detected = True
            current_time = time.time()
            if current_time - last_beep_time > 3:  # beep interval 3 sec
                winsound.Beep(1000, 300)
                last_beep_time = current_time

            # Log accident
            with open("accident_log.txt", "a") as f:
                f.write(f"Accident detected at {datetime.datetime.now()}\n")

    if accident_detected:
        cv2.putText(frame1, "Accident Detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame1, str(datetime.datetime.now()), (10, frame1.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Vehicle detection
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(60,60)
    )

    for (x,y,w,h) in cars:
        if w*h > 4000:  # only rectangles bigger than 4000 px
            cv2.rectangle(frame1, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.putText(frame1, "Vehicle Detected", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cv2.imshow("Accident & Vehicle Detection", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    # Quit on q or ESC
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()