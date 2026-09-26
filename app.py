import os
import cv2
import tempfile
import threading
import winsound
import streamlit as st
from datetime import datetime

from detector import detect
from tracker import update_tracker
from accident import detect_accident


# ---------------- Alarm ---------------- #

def play_alarm():

    sound_path = os.path.join(
        os.path.dirname(__file__),
        "alaram.wav"
    )

    if os.path.exists(sound_path):

        winsound.PlaySound(
            sound_path,
            winsound.SND_FILENAME |
            winsound.SND_ASYNC
        )


# ---------------- Streamlit UI ---------------- #

st.set_page_config(
    page_title="AI Accident Detection",
    layout="wide"
)

st.title("🚗 Accident Detection System")

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    temp = tempfile.NamedTemporaryFile(
        delete=False
    )

    temp.write(uploaded_file.read())
    temp.close()

    cap = cv2.VideoCapture(temp.name)

    stframe = st.empty()
    alarm_played = False

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # ---------------- Detection ---------------- #

        results = detect(frame)

        vehicles = update_tracker(results)

        accidents = detect_accident(vehicles)
        print("=" * 50)
        print("Accidents Found :", len(accidents))
        print(accidents)
        print("=" * 50)

        annotated_frame = frame.copy()

        accident_ids = set()

        for accident in accidents:

            accident_ids.update(
                accident["pair"]
            )

        vehicle_count = len(vehicles)     
        if results[0].boxes is not None:

            for box in results[0].boxes:

                if box.id is None:
                    continue

                cls = int(box.cls.item())

                # Car, Motorcycle, Bus, Truck
                if cls not in [2, 3, 5, 7]:
                    continue

                track_id = int(box.id.item())

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Default Green
                color = (0, 255, 0)
                label = f"ID: {track_id}"

                # Accident Vehicle → Red
                if track_id in accident_ids:
                    color = (0, 0, 255)
                    label = f"ACCIDENT | ID: {track_id}"

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # ---------------- Vehicle Counter ---------------- #

        cv2.putText(
            annotated_frame,
            f"Vehicles : {vehicle_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2
        )
# ---------------- Accident Alert ---------------- #            
        if len(accidents) > 0:

            if not alarm_played:

                threading.Thread(
                    target=play_alarm,
                    daemon=True
                ).start()

                alarm_played = True

                # Create folders
                screenshot_folder = os.path.join(
                    os.path.dirname(__file__),
                    "accidents"
                )

                log_folder = os.path.join(
                    os.path.dirname(__file__),
                    "logs"
                )

                os.makedirs(
                    screenshot_folder,
                    exist_ok=True
                )

                os.makedirs(
                    log_folder,
                    exist_ok=True
                )

                # Screenshot
                image_name = datetime.now().strftime(
                    "accident_%Y%m%d_%H%M%S.jpg"
                )

                image_path = os.path.join(
                    screenshot_folder,
                    image_name
                )

                cv2.imwrite(
                    image_path,
                    annotated_frame
                )

                # CSV Log
                log_file = os.path.join(
                    log_folder,
                    "accident_log.csv"
                )

                with open(
                    log_file,
                    "a"
                ) as file:

                    file.write(
                        f"{datetime.now()},{image_name}\n"
                    )

            cv2.putText(
                annotated_frame,
                "ACCIDENT DETECTED",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        else:

            alarm_played = False

        # ---------------- Display ---------------- #
        stframe.image(
            annotated_frame,
            channels="BGR",
            
        )

    cap.release() 