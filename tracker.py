import math

vehicle_history = {}

def update_tracker(results):

    vehicles = []

    if results[0].boxes is None:
        return vehicles

    for box in results[0].boxes:

        if box.id is None:
            continue

        cls = int(box.cls.item())

        # Car, Motorcycle, Bus, Truck
        if cls not in [2, 3, 5, 7]:
            continue

        track_id = int(box.id.item())

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        speed = 0

        if track_id in vehicle_history:
            px, py = vehicle_history[track_id]
            speed = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

        vehicle_history[track_id] = (cx, cy)

        # -------- DEBUG --------
        print(f"Vehicle ID: {track_id} | Speed: {speed:.2f}")
        # -----------------------

        vehicles.append({
            "id": track_id,
            "class": cls,
            "box": (x1, y1, x2, y2),
            "center": (cx, cy),
            "speed": speed
        })

    return vehicles