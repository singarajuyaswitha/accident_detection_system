import math

collision_counter = {}

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0

    intersection = (x2 - x1) * (y2 - y1)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union = area1 + area2 - intersection

    return intersection / union


def detect_accident(vehicles):

    accidents = []

    for i in range(len(vehicles)):

        for j in range(i + 1, len(vehicles)):

            v1 = vehicles[i]
            v2 = vehicles[j]

            box1 = v1["box"]
            box2 = v2["box"]

            iou = calculate_iou(box1, box2)
           

            x1, y1 = v1["center"]
            x2, y2 = v2["center"]

            distance = math.sqrt(
                (x1 - x2) ** 2 +
                (y1 - y2) ** 2
            )

            pair = tuple(sorted([v1["id"], v2["id"]]))
            print(f"Pair: {pair} | IOU: {iou:.2f} | Distance: {distance:.2f} | Speed1: {v1['speed']:.2f} | Speed2: {v2['speed']:.2f}")
            # Better accident condition
            if (
              
                distance < 20 and
                (v1["speed"] > 8 or v2["speed"] > 8)
            ):

                collision_counter[pair] = collision_counter.get(pair, 0) + 1

            else:

                collision_counter[pair] = 0

            # Confirm after 3 consecutive frames
            if collision_counter[pair] >= 1:

                accidents.append({
                    "pair": pair,
                    "vehicle1": v1,
                    "vehicle2": v2
                })

    return accidents