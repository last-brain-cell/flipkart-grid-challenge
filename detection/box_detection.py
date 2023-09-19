import time
import json
from ultralytics import YOLO
import cv2
import paho.mqtt.client as mqtt


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
    else:
        print("Connection failed")


# start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)


# mqttBroker = "mqtt.eclipseprojects.io"
mqttBroker = "localhost"
client = mqtt.Client("Boxes", clean_session=False)
client.on_connect = on_connect

# model
model = YOLO("../models/best.pt")

prev_frame_time = 0
new_frame_time = 0

client = mqtt.Client("publish", clean_session=False)
client.connect(mqttBroker, 1884)

while True:
    success, img = cap.read()
    if img is None:
        continue

    results = model(img, stream=True)
    coordinates = dict()

    # coordinates
    for r in results:
        boxes = r.boxes

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (0, 255, 255)
            thickness = 2

            x = (x1 + x2) // 2
            y = (y1 + y2) // 2
            coordinates[f"Box {i + 1}"] = {
                "x": x,
                "y": y,
                "Priority": i + 1,
            }
            radius = 10
            cv2.circle(img, (x, y), radius, color, -1)

            cv2.putText(img, f"{i + 1}", org, font, fontScale, color, thickness)

    client.publish("coordinates", json.dumps(coordinates))
    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    fps = str(round(1 / (new_frame_time - prev_frame_time), 2)) + "fps"
    prev_frame_time = new_frame_time

    cv2.putText(img, fps, (7, 70), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
