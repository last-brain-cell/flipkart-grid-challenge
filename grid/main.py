import cv2
from PIL import Image
import time
from roboflow import Roboflow

rf = Roboflow(api_key="SFUTqZCRuFcSktmRqe5V")
project = rf.workspace().project("cardboard-boxes-jaknz")
model = project.version(1).model

cap = cv2.VideoCapture(0)
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 800, 800)

start = 0
end = 0
while cap.isOpened():
    ret, frame = cap.read()
    start = time.time()

    data = Image.fromarray(frame)
    data.save("your_file.jpeg")

    prediction = model.predict(
        "your_file.jpeg",
        confidence=40,
        overlap=30,
    ).json()

    print(prediction)
    color = (0, 0, 255)
    radius = 40
    for pred in prediction["predictions"]:
        x = pred["x"]  # X COORDINATE
        y = pred["y"]  # X COORDINATE
        cv2.circle(frame, (x, y), radius, color, -1)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Kal ek
