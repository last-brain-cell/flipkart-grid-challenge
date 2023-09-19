import cv2
import numpy as np

# Load the predefined dictionary for ArUco markers
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Create a marker detector object
aruco_params = cv2.aruco.DetectorParameters_create()

# Define the camera matrix and distortion coefficients
camera_matrix = np.array(
    [[focal_length_x, 0, center_x], [0, focal_length_y, center_y], [0, 0, 1]],
    dtype=np.float32,
)

dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# Create a VideoCapture object to capture video from the camera (0 is usually the built-in camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect ArUco markers
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        frame, aruco_dict, parameters=aruco_params
    )

    if ids is not None:
        # Draw detected markers and estimate pose for each one
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, camera_matrix, dist_coeffs
        )

        for i in range(len(ids)):
            cv2.aruco.drawAxis(
                frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1
            )  # Draw coordinate axes
            cv2.aruco.drawDetectedMarkers(frame, corners)  # Draw detected markers

    # Display the frame with detected markers and coordinate axes
    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
