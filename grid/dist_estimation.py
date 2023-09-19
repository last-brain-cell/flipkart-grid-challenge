import cv2
import numpy as np

# Define the known size of the ArUco marker (in meters)
marker_size = 0.1  # Adjust this value based on your ArUco marker size

# Load the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Load camera calibration parameters obtained from camera calibration
camera_matrix = np.load("camera_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

# Create an ArUco board for pose estimation
board = cv2.aruco.GridBoard_create(4, 4, marker_size, 1, aruco_dict)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect ArUco markers in the current frame
    corners, ids, rejected = cv2.aruco.detectMarkers(
        frame, aruco_dict, parameters=aruco_params
    )

    if ids is not None:
        # Estimate marker poses
        rvecs, tvecs, _ = cv2.aruco.estimatePoseBoard(
            corners, ids, board, camera_matrix, dist_coeffs
        )

        for i in range(len(ids)):
            # Calculate distance to each detected marker
            distance = tvecs[i][2]  # The Z-coordinate of the translation vector
            marker_id = ids[i][0]

            cv2.aruco.drawAxis(
                frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_size * 0.5
            )
            cv2.putText(
                frame,
                f"Marker {marker_id}: {distance:.2f} meters",
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

    cv2.imshow("ArUco Marker Distance Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
