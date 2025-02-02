import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import deque

def find_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return (cX, cY)

max_frames = 20
green_left_deque = deque(maxlen=max_frames)
green_right_deque = deque(maxlen=max_frames)
blue_left_deque = deque(maxlen=max_frames)
blue_right_deque = deque(maxlen=max_frames)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access camera.")
else:
    print("Camera is working.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        color_ranges = {
            'blue': [(100, 50, 50), (140, 255, 255)],  # Lower and upper range for blue
            'green': [(40, 20, 20), (90, 170, 170)],  # Lower and upper range for green
        }

        current_green_centers = None
        current_blue_centers = None

        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            centres = []
            valid_contours = []

            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

                if len(approx) >= 3:
                    center = find_center(contour)
                    centres.append(center)
                    valid_contours.append(contour)

            if len(centres) >= 2:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(centres)
                cluster_centers = kmeans.cluster_centers_

                if color == 'green' and len(cluster_centers) > 0:
                    current_green_centers = cluster_centers
                    left_cluster = min(current_green_centers, key=lambda x: x[0])
                    right_cluster = max(current_green_centers, key=lambda x: x[0])

                    green_left_deque.append(left_cluster)
                    green_right_deque.append(right_cluster)

                elif color == 'blue' and len(cluster_centers) > 0:
                    current_blue_centers = cluster_centers
                    left_cluster = min(current_blue_centers, key=lambda x: x[0])
                    right_cluster = max(current_blue_centers, key=lambda x: x[0])

                    blue_left_deque.append(left_cluster)
                    blue_right_deque.append(right_cluster)

                cluster_colors = [(0, 255, 0), (255, 0, 0)] if color == 'green' else [(255, 255, 0), (0, 255, 255)]
                for i, contour in enumerate(valid_contours):
                    cv2.drawContours(frame, [contour], -1, cluster_colors[kmeans.labels_[i]], 2)
                    cv2.circle(frame, centres[i], 5, cluster_colors[kmeans.labels_[i]], -1)

        if len(green_left_deque) == max_frames and len(green_right_deque) == max_frames:
            avg_left_green = np.mean(green_left_deque, axis=0)
            avg_right_green = np.mean(green_right_deque, axis=0)
        else:
            avg_left_green = avg_right_green = np.array([0.0, 0.0])

        if len(blue_left_deque) == max_frames and len(blue_right_deque) == max_frames:
            avg_left_blue = np.mean(blue_left_deque, axis=0)
            avg_right_blue = np.mean(blue_right_deque, axis=0)
        else:
            avg_left_blue = avg_right_blue = np.array([0.0, 0.0])

        if current_green_centers is not None and current_blue_centers is not None:
            all_centers = np.vstack((
                avg_left_green,
                avg_right_green,
                avg_right_blue,
                avg_left_blue
            )).astype(int)

            if len(all_centers) == 4:
                cv2.polylines(frame, [all_centers], isClosed=True, color=(0, 255, 255), thickness=2)

        cv2.imshow('Shape Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
