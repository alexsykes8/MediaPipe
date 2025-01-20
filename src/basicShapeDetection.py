import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import deque

# Function to find the center of a contour
def find_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return (cX, cY)

# Initialize deques for tracking the last 20 frame centers
max_frames = 20
green_left_deque = deque(maxlen=max_frames)
green_right_deque = deque(maxlen=max_frames)
blue_left_deque = deque(maxlen=max_frames)
blue_right_deque = deque(maxlen=max_frames)

# Open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access camera.")
else:
    print("Camera is working.")

    # Continuously capture frames
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert the frame to HSV for color-based processing
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color ranges for each object
        color_ranges = {
            'blue': [(100, 50, 50), (140, 255, 255)],  # Lower and upper range for blue
            'green': [(40, 20, 20), (90, 170, 170)],  # Lower and upper range for green
        }

        current_green_centers = None
        current_blue_centers = None

        for color, (lower, upper) in color_ranges.items():
            # Create a mask for the current color
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            centres = []
            valid_contours = []

            for contour in contours:
                # Approximate the contour to identify the shape
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

                # Check if the contour is a valid shape (e.g., polygon)
                if len(approx) >= 3:
                    # Find the center of the shape
                    center = find_center(contour)
                    centres.append(center)
                    valid_contours.append(contour)

            # Use KMeans to cluster the centers of the shapes
            if len(centres) >= 2:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(centres)
                cluster_centers = kmeans.cluster_centers_

                # Update the running averages based on cluster position (left or right)
                if color == 'green' and len(cluster_centers) > 0:
                    current_green_centers = cluster_centers
                    # Leftmost and rightmost clusters based on x-coordinate
                    left_cluster = min(current_green_centers, key=lambda x: x[0])
                    right_cluster = max(current_green_centers, key=lambda x: x[0])

                    green_left_deque.append(left_cluster)
                    green_right_deque.append(right_cluster)

                elif color == 'blue' and len(cluster_centers) > 0:
                    current_blue_centers = cluster_centers
                    # Leftmost and rightmost clusters based on x-coordinate
                    left_cluster = min(current_blue_centers, key=lambda x: x[0])
                    right_cluster = max(current_blue_centers, key=lambda x: x[0])

                    blue_left_deque.append(left_cluster)
                    blue_right_deque.append(right_cluster)

                cluster_colors = [(0, 255, 0), (255, 0, 0)] if color == 'green' else [(255, 255, 0), (0, 255, 255)]
                for i, contour in enumerate(valid_contours):
                    cv2.drawContours(frame, [contour], -1, cluster_colors[kmeans.labels_[i]], 2)
                    cv2.circle(frame, centres[i], 5, cluster_colors[kmeans.labels_[i]], -1)

        # Calculate the running average of the centers over the last 20 frames
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

        # Draw polygons connecting all four cluster centers (leftmost and rightmost for green and blue)
        if current_green_centers is not None and current_blue_centers is not None:
            # Collect the four centers (leftmost and rightmost from green and blue)
            all_centers = np.vstack((
                avg_left_green,
                avg_right_green,
                avg_right_blue,
                avg_left_blue
            )).astype(int)

            if len(all_centers) == 4:  # Ensure there are exactly four centers
                cv2.polylines(frame, [all_centers], isClosed=True, color=(0, 255, 255), thickness=2)

        # Display the current frame with shapes drawn
        cv2.imshow('Shape Detection', frame)

        # Press 'q' to quit the feed window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close the window
    cap.release()
    cv2.destroyAllWindows()
