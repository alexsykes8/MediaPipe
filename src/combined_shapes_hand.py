import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import deque
import mediapipe as mp

# Function to filter out anomalies
def filter_anomalies(deque):
    array = np.array(deque)
    median = np.median(array, axis=0)
    # Compute Euclidean distances from the median
    distances = np.linalg.norm(array - median, axis=1)
    # Filter points with distances within a reasonable threshold (e.g., less than 2 times the median distance)
    threshold = 2 * np.median(distances)
    filtered_array = array[distances < threshold]
    return filtered_array

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
purple_top_left_deque = deque(maxlen=max_frames)
purple_top_right_deque = deque(maxlen=max_frames)
purple_bottom_left_deque = deque(maxlen=max_frames)
purple_bottom_right_deque = deque(maxlen=max_frames)

# Open the default camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

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
            'purple': [(120, 16, 26), (140, 255, 255)],  # Lower and upper range for blue
        }

        current_purple_centers = None

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
            if len(centres) >= 4:
                kmeans = KMeans(n_clusters=4, random_state=0).fit(centres)
                cluster_centers = kmeans.cluster_centers_


                # Update the running averages based on cluster position
                if color == 'purple' and len(cluster_centers) > 0:
                    current_purple_centers = cluster_centers

                    # Find top-left, top-right, bottom-left, and bottom-right clusters
                    top_left_cluster = min(current_purple_centers, key=lambda x: (x[1], x[0]))  # Min y, then min x
                    top_right_cluster = min(current_purple_centers, key=lambda x: (x[1], -x[0]))  # Min y, then max x
                    bottom_left_cluster = max(current_purple_centers, key=lambda x: (x[1], -x[0]))  # Max y, then min x
                    bottom_right_cluster = max(current_purple_centers, key=lambda x: (x[1], x[0]))  # Max y, then max x

                    # Update deques for each corner
                    purple_top_left_deque.append(top_left_cluster)
                    purple_top_right_deque.append(top_right_cluster)
                    purple_bottom_left_deque.append(bottom_left_cluster)
                    purple_bottom_right_deque.append(bottom_right_cluster)

                cluster_colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
                for i, contour in enumerate(valid_contours):
                    cv2.drawContours(frame, [contour], -1, cluster_colors[kmeans.labels_[i]], 2)
                    cv2.circle(frame, centres[i], 5, cluster_colors[kmeans.labels_[i]], -1)


        # Calculate the running average of the centers over the last 20 frames, ignoring anomalies
        if len(purple_top_left_deque) == max_frames and len(purple_bottom_left_deque) == max_frames and len(purple_top_right_deque) == max_frames and len(purple_bottom_right_deque) == max_frames:
            filtered_top_left_purple = filter_anomalies(purple_top_left_deque)
            filtered_top_right_purple = filter_anomalies(purple_top_right_deque)
            filtered_bottom_left_purple = filter_anomalies(purple_bottom_left_deque)
            filtered_bottom_right_purple = filter_anomalies(purple_bottom_right_deque)
            avg_top_left_purple = np.mean(filtered_top_left_purple, axis=0) if len(filtered_top_left_purple) > 0 else np.array([0.0, 0.0])
            avg_top_right_purple = np.mean(filtered_top_right_purple, axis=0) if len(filtered_top_right_purple) > 0 else np.array([0.0, 0.0])
            avg_bottom_left_purple = np.mean(filtered_bottom_left_purple, axis=0) if len(filtered_bottom_left_purple) > 0 else np.array([0.0, 0.0])
            avg_bottom_right_purple = np.mean(filtered_bottom_right_purple, axis=0) if len(filtered_bottom_right_purple) > 0 else np.array([0.0, 0.0])

        else:
            avg_bottom_right_purple = avg_top_right_purple = avg_bottom_left_purple = avg_top_left_purple = np.array([0.0, 0.0])


        # Draw polygons connecting all four cluster centers (leftmost and rightmost for green and blue)
        if current_purple_centers is not None:
            # Collect the four centers (leftmost and rightmost from green and blue)
            all_centers = np.vstack((
                avg_top_left_purple,
                avg_top_right_purple,
                avg_bottom_right_purple,
                avg_bottom_left_purple
            )).astype(int)

            if len(all_centers) == 4:  # Ensure there are exactly four centers
                cv2.polylines(frame, [all_centers], isClosed=True, color=(0, 255, 255), thickness=2)

        # Process the frame to detect hands
        results = hands.process(frame)

        # Draw landmarks and connections for detected hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the current frame with shapes and hands drawn
        cv2.imshow('Shape and Hand Detection', frame)

        # Press 'q' to quit the feed window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close the window
    cap.release()
    cv2.destroyAllWindows()
