import cv2
import numpy as np
import mediapipe as mp
from collections import deque


# Function to update HSV values from trackbars
def get_hsv_values_green():
    lower_h = cv2.getTrackbarPos("Lower H", "HSV Adjustments Green")
    lower_s = cv2.getTrackbarPos("Lower S", "HSV Adjustments Green")
    lower_v = cv2.getTrackbarPos("Lower V", "HSV Adjustments Green")
    upper_h = cv2.getTrackbarPos("Upper H", "HSV Adjustments Green")
    upper_s = cv2.getTrackbarPos("Upper S", "HSV Adjustments Green")
    upper_v = cv2.getTrackbarPos("Upper V", "HSV Adjustments Green")

    return (lower_h, lower_s, lower_v), (upper_h, upper_s, upper_v)


# Function to update trackbars
def nothing(x):
    pass


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
green_left_deque = deque(maxlen=max_frames)
green_right_deque = deque(maxlen=max_frames)
blue_left_deque = deque(maxlen=max_frames)
blue_right_deque = deque(maxlen=max_frames)


cv2.namedWindow("HSV Adjustments Green")
cv2.resizeWindow("HSV Adjustments Green", 400, 300)

# create a trackbar so that the colour ranges can be edited dynamically
cv2.createTrackbar("Lower H", "HSV Adjustments Green", 20, 179, nothing)
cv2.createTrackbar("Lower S", "HSV Adjustments Green", 130, 255, nothing)
cv2.createTrackbar("Lower V", "HSV Adjustments Green", 20, 255, nothing)
cv2.createTrackbar("Upper H", "HSV Adjustments Green", 100, 179, nothing)
cv2.createTrackbar("Upper S", "HSV Adjustments Green", 219, 255, nothing)
cv2.createTrackbar("Upper V", "HSV Adjustments Green", 140, 255, nothing)


# Distance of the fingertip to a colour block for it to be considered correct placement
distance_threshold = -20

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
            'blue': [(100, 50, 50), (140, 255, 255)],  # Lower and upper range for blue
            'green': [get_hsv_values_green()[0], get_hsv_values_green()[1]],  # Lower and upper range for green
        }

        current_green_centers = None
        current_blue_centers = None

        valid_contours_blue = []
        valid_contours_green = []


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


            # Update the running averages based on cluster position (left or right)
            if color == 'green' and len(centres) > 0:
                current_green_centers = centres
                # Leftmost and rightmost clusters based on x-coordinate
                left_cluster = min(current_green_centers, key=lambda x: x[0])
                right_cluster = max(current_green_centers, key=lambda x: x[0])

                green_left_deque.append(left_cluster)
                green_right_deque.append(right_cluster)

            elif color == 'blue' and len(centres) > 0:
                current_blue_centers = centres
                # Leftmost and rightmost clusters based on x-coordinate
                left_cluster = min(current_blue_centers, key=lambda x: x[0])
                right_cluster = max(current_blue_centers, key=lambda x: x[0])



                blue_left_deque.append(left_cluster)
                blue_right_deque.append(right_cluster)

            cluster_colors = [(0, 255, 0), (255, 0, 0)] if color == 'green' else [(255, 255, 0), (0, 255, 255)]
            # Sort the valid contours by area in descending order
            valid_contours_sorted = sorted(valid_contours, key=cv2.contourArea, reverse=True)

            # Select the two largest contours
            if color == 'green':
                largest_contours = valid_contours_sorted[:3]
            if color == 'blue':
                largest_contours = valid_contours_sorted[:1]

            # Draw the two largest contours and their centers
            for contour in largest_contours:
                # Find the center of the contour
                center = find_center(contour)

                # Draw the contour and center
                if color == 'green':
                    valid_contours_green.append(contour)

                if color == 'blue':
                    valid_contours_blue.append(contour)




        # Calculate the running average of the centers over the last 20 frames, ignoring anomalies
        if len(green_left_deque) == max_frames and len(green_right_deque) == max_frames:
            filtered_left_green = filter_anomalies(green_left_deque)
            filtered_right_green = filter_anomalies(green_right_deque)
            avg_left_green = np.mean(filtered_left_green, axis=0) if len(filtered_left_green) > 0 else np.array([0.0, 0.0])
            avg_right_green = np.mean(filtered_right_green, axis=0) if len(filtered_right_green) > 0 else np.array([0.0, 0.0])
        else:
            avg_left_green = avg_right_green = np.array([0.0, 0.0])

        if len(blue_left_deque) == max_frames and len(blue_right_deque) == max_frames:
            filtered_left_blue = filter_anomalies(blue_left_deque)
            filtered_right_blue = filter_anomalies(blue_right_deque)
            avg_left_blue = np.mean(filtered_left_blue, axis=0) if len(filtered_left_blue) > 0 else np.array([0.0, 0.0])
            avg_right_blue = np.mean(filtered_right_blue, axis=0) if len(filtered_right_blue) > 0 else np.array([0.0, 0.0])
        else:
            avg_left_blue = avg_right_blue = np.array([0.0, 0.0])


        # Process the frame to detect hands
        results = hands.process(frame)

        # Draw landmarks and connections for detected hands
        if results.multi_hand_landmarks:
            for idx in range(len(results.multi_hand_landmarks)):
                hand_landmarks = results.multi_hand_landmarks[idx]

                # Determine if it's a left or right hand
                handedness = results.multi_handedness[idx].classification[0].label

                # Locate the index finger
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_coords = (int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0]))

                # Locate the pinky finger
                left_pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                left_pinky_coords = (int(left_pinky.x * frame.shape[1]), int(left_pinky.y * frame.shape[0]))

                left_hand_index = False
                right_hand_index = False
                left_hand_pinky = False
                right_hand_pinky = False

                # check placement of left hand
                if handedness == 'Left':
                    # check for left hand index being in the correct green contour
                    if current_green_centers is not None and len(current_green_centers) >= 2:
                        try:
                            second_largest_contour = sorted(valid_contours, key=cv2.contourArea, reverse=True)[2]
                            second_largest_center = find_center(second_largest_contour)
                            distance = cv2.pointPolygonTest(second_largest_contour, tuple(index_coords), True)
                            #cv2.putText(frame, f"Distance to Green 1: {distance:.2f}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            if distance > distance_threshold:
                                left_hand_index = True
                        # if the hand is fully covering a contour, consider placement to be incorrect
                        except:
                            left_hand_index = False

                    # check for left hand pinky being in the correct green contour
                    if current_green_centers is not None and len(current_green_centers) >= 2:
                        try:
                            second_largest_contour = sorted(valid_contours, key=cv2.contourArea, reverse=True)[0]
                            second_largest_center = find_center(second_largest_contour)

                            distance = cv2.pointPolygonTest(second_largest_contour, tuple(left_pinky_coords), True)
                            if distance > distance_threshold:
                                left_hand_pinky = True
                        # if the hand is fully covering a contour, consider placement to be incorrect
                        except:
                            left_hand_pinky = False

                if handedness == 'Right':
                    # check for right hand index being in the correct blue contour
                    if current_blue_centers is not None and len(current_blue_centers) >= 2:
                        try:
                            second_largest_contour = sorted(valid_contours_blue, key=cv2.contourArea, reverse=True)[0]
                            second_largest_center = find_center(second_largest_contour)
                            distance = cv2.pointPolygonTest(second_largest_contour, tuple(index_coords), True)
                            if distance > distance_threshold:
                                right_hand_index = True
                        # if the hand is fully covering a contour, consider placement to be incorrect
                        except:
                            right_hand_index = False
                    # check for right hand pinky being in the correct green contour
                    if current_green_centers is not None and len(current_green_centers) >= 2:
                        try:
                            second_largest_contour = sorted(valid_contours, key=cv2.contourArea, reverse=True)[2]
                            second_largest_center = find_center(second_largest_contour)

                            distance = cv2.pointPolygonTest(second_largest_contour, tuple(left_pinky_coords), True)
                            if distance > distance_threshold:
                                right_hand_pinky = True
                        # if the hand is fully covering a contour, consider placement to be incorrect
                        except:
                            right_hand_pinky = False


                # release message if hands are placed correctly
                if (left_hand_pinky and left_hand_index):
                    cv2.putText(frame, f"Correct placement of right hand", (75, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if (right_hand_pinky and right_hand_index):
                    cv2.putText(frame, f"Correct placement of left hand", (75, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw the three largest green contours and the one largest blue contour
        for contour in valid_contours_green:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)

        # Draw the blue contours
        for contour in valid_contours_blue:
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), 3)

        # Display the current frame with shapes and hands drawn
        cv2.imshow('Shape and Hand Detection', frame)

        # Press 'q' to quit the feed window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close the window
    cap.release()
    cv2.destroyAllWindows()