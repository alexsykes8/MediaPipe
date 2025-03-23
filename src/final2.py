import cv2
import numpy as np
import mediapipe as mp

# Function to update HSV values from trackbars for green
def get_hsv_values_green():
    lower_h = cv2.getTrackbarPos("Lower H", "HSV Adjustments Green")
    lower_s = cv2.getTrackbarPos("Lower S", "HSV Adjustments Green")
    lower_v = cv2.getTrackbarPos("Lower V", "HSV Adjustments Green")
    upper_h = cv2.getTrackbarPos("Upper H", "HSV Adjustments Green")
    upper_s = cv2.getTrackbarPos("Upper S", "HSV Adjustments Green")
    upper_v = cv2.getTrackbarPos("Upper V", "HSV Adjustments Green")

    return (lower_h, lower_s, lower_v), (upper_h, upper_s, upper_v)



# Function to update HSV values from trackbars for blue
def get_hsv_values_blue():
    lower_h = cv2.getTrackbarPos("Lower H", "HSV Adjustments Blue")
    lower_s = cv2.getTrackbarPos("Lower S", "HSV Adjustments Blue")
    lower_v = cv2.getTrackbarPos("Lower V", "HSV Adjustments Blue")
    upper_h = cv2.getTrackbarPos("Upper H", "HSV Adjustments Blue")
    upper_s = cv2.getTrackbarPos("Upper S", "HSV Adjustments Blue")
    upper_v = cv2.getTrackbarPos("Upper V", "HSV Adjustments Blue")

    return (lower_h, lower_s, lower_v), (upper_h, upper_s, upper_v)

# Function to find the center of a contour
def find_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return (cX, cY)

# Function for trackbars (does nothing)
def nothing(x):
    pass

cv2.namedWindow("HSV Adjustments Green")
cv2.resizeWindow("HSV Adjustments Green", 400, 300)

cv2.namedWindow("HSV Adjustments Blue")
cv2.resizeWindow("HSV Adjustments Blue", 400, 300)

# create trackbar for adjusting green hsv values
cv2.createTrackbar("Lower H", "HSV Adjustments Green", 35, 179, nothing)
cv2.createTrackbar("Lower S", "HSV Adjustments Green", 100, 255, nothing)
cv2.createTrackbar("Lower V", "HSV Adjustments Green", 25, 255, nothing)
cv2.createTrackbar("Upper H", "HSV Adjustments Green", 85, 179, nothing)
cv2.createTrackbar("Upper S", "HSV Adjustments Green", 255, 255, nothing)
cv2.createTrackbar("Upper V", "HSV Adjustments Green", 255, 255, nothing)

# create trackbar for adjusting blue hsv values
cv2.createTrackbar("Lower H", "HSV Adjustments Blue", 100, 179, nothing)
cv2.createTrackbar("Lower S", "HSV Adjustments Blue", 100, 255, nothing)
cv2.createTrackbar("Lower V", "HSV Adjustments Blue", 50, 255, nothing)
cv2.createTrackbar("Upper H", "HSV Adjustments Blue", 140, 179, nothing)
cv2.createTrackbar("Upper S", "HSV Adjustments Blue", 255, 255, nothing)
cv2.createTrackbar("Upper V", "HSV Adjustments Blue", 255, 255, nothing)

# Open the default camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

distance_threshold = -15


# Apply CLAHE to improve contrast
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Adjust brightness using gamma correction
def adjust_gamma(image, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

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

        # Process the frame to detect hands
        results = hands.process(frame)

        frame = adjust_gamma(frame, gamma=1.2)
        frame = apply_clahe(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color ranges for each object
        color_ranges = {
            'blue': [get_hsv_values_blue()[0], get_hsv_values_blue()[1]],  # Lower and upper range for blue
            'green': [get_hsv_values_green()[0], get_hsv_values_green()[1]],  # Lower and upper range for green
        }

        largest_contours_green = []
        largest_contours_blue = []

        for color, (lower, upper) in color_ranges.items():
            # Create a mask for the current color
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_contours = []
            for contour in contours:
                # Approximate the contour to identify the shape
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

                # Check if the contour is a valid shape (e.g., polygon)
                if len(approx) >= 3:
                    valid_contours.append(contour)

            # Sort the valid contours by area in descending order
            valid_contours_sorted = sorted(valid_contours, key=cv2.contourArea, reverse=True)

            # Select the three largest green contours and the one largest blue contour
            if color == 'green':
                largest_contours_green = valid_contours_sorted[:3]
            elif color == 'blue':
                largest_contours_blue = valid_contours_sorted[:1]


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
                pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                pinky_coords = (int(pinky.x * frame.shape[1]), int(pinky.y * frame.shape[0]))

                # initialise the logic for if the fingers are correctly placed
                left_hand_index = False
                right_hand_index = False
                left_hand_pinky = False
                right_hand_pinky = False

                # check placement of left hand
                if handedness == 'Left':
                    # Check for left hand index being in the correct green contour
                    if largest_contours_green is not None and len(largest_contours_green) >= 2:
                        try:
                            # Sort contours by x-coordinate (left to right)
                            sorted_contours = sorted(largest_contours_green, key=lambda c: cv2.boundingRect(c)[0])

                            # Select the second leftmost contour
                            second_leftmost_contour = sorted_contours[1]  # Index 1 gives the second leftmost
                            second_leftmost_center = find_center(second_leftmost_contour)

                            distance = cv2.pointPolygonTest(second_leftmost_contour, tuple(index_coords), True)
                            cv2.putText(frame, f"Distance to 2: {distance:.2f}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                            if distance > distance_threshold:
                                left_hand_index = True
                        # If the hand is fully covering a contour, consider placement to be incorrect
                        except:
                            left_hand_index = False

                    # Check for left-hand pinky being in the correct green contour
                    if largest_contours_green is not None and len(largest_contours_green) >= 2:
                        try:
                            # Find the leftmost contour by sorting based on the minimum x-coordinate
                            leftmost_contour = min(largest_contours_green, key=lambda c: cv2.boundingRect(c)[0])
                            leftmost_center = find_center(leftmost_contour)

                            distance = cv2.pointPolygonTest(leftmost_contour, tuple(pinky_coords), True)
                            cv2.putText(frame, f"Distance to 1: {distance:.2f}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                            if distance > distance_threshold:
                                left_hand_pinky = True
                        # If the hand is fully covering a contour, consider placement to be incorrect
                        except:
                            left_hand_pinky = False

                # check placement of right hand
                if handedness == 'Right':
                    # Check for right hand pinky being in the correct green contour
                    if largest_contours_green is not None and len(largest_contours_green) >= 2:
                        try:
                            # Sort contours by x-coordinate (left to right)
                            sorted_contours = sorted(largest_contours_green, key=lambda c: cv2.boundingRect(c)[0])

                            # Select the rightmost contour
                            right_most_contour = sorted_contours[2]  # Index 2 gives the rightmost
                            right_most_centre = find_center(right_most_contour)

                            distance = cv2.pointPolygonTest(right_most_contour, tuple(pinky_coords), True)
                            cv2.putText(frame, f"Distance to 4: {distance:.2f}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                            if distance > distance_threshold:
                                right_hand_pinky = True
                        # If the hand is fully covering a contour, consider placement to be incorrect
                        except:
                            right_hand_pinky = False

                    # Check for right-hand index being in the correct blue contour
                    if largest_contours_blue is not None:
                        try:
                            distance = cv2.pointPolygonTest(largest_contours_blue[0], tuple(index_coords), True)
                            cv2.putText(frame, f"Distance to 3: {distance:.2f}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                            if distance > distance_threshold:
                                right_hand_index = True
                        # If the hand is fully covering a contour, consider placement to be incorrect
                        except:
                            right_hand_index = False

                # release message if hands are placed correctly
                if (left_hand_pinky and left_hand_index):
                    cv2.putText(frame, f"Correct placement of right hand", (75, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if (right_hand_pinky and right_hand_index):
                    cv2.putText(frame, f"Correct placement of left hand", (75, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for idx, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(f"Landmark {idx}: ({cx}, {cy})")

        # Draw the three largest green contours
        for contour in largest_contours_green:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)  # Green

        # Draw the largest blue contour
        for contour in largest_contours_blue:
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), 3)  # Blue

        # Display the current frame with shapes and hands drawn
        cv2.imshow('Shape and Hand Detection', frame)

        # Press 'q' to quit the feed window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close the window
    cap.release()
    cv2.destroyAllWindows()
