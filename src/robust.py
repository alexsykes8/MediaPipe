import cv2
import numpy as np
import mediapipe as mp

# Open the default camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

distance_threshold = -10

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
            'blue': [(100, 100, 50), (140, 255, 255)],
            'green': [(35, 100, 25), (85, 255, 255)]
        }

        largest_contours_green = []
        largest_contours_blue = []

        for color, (lower, upper) in color_ranges.items():
            # Create a mask for the current color
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if len(cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)) >= 3]

            # Sort the valid contours by area in descending order
            valid_contours_sorted = sorted(valid_contours, key=cv2.contourArea, reverse=True)

            if color == 'green':
                largest_contours_green = valid_contours_sorted[:3]
            elif color == 'blue':
                largest_contours_blue = valid_contours_sorted[:1]




        # Draw landmarks and connections for detected hands
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determine if it's a left or right hand
                handedness = results.multi_handedness[idx].classification[0].label

                # Locate the index finger
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_coords = (int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0]))

                # Locate the pinky finger
                pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                pinky_coords = (int(pinky.x * frame.shape[1]), int(pinky.y * frame.shape[0]))

                # initialise the logic for if the fingers are correctly placed
                left_hand_index = left_hand_pinky = right_hand_index = right_hand_pinky = False

                # check placement of left hand
                if handedness == 'Left' and len(largest_contours_green) >= 2:
                    try:
                        # Sort contours by x-coordinate (left to right)
                        sorted_contours = sorted(largest_contours_green, key=lambda c: cv2.boundingRect(c)[0])
                        if cv2.pointPolygonTest(sorted_contours[1], tuple(index_coords), True) > distance_threshold:
                            left_hand_index = True
                        if cv2.pointPolygonTest(sorted_contours[0], tuple(pinky_coords), True) > distance_threshold:
                            left_hand_pinky = True
                    # If the hand is fully covering a contour, consider placement to be incorrect
                    except:
                        pass

                # Check for left-hand pinky being in the correct green contour
                if handedness == 'Right' and len(largest_contours_green) >= 2:
                    try:
                        # Find the leftmost contour by sorting based on the minimum x-coordinate
                        sorted_contours = sorted(largest_contours_green, key=lambda c: cv2.boundingRect(c)[0])
                        if cv2.pointPolygonTest(sorted_contours[2], tuple(pinky_coords), True) > distance_threshold:
                            right_hand_pinky = True
                        if largest_contours_blue and cv2.pointPolygonTest(largest_contours_blue[0], tuple(index_coords), True) > distance_threshold:
                            right_hand_index = True
                    # If the hand is fully covering a contour, consider placement to be incorrect
                    except:
                        pass

                # release message if hands are placed correctly
                if left_hand_pinky and left_hand_index:
                    cv2.putText(frame, "Correct placement of left hand", (75, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if right_hand_pinky and right_hand_index:
                    cv2.putText(frame, "Correct placement of right hand", (75, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for idx, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(f"Landmark {idx}: ({cx}, {cy})")


        # Draw the three largest green contours
        for contour in largest_contours_green:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
        # Draw the largest blue contour
        for contour in largest_contours_blue:
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), 3)

        # Display the current frame with shapes and hands drawn
        cv2.imshow('Shape and Hand Detection', frame)

        # Press 'q' to quit the feed window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close the window
    cap.release()
    cv2.destroyAllWindows()
