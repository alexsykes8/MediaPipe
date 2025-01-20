import mediapipe as mp
import cv2


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

        # Display the current frame
        cv2.imshow('Camera Feed', frame)

        # Press 'q' to quit the feed window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close the window
    cap.release()
    cv2.destroyAllWindows()

'''

cap = cv2.VideoCapture(0)  # Try with 0, 1, 2 for different camera indices

if not cap.isOpened():
    print("Error: Unable to access camera.")
else:
    print("Camera is working.")
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
    else:
        cv2.imshow('Test Image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cap.release()

print(f"MediaPipe version: {mp.__version__}")

# Open the camera (0 is typically the default webcam)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Unable to access camera.")
else:
    print("Camera is working.")

    # Loop to continuously capture frames from the camera
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the frame in a window
        cv2.imshow("Camera Feed", frame)

        # Wait for the 'q' key to be pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window when done
    cap.release()
    cv2.destroyAllWindows()

'''