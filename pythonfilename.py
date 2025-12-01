import cv2
import numpy as np

# --- 1. Load a video or webcam ---
# Solution 1: Try to open a video file named 'mavideo.mp4'
cap = cv2.VideoCapture('mavideo.mp4')

# Solution 2: If the video file is not found, try to open the default webcam
if not cap.isOpened():
    print("Video file 'mavideo.mp4' not found. Trying to open webcam...")
    cap = cv2.VideoCapture(0)

# Check if the video stream or webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream or webcam.")
    exit()

# --- 3. Using the background subtraction technique ---
# Initialize the background subtractor (MOG2)
# This will be used to create a binary mask of moving objects
subbg = cv2.createBackgroundSubtractorMOG2()

# --- Main loop to process each frame of the video ---
while True:
    # --- 2. Read each frame from the webcam or video ---
    ret, frame = cap.read()

    # If the frame is not read correctly (e.g., end of video), break the loop
    if not ret:
        break

    # Apply the background subtraction to the current frame
    # This generates a binary mask where white pixels represent movement
    mask = subbg.apply(frame)

    # --- 4. Noise reduction in the binary mask ---
    # Erode the mask to remove small white pixels (noise)
    mask = cv2.erode(mask, None, iterations=2)
    # Dilate the mask to restore the size of the remaining white areas
    mask = cv2.dilate(mask, None, iterations=2)

    # --- 5. Obtain the contours ---
    # Find the contours (outlines) of the white areas in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- 6. Draw the rectangles on the image and display the result ---
    # Loop through each contour found
    for contour in contours:
        # Filter out small contours to avoid detecting noise as objects
        if cv2.contourArea(contour) < 500:
            continue

        # Get the coordinates of the bounding rectangle for the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        # Draw a green rectangle on the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the original video frame with the detected objects
    cv2.imshow('Video frame', frame)
    # Display the binary mask to visualize the detected movement
    cv2.imshow('The binary mask', mask)

    # Wait for the user to press the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Clean up and release resources ---
# Release the video capture object
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
