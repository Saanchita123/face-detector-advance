import cv2
import dlib
import numpy as np

# Load the pre-trained dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib repository

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Predict landmarks
        landmarks = predictor(gray, face)
        landmarks_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
        
        # Draw landmarks on the face
        for point in landmarks_points:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
        
        # Calculate angles based on key landmarks (e.g., eyes, nose, chin)
        left_eye = np.mean(landmarks_points[36:42], axis=0)
        right_eye = np.mean(landmarks_points[42:48], axis=0)
        nose = landmarks_points[30]
        chin = landmarks_points[8]

        # Example: Calculate angle between the eyes and nose
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Label the face with the detected angle
        cv2.putText(frame, f"Angle: {angle:.2f}", (face.left(), face.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Confirm it's a human face based on detection and angles
        if -45 <= angle <= 45:  # Example condition for frontal or tilted face
            cv2.putText(frame, "Human Detected", (face.left(), face.top() - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Face Angled", (face.left(), face.top() - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame

# Start the webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Process the frame
    processed_frame = detect_face(frame)
    
    # Show the processed frame
    cv2.imshow("Live Face Detection", processed_frame)

    # Exit on pressing 'ESC'
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
