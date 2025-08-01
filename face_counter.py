import cv2

# Load pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

# Initialize the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set the camera resolution to reduce processing load
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize the frame for faster processing
    frame_resized = cv2.resize(frame, (320, 240))

    # Convert frame to grayscale for better detection performance
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Count number of faces detected
    count = len(faces)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the count on the frame
    cv2.putText(frame_resized, f"People Count: {count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the processed frame
    cv2.imshow('Camera Feed', frame_resized)

    # Exit loop if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
