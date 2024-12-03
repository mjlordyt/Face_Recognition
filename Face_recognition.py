import cv2 
import face_recognition 
import numpy as np

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0) 
# Load images and learn how to recognize them
known_images = {
    "URK21DS3021": face_recognition.load_image_file("urk21ds3021.jpg"),
    "URK21DS3019": face_recognition.load_image_file("URK21DS3019.jpg"),
    "URK21DS3027": face_recognition.load_image_file("URK21DS3027.jpg"),
    "URK21DS3011": face_recognition.load_image_file("URK21DS3011.jpg")
}

known_encodings = {}
for name, image in known_images.items():
    encoding = face_recognition.face_encodings(image)[0]
    known_encodings[name] = encoding

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(list(known_encodings.values()), face_encoding)
        name = "Unknown"
        color = (0, 0, 255)  # Red by default

        # If a match was found in known_face_encodings, use the first one
        if True in matches:
            first_match_index = matches.index(True)
            name = list(known_encodings.keys())[first_match_index]
            color = (0, 255, 0)  # Green for known people

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()