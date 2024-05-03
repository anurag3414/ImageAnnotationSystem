import streamlit as st
import numpy as np
import cv2

# Function to perform object detection
def detect_objects(frame, net):
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    return frame

# Load the MobileNet SSD model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Initialize the camera
vs = cv2.VideoCapture(0)
if not vs.isOpened():
    st.error("Cannot open camera.")
    st.stop()

st.title("Real-time Object Detection")

frame_placeholder = st.empty()
capture_button = st.button("Capture Photo")

# Run the video stream
try:
    while True:
        ret, frame = vs.read()
        if not ret:
            st.error("Failed to fetch frame.")
            break

        frame = cv2.resize(frame, (400, 300))
        frame = detect_objects(frame, net)

        frame_placeholder.image(frame, channels="BGR", use_column_width=True)

        if capture_button:
            cv2.imwrite("captured_photo.jpg", frame)
            with open("captured_photo.jpg", "rb") as file:
                btn = st.download_button(
                    label="Download Captured Image",
                    data=file,
                    file_name="captured_photo.jpg",
                    mime="image/jpeg"
                )
            st.success("Photo captured!")
            break
finally:
    vs.release()
    cv2.destroyAllWindows()


