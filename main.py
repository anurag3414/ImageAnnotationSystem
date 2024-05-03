import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import pandas as pd

def load_mobilenet_ssd():
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
    return net

def load_yolo():
    weightsPath = 'yolov3.weights'
    configPath = 'yolov3.cfg'
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    return net

# Function to perform object detection using MobileNet SSD
def detect_objects_mobilenet(frame, net):
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    detections_list = []

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
            detections_list.append((CLASSES[idx], startX, startY, endX, endY))
    return frame, detections_list

# Function to perform object detection using YOLO
def detect_objects_yolo(image, net):
    labelsPath = 'coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    detections_list = []

    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    ln = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            detections_list.append((LABELS[classIDs[i]], x, y, x + w, y + h))

    return image, detections_list

# Function to download detection data as CSV
def create_downloadable_data(detections_list):
    df = pd.DataFrame(detections_list, columns=["Class", "StartX", "StartY", "EndX", "EndY"])
    csv = df.to_csv(index=False)
    return csv

def main():
    st.title("Object Annotation Dashboard")
    mobilenet_ssd_net = load_mobilenet_ssd()
    yolo_net = load_yolo()

    run = st.checkbox("Run Camera for live annotation")
    FRAME_WINDOW = st.empty()
    camera = cv2.VideoCapture(0)

    # Prepare placeholders for downloads
    img_placeholder = st.empty()
    csv_placeholder = st.empty()

    captured_frame = None
    detections_list = []
    capture_button = st.button("Capture Photo")

    while run and camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to fetch frame.")
            break
        
        frame, detections_list = detect_objects_mobilenet(frame, mobilenet_ssd_net)
        FRAME_WINDOW.image(frame, channels="BGR", use_column_width=True)

        
        if capture_button and run:
            captured_frame = frame.copy()  # Copy the frame to a new variable
            st.success("Photo captured!")
            break

    camera.release()
    cv2.destroyAllWindows()

    if captured_frame is not None:
        # Show download buttons if a photo was captured
        cv2.imwrite("captured_photo.jpg", captured_frame)
        with open("captured_photo.jpg", "rb") as file:
            img_placeholder.download_button(
                label="Download Captured Image",
                data=file,
                file_name="captured_photo.jpg",
                mime="image/jpeg"
            )
        
        csv_data = create_downloadable_data(detections_list)
        csv_placeholder.download_button(
            label="Download Detection Data",
            data=csv_data,
            file_name="detection_data.csv",
            mime="text/csv"
        )

    # Static image detection section
    st.header("Annotate in Uploaded Images")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        st.image(image_np, caption='Uploaded Image', use_column_width=True)

        if st.button('Annotate Objects'):
            
            annotated_image, detections_list = detect_objects_yolo(image_np.copy(), yolo_net)
            st.image(annotated_image, caption='Detected Objects', use_column_width=True)
            csv_data = create_downloadable_data(detections_list)
            result_img = Image.fromarray(annotated_image.astype("uint8"))
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            result_img.save(tmp_file.name)
            with open(tmp_file.name, "rb") as file:
                st.download_button(
                    label="Download Annotated Image",
                    data=file,
                    file_name="detected_image.jpg",
                    mime="image/jpeg",
                    key='download-image2'
                )
            st.download_button(
                "Download Detection Data",
                csv_data,
                "detection_data.csv",
                "text/csv",
                key='download-csv2'
            )

if __name__ == "__main__":
    main()
