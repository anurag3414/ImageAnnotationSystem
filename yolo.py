# # USAGE
# # python yolo.py --image images/baggage_claim.jpg
# # import the necessary packages
# import numpy as np
# import argparse
# import time
# import cv2
# import os

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
# 	help="minimum probability to filter weak detections")
# ap.add_argument("-t", "--threshold", type=float, default=0.3,
# 	help="threshold when applyong non-maxima suppression")
# args = vars(ap.parse_args())

# # load the COCO class labels our YOLO model was trained on
# labelsPath = 'coco.names'
# LABELS = open(labelsPath).read().strip().split("\n")

# # initialize a list of colors to represent each possible class label
# np.random.seed(42)
# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
# 	dtype="uint8")

# # derive the paths to the YOLO weights and model configuration
# weightsPath = 'yolov3.weights'
# configPath = 'yolov3.cfg'

# # load our YOLO object detector trained on COCO dataset (80 classes)
# print("[INFO] loading YOLO from disk...")
# net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# # load our input image and grab its spatial dimensions
# image = cv2.imread(args["image"])
# (H, W) = image.shape[:2]

# # determine only the *output* layer names that we need from YOLO
# layer_names = net.getLayerNames()
# ln = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# # construct a blob from the input image and then perform a forward
# # pass of the YOLO object detector, giving us our bounding boxes and
# # associated probabilities
# blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
# 	swapRB=True, crop=False)
# net.setInput(blob)
# start = time.time()
# layerOutputs = net.forward(ln)
# # [,frame,no of detections,[classid,class score,conf,x,y,h,w]
# end = time.time()

# # show timing information on YOLO
# print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# # initialize our lists of detected bounding boxes, confidences, and
# # class IDs, respectively
# boxes = []
# confidences = []
# classIDs = []

# # loop over each of the layer outputs
# for output in layerOutputs:
# 	# loop over each of the detections
# 	for detection in output:
# 		# extract the class ID and confidence (i.e., probability) of
# 		# the current object detection
# 		scores = detection[5:]
# 		classID = np.argmax(scores)
# 		confidence = scores[classID]

# 		# filter out weak predictions by ensuring the detected
# 		# probability is greater than the minimum probability
# 		if confidence > args["confidence"]:
# 			# scale the bounding box coordinates back relative to the
# 			# size of the image, keeping in mind that YOLO actually
# 			# returns the center (x, y)-coordinates of the bounding
# 			# box followed by the boxes' width and height
# 			box = detection[0:4] * np.array([W, H, W, H])
# 			(centerX, centerY, width, height) = box.astype("int")

# 			# use the center (x, y)-coordinates to derive the top and
# 			# and left corner of the bounding box
# 			x = int(centerX - (width / 2))
# 			y = int(centerY - (height / 2))

# 			# update our list of bounding box coordinates, confidences,
# 			# and class IDs
# 			boxes.append([x, y, int(width), int(height)])
# 			confidences.append(float(confidence))
# 			classIDs.append(classID)

# # apply non-maxima suppression to suppress weak, overlapping bounding
# # boxes
# idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
# 	args["threshold"])

# # ensure at least one detection exists
# if len(idxs) > 0:
# 	# loop over the indexes we are keeping
# 	for i in idxs.flatten():
# 		# extract the bounding box coordinates
# 		(x, y) = (boxes[i][0], boxes[i][1])
# 		(w, h) = (boxes[i][2], boxes[i][3])

# 		# draw a bounding box rectangle and label on the image
# 		color = [int(c) for c in COLORS[classIDs[i]]]
# 		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
# 		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
# 		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
# 			0.5, color, 2)

# # show the output image
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams['axes.unicode_minus'] = False  # Fix for negative sign display
# resized_image = cv2.resize(image, (800, 600))  # Resize to a smaller size
# plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
# plt.axis('off')  # turn off axis numbers
# plt.show()

# # show the output image
# output_path = 'output.jpg'
# cv2.imwrite(output_path, image)
# print(f"Image saved to {output_path}")



import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

def load_model():
    # Paths to YOLO weights and config
    weightsPath = 'yolov3.weights'
    configPath = 'yolov3.cfg'
    
    # Load the YOLO network
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    return net

def detect_objects(image, net):
    # Load labels
    labelsPath = 'coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")

    # Generate random colors
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # Get dimensions of image
    (H, W) = image.shape[:2]

    # Create blob from image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    # layer_names = net.getLayerNames()
	# ln = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    layer_names = net.getLayerNames()
    ln = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    # Loop over each detection
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

    # Apply non-maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def main():
    st.title("Object Detection in Images")
    st.subheader('Embedded ES-12 Project')
    st.subheader('we can use this tool to annotate any image with objects')
    net = load_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        st.image(image_np, caption='Uploaded Image', use_column_width=True)

        if st.button('Detect Objects'):
            annotated_image = detect_objects(image_np.copy(), net)
            st.image(annotated_image, caption='Detected Objects', use_column_width=True)

            # Save and download annotated image
            result_img = Image.fromarray(annotated_image.astype("uint8"))
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            result_img.save(tmp_file.name)
            with open(tmp_file.name, "rb") as file:
                btn = st.download_button(
                    label="Download Annotated Image",
                    data=file,
                    file_name="detected_image.jpg",
                    mime="image/jpeg"
                )

if __name__ == "__main__":
    main()

