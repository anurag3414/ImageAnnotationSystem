
# Embedded Systems - Video Dataset with Annotations


In this project, we applied various methods to generate annotated images of a drone image dataset. The methods that we used were - Roboflow, GAN, YOLOV3 live object annotation and YOLOV3 annotation via uploading image. Let’s dive into the specifications of each method.


## Team Members

- Anurag Singh - B21ES004
- Anupam Singh Bhadouriya - B21CS086

## Instruction to run the image annotation part

This tool provides a graphical user interface for annotating videos using the YOLO (You Only Look Once) object detection algorithm. It allows users to upload a video , convert video into frames and then annotate each frame , and annotate them with bounding boxes.

## Installation

1. create a virtual environment of python
   ```
   python -m venv venv
   source venv/bin/activate  # for Linux/macOS
   venv\Scripts\activate  # for Windows
   
2. install requirement.txt
   ```
    pip install -r requirements.txt
   
3. Install the required weights and config files from the below links and add these files to the root:

   - [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

   - [yolov3.cfg](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg)

   - [coco.names](https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names)

   - [MobileNetSSD_deploy.caffemodel](https://github.com/yash42828/YOLO-object-detection-with-OpenCV/blob/master/real-time-object-detection/MobileNetSSD_deploy.caffemodel)

   - [MobileNetSSD_deploy.prototxt.txt](https://github.com/yash42828/YOLO-object-detection-with-OpenCV/blob/master/real-time-object-detection/MobileNetSSD_deploy.prototxt.txt)
     
4. then run the main file
   ```
   python .\demo.py
   


## Roboflow

Roboflow is an intelligent tool that smartly annotates the images based on various classes that it has been trained on. We annotated 500 videoes  using roboflow by dividing them into frames of 40FPS . Following are the steps :   

### Step 1 : Upload the video

![image](https://github.com/dilip-choudhary1/ImageAnnotationSystem/assets/116138151/fad3814f-cea2-4a53-8a47-4257fc83e25b)


### Step 2 : Choose FPS

![image](https://github.com/dilip-choudhary1/ImageAnnotationSystem/assets/116138151/4b4d070e-c558-4975-9a2e-d691b85be848)



### Step 3 : Choose appropriate confidence intervals

![image](https://github.com/dilip-choudhary1/ImageAnnotationSystem/assets/116138151/6583157e-8daf-4e98-9c33-541774ca5541)



## YOLOV3 Model Live Annotation


YOLOv8 is a cutting-edge object detection algorithm employing a powerful backbone network, feature pyramid, and advanced training techniques to accurately detect objects across various scales. Its efficient architecture and post-processing methods make it a widely utilized solution for real-time applications such as autonomous driving and surveillance.




### Trial Run

<img title="a title" alt="Alt text" src="/assets/6ed32333-71d8-43cc-95bf-1c0a4e404b13.jpg" width=100%>

### Model Dashboard

<img title="a title" alt="Alt text" src="/assets/52e17157-46b4-43fd-9300-ae72d7060a92.jpg" width=100%>


### Live Annotation Output

<img title="a title" alt="Alt text" src="/assets/6e876599-7403-4b3c-9b81-e7bcf5f7427b.jpg" width=100%>




## YOLOV3 Model on an uploaded image

In this model, we will directly upload the video to get the video with annotated boxes and their coordinates


### Dashboard

![image](https://github.com/dilip-choudhary1/ImageAnnotationSystem/assets/116138151/846767ce-3ad2-4461-bf79-19c1432e0d0e)



### Raw video



### Uploaded image Output with annotations






