
# Embedded Systems - Drone Dataset with Annotations


In this project, we applied various methods to generate annotated images of a drone image dataset. The methods that we used were - Roboflow, GAN, YOLOV3 live object annotation and YOLOV3 annotation via uploading image. Let’s dive into the specifications of each method.


## Team Members

- Dilip Choudhary - B21ES009
- Vilok Parkhi - B21ES021


## Roboflow

Roboflow is an intelligent tool that smartly annotates the images based on various classes that it has been trained on. We annotated 1000 images using roboflow. Following are the steps :   

### Step 1 : Upload the images

<img title="a title" alt="Alt text" src="/assets/Screenshot 2024-05-02 172821.jpg" width=100%>

### Step 2 : Choose classes that you want to detect

<img title="a title" alt="Alt text" src="/assets/Screenshot 2024-05-02 173050.png" width=100%>


### Step 2 : Choose appropriate confidence intervals

<img title="a title" alt="Alt text" src="/assets/Screenshot 2024-05-02 173227.png" width=100%>




## GAN

GANs, or Generative Adversarial Networks, are a fascinating type of neural network architecture used in machine learning for generating new data. They consist of two neural networks: a generator and a discriminator.
We trained the model on 200+ images and the GAN was able to generate 2800 images with 100 epochs. 

### Code Snippet

<img title="a title" alt="Alt text" src="/assets/WhatsApp Image 2024-05-02 at 15.48.47.jpeg" width=100%>

### GAN Output Image

<img title="a title" alt="Alt text" src="/assets/WhatsApp Image 2024-05-02 at 15.48.48.jpeg" width=100%>


### Step 2 : Choose appropriate confidence intervals

<img title="a title" alt="Alt text" src="/assets/Screenshot 2024-05-02 173227.png" width=100%>


## YOLOV3 Model Live Annotation

YOLOv3 is a popular object detection algorithm that stands for "You Only Look Once version 3." It's an improvement over its predecessors, offering better accuracy and speed. YOLO algorithms are known for their ability to detect objects in images and videos in real-time with a 
single forward pass through a neural network.


### Trial Run

<img title="a title" alt="Alt text" src="/assets/6ed32333-71d8-43cc-95bf-1c0a4e404b13.jpg" width=100%>

### Model Dashboard

<img title="a title" alt="Alt text" src="/assets/52e17157-46b4-43fd-9300-ae72d7060a92.jpg" width=100%>


### Live Annotation Output

<img title="a title" alt="Alt text" src="/assets/6e876599-7403-4b3c-9b81-e7bcf5f7427b.jpg" width=100%>




## YOLOV3 Model on an uploaded image

In this model, we will directly upload the image to get the image with annotated boxes and their coordinates


### Dashboard

<img title="a title" alt="Alt text" src="/assets/ce36895f-1b8c-4239-87ae-ef9ded604dc7.jpg" width=100%>

### Model Dashboard

<img title="a title" alt="Alt text" src="/assets/b0d07e89-25e1-4312-ab08-ead2c74394e3.jpg" width=100%>


### Uploaded image Output with annotations

<img title="a title" alt="Alt text" src="/assets/f0a7bf8c-8eb0-405c-977c-ed5bb68274c2.jpg" width=100%>




