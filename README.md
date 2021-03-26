
# Face Mask Detector 
#### Strive School Build Week 3 project

![](/work/faceDetected.jpg)

# Contributors List
* [Bence Kovacs](https://github.com/kovacsbelsen)
* [Lorenzo Demiri](https://github.com/lorenzodemiri)
* [Michal Podlaszuk](https://github.com/MichalPodlaszuk)
* [Agnese Marchisella Navarro](https://github.com/comicodex)

# Introduction
Considering the state of the world today with thousands of people dying everyday due to the Covid-19 pandemic, we decided to help a little by creating this project. Our goal is to train a custom deep learning model to detect whether a person is wearing or not wearing a face mask and if he's wearing it correctly, that works both on images and in real-time video streams.

# Approach
To accomplish this task, we utilize Computer Vision and Deep Learning modules.

# Tasks
## Creation of a custom dataset
* We start by creating our own custom dataset, capturing images with our desktop camera and the OpenCV library wearing MASK, NO MASK and BAD MASK.
* We then proceed with annotating and labelling facial landmarks by drawing a rectangular bounding box around the face and keypoints on the eyes.
* Finally, we merge all images (our own and our peers') in the respective mask, no mask and bad mask folders for the creation of a larger dataset to feed our training model.

## Pre-processing the images
* Our images need to be pre-processed in order for the model to learn from them and properly classify them.
* With Computer Vision techniques, we convert the color from BGR to grayscale and resize them to 224x224 pixels.

## Training the model and Fast.AI
We start tackling our classification problem by training different pre-trained classifiers but soon switch our attention to [Fast.AI](https://docs.fast.ai/index.html), a higher-level built in library in Pytorch, which gives us the best accuracy results (96%).  

<br>





