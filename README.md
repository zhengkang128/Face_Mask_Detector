# Face Mask Detector with Python, OpenCV and YOLOv3/4
 A custom trained face mask detector based on YOLO v3/4 network built with Darknet, Python and OpenCV.
 This repository consists of a notebook with detailed steps in training a custom mask detector model with darknet, as well as an inference python script to run detection of images and video files with the trained model with OpenCV.
 
![alt text](https://github.com/zhengkang128/face_mask_detector/blob/main/etc/Screenshot%20from%202021-10-03%2012-41-35.png)
## Project Structure

1. The ```train.ipynb``` notebook consists of detailed steps to train the models.
2. ```darknet``` directory consists of the source code to train our yolo models. This folder will be built in section 2
3. ```data``` folder consists of all of our dataset of both images and annotations (.txt files). Each txt files corresponding to the image share the same name (different ext)
4. ```model_configs``` consists of two folders, ```yolov3-mask``` and ```yolov4-mask```, which contains all configurations of the yolov3/4 models for training and inference. Weights are stored in the backup folder.

For inferencing,

4. ```run_inference.py``` - the python script to run inference on all input images and videos in input directory
5. ```input``` folder consists of all the images and video files to run inference on
6. ```output``` folder consists of all outputs that are produced by ```run_inference.py```. They are seperated into 2 folders, ```yolov3``` and ```yolov4```, depending on which model is used for inference.


```
face_mask_detector
├── darknet
│   └── <darknet_source_code>
├── data
│   ├── 0.jpg
│   ├── 0.txt
│   ├── 1.jpeg
│   ├── 1.txt
│   └── <dataset_labels_and_images>
├── input
│   ├── test-image1.jpg
│   ├── test-image2.jpg
│   ├── test-image3.jpg
│   ├── test-image4.jpg
│   ├── test-video1.mp4
│   └── test-video2.mp4
├── model_configs
│   ├── yolov3_mask
│   │   ├── backup
│   │   │   └── yolov3-mask-train_best.weights
│   │   └── <yolov4_config_files>
│   └── yolov4_mask
│       ├── backup
│       │   └── yolov4-mask-train_best.weights
│       └── <yolov4_config_files>
├── output
│   ├── yolov3
│   │   ├── test-image1.jpg
│   │   ├── test-image2.jpg
│   │   ├── test-image3.jpg
│   │   ├── test-image4.jpg
│   │   ├── test-video1.mp4
│   │   └── test-video2.mp4
│   └── yolov4
│       ├── test-image1.jpg
│       ├── test-image2.jpg
│       ├── test-image3.jpg
│       ├── test-image4.jpg
│       ├── test-video1.mp4
│       └── test-video2.mp4
├── run_inference.py
└── train.ipynb
```

## Installation
### Clone Repo
```
git clone https://github.com/zhengkang128/face_mask_detector.git
```
### Download pre-trained weights for inferencing
```yolov3-mask-train_best.weights```: https://drive.google.com/file/d/1udKe-qGq1MF2mIuglyIk9MMRpMdFsxs7/view?usp=sharing 

```yolov4-mask-train_best.weights```: https://drive.google.com/file/d/1VsIB3OkfRoj-Brag568Hm6hLrOZZPmUK/view?usp=sharing

Put ```yolov3-mask-train_best.weights``` in ```model_configs/yolov3_mask/backup directory```

Put ```yolov3-mask-train_best.weights``` in ```model_configs/yolov4_mask/backup directory```

## Install dependencies
```
pip3 install -r requirements.txt
```
## Training Instruction
Refer to ```train.ipynb``` for detailed instructions to train yolov3/4 mask detector 

## Run Inference

Run the code below to perform yolov3-mask inference on all images and videos in ```input``` folder. Output will be stored in ```output/yolov3```

```python3 run_inference --model yolov3 --confidence 0.1 --threshold 0.2 --size_img 416 --size_vid 768```

Run the code below to perform yolov4-mask inference on all images and videos in ```input``` folder. Output will be stored in ```output/yolov4```

```python3 run_inference --model yolov4 --confidence 0.1 --threshold 0.2 --size_img 416 --size_vid 768```

## Future Plans
1. Real-Time Inference for mask detection
2. Integrate tracking algorithm for counting objects (DeepSORT)

## Inference Output (yolov4-mask)
![alt text](https://github.com/zhengkang128/face_mask_detector/blob/main/etc/test-image1.jpg)
![alt text](https://github.com/zhengkang128/face_mask_detector/blob/main/etc/test-image2.jpg)
![alt text](https://github.com/zhengkang128/face_mask_detector/blob/main/etc/test-image3.jpg)
![alt text](https://github.com/zhengkang128/face_mask_detector/blob/main/etc/test-image4.jpg)





