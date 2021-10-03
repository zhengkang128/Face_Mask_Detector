# face_mask_detector

## Project Structure
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
```
git clone https://github.com/zhengkang128/face_mask_detector.git
cd face_mask_detector
unzip -a files.zip
rm files.zip
```

## Install dependencies
```
pip3 install -r requirements.txt
```

## Training Instruction
Refer to ```train.ipynb``` for detailed instructions to train yolov3/4 mask detector 
