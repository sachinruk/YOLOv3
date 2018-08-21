# YOLO: Keras vs OpenCV

This is a slight modification of https://github.com/xiaochus/YOLOv3 to accept different config files. For instance this version can run tiny yolo.

## Installation
Simply run:
```
make
```
This will download all necessary config files, weights, and compile them to keras format.

## Demos
To run the keras demo do:
```
python demo.py
```
For the opencv version run:
```
python demo_opencv.py
```
This will save the results in folder `images/res/`.

As pointed out by Satya Mallick in [this post](https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/) the opencv is significantly faster (atleast 5x). The speedup may be partly due to the fact that in opencv version the image is read in with channels first (facilitated by `cv.dnn.blobFromImage`).

It should be noted that the keras version can run on a Nvidia GPU if available, while the openCV support for this seems experimental.