all:
	wget -N https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -P ./cfg/
	wget -N https://pjreddie.com/media/files/yolov3.weights -P ./weights/

	# get tiny yolo
	wget -N https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg -P ./cfg/
	wget -N https://pjreddie.com/media/files/yolov3-tiny.weights -P ./weights/

	# convert the models to keras models
	python yad2k.py cfg/yolov3.cfg ./weights/yolov3.weights data/yolo.h5
	python yad2k.py cfg/yolov3-tiny.cfg ./weights/yolov3-tiny.weights data/yolov3-tiny.h5
