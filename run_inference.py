import cv2
import numpy as np 
import argparse
import time
import os
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--model", type=str, default="yolov3",
	help="yolov3 or yolov4")
ap.add_argument("-c", "--confidence", type=float, default=0.1,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.2,
	help="threshold when applying non-maxima suppression")
ap.add_argument("-i", "--size_img", type=int, default=416,
	help="Input size for img (multiple of 32)")
ap.add_argument("-v", "--size_vid", type=int, default=768,
	help="Input size for video (multiple of 32)")

args = vars(ap.parse_args())

model_name = args["model"] 
conf_threshold = args["confidence"]
nms_thresh = args["threshold"]
vid_size = args["size_vid"]
img_size=args["size_img"]




def load_yolo(model_name):
	if model_name == "yolov3":
		net = cv2.dnn.readNet("./model_configs/yolov3_mask/backup/yolov3-mask-train_best.weights", "./model_configs/yolov3_mask/yolov3-mask-train.cfg")
	if model_name == "yolov4":
		net = cv2.dnn.readNet("./model_configs/yolov4_mask/backup/yolov4-mask-train_best.weights", "./model_configs/yolov4_mask/yolov4-mask-train.cfg")	
	classes = []
	if model_name == "yolov3":
		names_file = "./model_configs/yolov3_mask/class.names"
	else:
		names_file = "./model_configs/yolov4_mask/class.names"
	with open(names_file, "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	#colors = np.random.uniform(0, 255, size=(len(classes), 3))
	colors = np.array([[0,255,0],[0,0,255]])

	return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers, size):
	img_resized = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)
	blob = cv2.dnn.blobFromImage(img_resized, scalefactor=0.00392, size=(size, size), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width, conf_threshold):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > conf_threshold:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img, conf_threshold, nms_thresh): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, nms_thresh)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			conf_label = round((confs[i] *100),2)
			label = str(classes[class_ids[i]]) + " (" + str(conf_label) +"%)"

			color = colors[class_ids[i]].tolist()
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 3)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 2)
	return img

def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	#img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels

def start_video(video_path, model, classes, colors, output_layers, conf_threshold, nms_thresh,model_name):
	global vid_size
	cap = cv2.VideoCapture('./input/'+video_path)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	fps = cap.get(cv2.CAP_PROP_FPS)
	print(fps)
	out = cv2.VideoWriter('./output/'+ model_name + "/" + video_path,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
	count = 0
	try:
		print(video_path)
		while (cap.isOpened()):
			_, frame = cap.read()
			if np.shape(frame) == () :
				break
			height, width, channels = frame.shape
			blob, outputs = detect_objects(frame, model, output_layers, vid_size)
			boxes, confs, class_ids = get_box_dimensions(outputs, height, width, conf_threshold)
			drawn_frame = draw_labels(boxes, confs, colors, class_ids, classes, frame, conf_threshold, nms_thresh)
			out.write(drawn_frame)
			count+=1
			print("Processing frame", count, "for ", video_path)
	finally:
		cap.release()
		out.release()

def start_image(img_path, model, classes,colors,output_layers, conf_threshold, nms_thresh, model_name):
	global img_size
	img,height,width,channels = load_image('input/' + img_path)
	blob,outputs = detect_objects(img,model, output_layers, img_size)
	boxes, confs, class_ids = get_box_dimensions(outputs, height,width, conf_threshold)
	drawn_img = draw_labels(boxes, confs, colors, class_ids, classes, img, conf_threshold, nms_thresh)
	cv2.imwrite("output/" + model_name + "/" + img_path, drawn_img)

if __name__ == '__main__':
	import time
	net, classes, colors, output_layers = load_yolo(model_name)
	all_files = os.listdir("./input")
	for i, filename in enumerate(all_files):
		t1 = time.time()
		if (".jpg" in filename) or (".jpeg" in filename) or (".png" in filename):
			start_image(filename, net, classes,colors,output_layers, conf_threshold, nms_thresh,model_name)
		elif (".avi" in filename) or (".mp4" in filename):
			start_video(filename, net, classes, colors, output_layers, conf_threshold, nms_thresh,model_name)
		elapsed = (time.time()-t1)
		print("Processed ", filename, " in ", str(elapsed), "seconds")

