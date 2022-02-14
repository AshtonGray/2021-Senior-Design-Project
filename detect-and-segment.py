#!/usr/bin/python3

import jetson.inference
import jetson.utils

import argparse
import sys

import numpy as np
from cv2 import grabCut, GC_INIT_WITH_RECT, rectangle
from matplotlib import pyplot as plt
import time

# ------ parse the command line ------
time1 = time.time()

parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# create video output object 
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
	
# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create video sources
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)


# process frames until the user exits
while True:
	# capture the next image
	img = input.Capture()

	# detect objects in the image (with overlay)
	detections = net.Detect(img, overlay=opt.overlay)

	# print the detections
	#print("detected {:d} objects in image".format(len(detections)))

	for detection in detections:
		print(detection)
		left = int(round(detection.Left))
		top = int(round(detection.Top))
		right = int(round(detection.Right))
		bottom = int(round(detection.Bottom))

	# render the image
	output.Render(img)

	# update the title bar
	output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

	# print out performance info
	#net.PrintProfilerTimes()

	time2 = time.time()
	# ------ segment image ------

	# Initialize Info
	img = plt.imread(opt.input_URI)
	imgdetection = plt.imread(opt.output_URI)
	mask = np.zeros(img.shape[:2],dtype="uint8")

	fgdModel = np.zeros((1,65), dtype="float")
	bgdModel = np.zeros((1,65), dtype="float")

	# Rectangle Information (replace rect with bounding box from object detection)
	rect = (left, top, right, bottom) # rectangle around image, form: (x1,y1,x2,y2) top left, bottom right
	# watermelon = (140, 15, 400, 290)


	start = rect[:2]
	end = rect[2:]
	color = (0,0,255)
	thickness = 2
	plotrect = rectangle(img.copy(), start, end, color, thickness)

	# Perform Grabcut
	grabCut(img,mask,rect,bgdModel,fgdModel, iterCount=1, mode=GC_INIT_WITH_RECT) # lower itercount for faster
	mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')

	imgSeg = img*mask[:,:,np.newaxis]

	time3 = time.time()

	print("Time for object Detection:",time2 - time1,"Seconds")
	print("Time for Image Segmentation:",time3 - time2,"Seconds")
	print("Total time:",time3 - time1,"Seconds")


	# Show Images
	fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
	ax1.imshow(img)
	ax2.imshow(imgdetection)
	ax3.imshow(plotrect)
	ax4.imshow(imgSeg)
	plt.show()

	# exit on input/output EOS
	if not input.IsStreaming() or not output.IsStreaming():
		break