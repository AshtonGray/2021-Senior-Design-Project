import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5) # less thresh = more objects detected

# create camera object
camera = jetson.utils.gstCamera(640,480, "csi://0") # check with ls /dev/video* or v4l2-ctl --list-devices , v4l2-ctl --device /dev/video0 --list-formats-ext

display = jetson.utils.glDisplay()

while display.IsOpen():
	img, width, height = camera.CaptureRGBA()
	detections = net.Detect(img, width, height) # gives bounding box coordinates as well (need for segmentation)
	display.RenderOnce(img, width, height)
	display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

