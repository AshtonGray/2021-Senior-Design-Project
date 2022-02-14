# 2021-Senior-Design-Project

An object detection system that given an object to search for, will scan a room, and if the item is visible, will point to it using a laser.

A user interface, run by an ATmega328p, can take in a user input to search for any named item, which is sent via Bluetooth connection to the main system, controlled by the Jetson Nano

The main system, recognizes what object to search for, and takes sectioned images of the room. Each image is run through Nvidia's DetectNet, a deep neural network used through tensorRT for object detection, that is trained on our own data.

Once the object is found, its location is updated in memory, and motors move the camera and laser to point to the object

my-detection.py detects objects in an inputted image, imageSegment.py segments an image into foreground and background using the grabcut algorithm using a user-inputted bounding box. Detect-and-segment.py combines the two by using the coordinates grabbed from the object-detection, and uses them for segmentation
