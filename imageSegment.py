import numpy as np
from cv2 import grabCut, GC_INIT_WITH_RECT, rectangle
from matplotlib import pyplot as plt
import time

time1 = time.time()
# Initialize Info
img = plt.imread("cat_1.jpg")
mask = np.zeros(img.shape[:2],dtype="uint8")

fgdModel = np.zeros((1,65), dtype="float")
bgdModel = np.zeros((1,65), dtype="float")

# Rectangle Information (replace rect with bounding box from object detection)
rect = (69, 22, 338, 414) # rectangle around image, form: (x1,y1,x2,y2) top left, bottom right
# watermelon = (140, 15, 400, 290)


start = rect[:2]
end = rect[2:]
color = (0,0,255)
thickness = 2
plotrect = rectangle(img.copy(), start, end, color, thickness)

# Perform Grabcut
grabCut(img,mask,rect,bgdModel,fgdModel, iterCount=5, mode=GC_INIT_WITH_RECT) # lower itercount for faster
mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')

imgSeg = img*mask[:,:,np.newaxis]

# Check timing of algo
time2 = time.time() - time1
print(time2)

# Show Images
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(img)
ax2.imshow(plotrect)
ax3.imshow(imgSeg)
plt.show()


