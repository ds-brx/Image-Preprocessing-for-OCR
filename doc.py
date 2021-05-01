import cv2
import pytesseract 
import numpy as np

img1 = cv2.imread("img4.jpg")
img2 = cv2.imread("img4.jpg")
img3 = cv2.imread("img4.jpg")
img4 = cv2.imread("img4.jpg")

# rotation to be done manually 
# or easy by homographic transofrmation if reference img available
img4 = cv2.rotate(img4,cv2.ROTATE_90_COUNTERCLOCKWISE)

## convert to gray
if len(img4.shape) != 2:
    gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
else:
    gray = img4

_,simple_threshold = cv2.threshold(gray,140,255,cv2.THRESH_BINARY)
_,simple_threshold = cv2.threshold(gray,120,255,cv2.THRESH_TRUNC)
simple_threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,51,7)
simple_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 51, 7)
gray = cv2.bitwise_not(gray)
bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 51, -5)

kernel = np.ones((5,5), np.uint8)
img_erosion = cv2.erode(gray, kernel, iterations=1)
img_dilation = cv2.dilate(gray, kernel, iterations=1)

horizontal = np.copy(gray)
vertical = np.copy(gray)

cols = horizontal.shape[1]
horizontal_size = cols // 30
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
horizontal = cv2.erode(horizontal, horizontalStructure)
horizontal = cv2.dilate(horizontal, horizontalStructure)

rows = vertical.shape[0]
verticalsize = rows // 30
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
vertical = cv2.erode(vertical, verticalStructure)
vertical = cv2.dilate(vertical, verticalStructure)
vertical = cv2.bitwise_not(vertical)


edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 3, -2)
kernel = np.ones((10, 10), np.uint8)
edges = cv2.dilate(edges, kernel)
smooth = np.copy(vertical)
smooth = cv2.blur(smooth, (10, 10))
(rows, cols) = np.where(edges != 0)
vertical[rows, cols] = smooth[rows, cols]

median = cv2.medianBlur(gray,21)

blur = cv2.bilateralFilter(gray,11,50,50)
blur = cv2.GaussianBlur(gray,(15,31),0)


cv2.imshow("median_blur", blur)
cv2.imwrite("output.png", blur)
cv2.waitKey(0) 

# h = img4.shape[0]
# w = img4.shape[1]

# print(h)
# print(w)
