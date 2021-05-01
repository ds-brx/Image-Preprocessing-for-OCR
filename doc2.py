import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import array, plot, show, axis, arange, figure, uint8
import pytesseract


doc= cv2.imread("doc5.jpeg")

def filtering(image):
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,191,50)
    maxIntensity = 255.0
    x = arange(maxIntensity) 
    phi = 1
    theta = 1
    newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**0.5
    newImage0 = array(newImage0,dtype=uint8)
    y = (maxIntensity/phi)*(x/(maxIntensity/theta))**0.5
    newImage1 = (maxIntensity/phi)*(image/(maxIntensity/theta))**2
    newImage1 = array(newImage1,dtype=uint8)
    text = (pytesseract.image_to_string(newImage1))
    text = " ".join(text.split('\n'))
    return newImage0, newImage1,text


def getSkewAngle(cvImage) -> float:
    gray = cvImage.copy()
    if len(gray.shape) != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    dilate = cv2.dilate(thresh, kernel, iterations=5)
    _,contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)




newImage0, newImage1,text = filtering(doc)
cv2.imwrite("newImage0.png",newImage0)

angle = getSkewAngle(doc)
rotated_img = rotateImage(doc,angle)
cv2.imwrite("rotated.png", rotated_img)

angle = getSkewAngle(newImage0)
rotated_img = rotateImage(newImage0,angle)
cv2.imwrite("rotated-contrasted.png", rotated_img)

angle = getSkewAngle(newImage0)
skewed_img = deskew(newImage0)
cv2.imwrite("skewed-contrasted.png", skewed_img)

angle = getSkewAngle(doc)
skewed_img = deskew(doc)
cv2.imwrite("skewed.png", skewed_img)


kernel = np.ones((2,2), np.uint8)
img_erosion = cv2.erode(newImage0, kernel, iterations=1)
img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
cv2.imwrite("erosion.png",img_erosion)
cv2.imwrite("dilation.png",img_dilation)





