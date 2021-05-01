import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import array, plot, show, axis, arange, figure, uint8
import pytesseract

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def alignImages(im1, im2):

  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY) 
  orb = cv2.ORB_create(MAX_FEATURES)

  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

  matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = matcher.match(descriptors1, descriptors2, None)
  matches = sorted(matches, key = lambda x:x.distance)
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
  return im1Reg, h

def filtering(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,49,21)
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


ref = cv2.imread("c1 copy.png")
test1 = cv2.imread("c1.png")

imReg, h = alignImages(test1, ref)
img1, img2, text = filtering(imReg)
cv2.imwrite("aligned.jpg", img2)
print(text)



