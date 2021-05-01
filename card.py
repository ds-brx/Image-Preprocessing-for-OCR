import cv2
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


ref = cv2.imread("card_ref.png")
test1 = cv2.imread("card_test.jpg")
test2 = cv2.imread("card_test_2.jpeg")
test1 = cv2.rotate(test1,cv2.ROTATE_90_COUNTERCLOCKWISE)

##rotate and match

ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
test1_gray = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
test2_gray = cv2.cvtColor(test2, cv2.COLOR_BGR2GRAY)


orb = cv2.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(test1_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(ref_gray, None)

# matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
# matches = matcher.match(descriptors1, descriptors2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(descriptors1,descriptors2, k=2)

# matches.sort(key=lambda x: x.distance, reverse=False)
# numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
# matches = matches[:numGoodMatches]
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)


# imMatches = cv2.drawMatchesKnn(test1_gray, keypoints1, ref_gray, keypoints2, good, None)
# cv2.imwrite("matches.jpg", imMatches)


points1 = np.zeros((len(good), 2), dtype=np.float32)
points2 = np.zeros((len(good), 2), dtype=np.float32)

for i,match in enumerate(good):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

h, mask = cv2.findHomography(points1, points2)

height, width = ref_gray.shape
alignedImg = cv2.warpPerspective(test1_gray, h, (width, height))

cv2.imwrite("img.jpg", alignedImg)


for m,n in matches:
    if m.distance < 0.75*n.distance:
        print(keypoints1[m.queryIdx].pt)







