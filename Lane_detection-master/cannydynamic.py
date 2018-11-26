import cv2
import numpy as np

img = cv2.imread("2018-11-25.jpg")

def on_canny():
    mix = cv2.getTrackbarPos("mix", "s")
    max = cv2.getTrackbarPos("max", "s")
    edges = cv2.Canny(img, mix, max)
    cv2.imshow("s", edges)

cv2.namedWindow("s")
cv2.createTrackbar("mix", "s", 0,255, on_canny)
cv2.createTrackbar("max", "s", 0,255, on_canny)

while(1):
    on_canny()
    k = cv2.waitKey(1)&0xFF
    if k == 27:
        break