import cv2
import numpy as np

img = cv2.imread("night1.jpg")

def on_canny():
    min = cv2.getTrackbarPos("min", "s")
    max = cv2.getTrackbarPos("max", "s")
    edges = cv2.Canny(img, min, max)
    cv2.imshow("s", edges)

cv2.namedWindow("s")
cv2.createTrackbar("min", "s", 0,255, on_canny)
cv2.createTrackbar("max", "s", 0,255, on_canny)

while(1):
    on_canny()
    k = cv2.waitKey(1)&0xFF
    if k == 27:
        break