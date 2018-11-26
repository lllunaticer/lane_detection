import cv2
import numpy as np

img = cv2.imread("2018-11-25.jpg")

def on_canny():
    x = cv2.getTrackbarPos("x", "s")
    y = cv2.getTrackbarPos("y", "s")
    edges = cv2.circle(img,(x,y),10,(0,0,255),-1)
    cv2.imshow("s", edges)

cv2.namedWindow("s", cv2.WINDOW_GUI_NORMAL)
cv2.createTrackbar("x", "s", 0,2000, on_canny)
cv2.createTrackbar("y", "s", 0,2000, on_canny)

while(1):
    on_canny()
    k = cv2.waitKey(1)&0xFF
    if k == 27:
        break