import cv2
import numpy as np


def nothing(x):
    pass


g_CannyThred = 50
g_CannyP = 1
g_HoughThred = 40
rawimg = cv2.imread('2018-11-25.jpg')
gray = cv2.cvtColor(rawimg, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(gray, (3, 3), 0)
result = img.copy()
cv2.namedWindow('hough_demo')
cv2.createTrackbar("CannyThred", "hough_demo", g_CannyThred, 255, nothing)
cv2.createTrackbar("g_CannyP", "hough_demo", g_CannyP, 100, nothing)
cv2.createTrackbar("g_HoughThred", "hough_demo", g_HoughThred, 200, nothing)
while (1):
    cv2.imshow('hough_demo', result)
    result = img.copy()
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        break
    g_CannyThred_1 = cv2.getTrackbarPos('CannyThred', 'hough_demo')
    g_CannyP_1 = cv2.getTrackbarPos('g_CannyP', 'hough_demo')
    g_HoughThred_1 = cv2.getTrackbarPos('g_HoughThred', 'hough_demo')
    cannyImage = cv2.Canny(img, (float)(g_CannyThred_1), (float)((g_CannyThred_1 + 1) * (2 + g_CannyP_1 / 100.0)),
                           apertureSize=3)
    HoughLines = cv2.HoughLines(cannyImage, 1, np.pi / 180, g_HoughThred_1 + 1)
    for line in HoughLines[0]:
        #         if line==None:
        #             break
        rho = line[0]  # 第一个元素是距离rho  
        theta = line[1]  # 第二个元素是角度theta  
        if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线  
            # 该直线与第一行的交点  
            pt1 = (int(rho / np.cos(theta)), 0)
            # 该直线与最后一行的焦点  
            pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
            # 绘制一条白线  
            cv2.line(result, pt1, pt2, (255))
        else:  # 水平直线  
            #  该直线与第一列的交点  
            pt1 = (0, int(rho / np.sin(theta)))
            # 该直线与最后一列的交点  
            pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
            # 绘制一条直线  
            cv2.line(result, pt1, pt2, (255), 1)

cv2.destroyAllWindows()

