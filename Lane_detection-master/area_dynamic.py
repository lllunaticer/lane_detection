import numpy as np
import cv2

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    '''Applies a Gaussian Noise Kernel'''  # 高斯模糊
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255  # 涂黑

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)  # 与操作
    return masked_image

def weighted_img(img, initial_img, a=0.8, b=1, r=0.0):
    return cv2.addWeighted(initial_img, a, img, b, r)

def yellow_enhance(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([40,100,20])
    upper_yellow = np.array([100,255,255])
    mask = cv2.inRange(img_hsv,lower_yellow,upper_yellow)
    gray = grayscale(img_rgb)
    return weighted_img(mask,gray,a=1.,b=1.,r=0.)

def show_img(ax,img,cmap,title):
    if cmap=='gray':
        ax.imshow(img,cmap='gray')
    else:
        ax.imshow(img)
    ax.axis('off')
    ax.set_title(title)

def pipeline(img ,vertices):
    # convert to grayscale + enhance yellow-ish tone
    gray = yellow_enhance(img)
    # remove /cleanup noise
    gray_blur = gaussian_blur(gray ,3)
    # dilate features for large gap between edges line
    gray =cv2.dilate(gray, (3, 3), iterations=10)

    # edges = canny(gray, 94, 63)
    edges = canny(gray, 94, 63)

    masked = region_of_interest(edges, vertices)

    return masked

def on_area():
    first_1 = cv2.getTrackbarPos("1_1", "panel")
    first_2 = cv2.getTrackbarPos("1_2", "panel")
    second_1 = cv2.getTrackbarPos("2_1","panel")
    second_2 = cv2.getTrackbarPos("2_2", "panel")
    third_1 = cv2.getTrackbarPos("3_1", "panel")
    third_2 = cv2.getTrackbarPos("3_2", "panel")
    four_1 = cv2.getTrackbarPos("4_1", "panel")
    four_2 = cv2.getTrackbarPos("4_2", "panel")
    # five = cv2.getTrackbarPos("5", "panel")

    imshape = img1.shape

    # first_1 = first_1/100
    # first_2 = first_2/100
    # second_1 = second_1/100
    # second_2 = second_2/100
    # third_1 = third_1/100
    # third_2 = third_1/100
    # four_1 = four_1/100
    # four_2 = four_2/100

    vertices = np.array([[(imshape[1]*first_1/100,imshape[0]*first_2/100), (imshape[1]*second_1/100,imshape[0]*second_2/100),\
                         (imshape[1]*third_1/100,imshape[0]*third_2/100),(imshape[1]*four_1/100,imshape[0]*four_2/100), \
                          ]], dtype=np.int32)
    # vertices = np.array([[(imshape[1] * 0.4, imshape[0] * 0.55), (imshape[1] * 0.75, imshape[0] * 0.65), (imshape[1] * 0.80, imshape[0] * 0.8), (imshape[1] * 0.01, imshape[0] * 0.8),]], dtype=np.int32)
    masked = pipeline(img1,vertices)

    cv2.imshow("s", masked)

img1_name = 'test_images/bridge2.jpg'
img1 = cv2.imread(img1_name)

cv2.namedWindow("s", cv2.WINDOW_NORMAL)
# cv2.namedWindow("p", cv2.WINDOW_NORMAL)

cv2.namedWindow("panel",cv2.WINDOW_NORMAL)
cv2.createTrackbar("1_1", "panel", 43, 100, on_area)
cv2.createTrackbar("1_2", "panel", 44, 100, on_area)
cv2.createTrackbar("2_1", "panel", 78, 100, on_area)
cv2.createTrackbar("2_2", "panel", 85, 100, on_area)
cv2.createTrackbar("3_1", "panel", 90, 100, on_area)
cv2.createTrackbar("3_2", "panel", 86, 100, on_area)
cv2.createTrackbar("4_1", "panel", 8, 100, on_area)
cv2.createTrackbar("4_2", "panel", 78, 100, on_area)
# cv2.createTrackbar("5", "panel", 0, 100, on_area)

while(1):
    on_area()
    k = cv2.waitKey(1)&0xFF
    if k == 27:
        break
