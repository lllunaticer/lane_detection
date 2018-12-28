import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def grayscale(img):
    '''
    灰度转换，返回只有一个颜色通道的图像
    '''
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # if you read an image with cv.imread(),
    # return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    '''Applies a Gaussian Noise Kernel'''  # 高斯模糊
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    '''
    Apply an image mask

    '''
    # define a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill
    # the mask with depending on the input image

    # Image properties include number of rows, columns
    # and channels, type of image data, number of pixels etc.
    # Shape of image is accessed by img.shape. It returns a
    # tuple of number of rows, columns and channels (if image is color):
    # If image is grayscale, tuple returned contains only
    # number of rows and columns. So it is a good method to
    # check if loaded image is grayscale or color image.
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255  # 涂黑

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)  # 与操作
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    '''
    Note:
    '''
    # print(lines)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def average_lines(img, lines, y_min, y_max):
    # return coordinates of the averaged lines
    # 通过直线拟合的方法返回左右车道线
    hough_pts = {'m_left': [], 'b_left': [], 'norm_left': [], 'm_right': [],
                 'b_right': [], 'norm_right': []}
    if lines != None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # 多项式拟合函数，参数“1”表示一次拟合，即直线拟合
                m, b = np.polyfit([x1, x2], (y1, y2), 1)
                # ** 表示 幂次运算
                norm = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if m > 0:  # 斜率right
                    hough_pts['m_right'].append(m)
                    hough_pts['b_right'].append(b)
                    hough_pts['norm_right'].append(norm)
                if m < 0:
                    hough_pts['m_left'].append(m)
                    hough_pts['b_left'].append(b)
                    hough_pts['norm_left'].append(norm)
    if len(hough_pts['b_left']) != 0 or len(hough_pts['m_left']) != 0 \
            or len(hough_pts['norm_left']) != 0:
        b_avg_left = np.mean(np.array(hough_pts['b_left']))
        m_avg_left = np.mean(np.array(hough_pts['m_left']))
        xmin_left = int((y_min - b_avg_left) / m_avg_left)
        xmax_left = int((y_max - b_avg_left) / m_avg_left)
        left_lane = [[xmin_left, y_min, xmax_left, y_max]]
        left_norm = max(hough_pts['norm_left'])
    else:
        left_lane = [[0, 0, 0, 0]]
        left_norm = 0
    if len(hough_pts['b_right']) != 0 or len(hough_pts['m_right']) != 0 \
            or len(hough_pts['norm_right']) != 0:
        b_avg_right = np.mean(np.array(hough_pts['b_right']))
        m_avg_right = np.mean(np.array(hough_pts['m_right']))
        xmin_right = int((y_min - b_avg_right) / m_avg_right)
        xmax_right = int((y_max - b_avg_right) / m_avg_right)
        right_lane = [[xmin_right, y_min, xmax_right, y_max]]
        right_norm = max(hough_pts['norm_right'])
    else:
        right_lane = [[0, 0, 0, 0]]
        right_norm = 0
    return [left_lane, right_lane, left_norm, right_norm]


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    '''
    img is the output of canny tranform
    return an image with hough lines draw
    '''
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # print(lines.shape)
    # line_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    # draw_lines(line_img,lines)
    return lines


def weighted_img(img, initial_img, a=0.8, b=1, r=0.0):
    return cv2.addWeighted(initial_img, a, img, b, r)

def show_img(ax,img,cmap,title):
    if cmap=='gray':
        ax.imshow(img,cmap='gray')
    else:
        ax.imshow(img)
    ax.axis('off')
    ax.set_title(title)

def bypass_angle_filter(lines,low_thres,hi_thres):
    '''
    前面的代码中，我们也实现了一个角度滤波器，但是这两个函数还是有区别的，
    前面的函数只有一个阈值，而这个函数有两个阈值，并只保留阈值之间的角度
    '''
    filtered_lines = []
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                angle = abs(np.arctan((y2-y1)/(x2-x1))*180/np.pi)
                if angle > low_thres and angle < hi_thres:
                    filtered_lines.append([[x1,y1,x2,y2]])
    return filtered_lines


def locationDectection(lines, median_band_left, median_band_right):
    caution_lines = []

    left_line = lines[0]
    right_line = lines[1]
    left_center = [(left_line[0][0]+left_line[0][2])/2, (left_line[0][1]+left_line[0][3])/2]
    right_center = [(right_line[0][0]+right_line[0][2])/2, (right_line[0][1]+right_line[0][3])/2]

    if left_center[0] > median_band_left and left_center[0]<median_band_right:
        caution_lines.append(left_line)

    if right_center[0]>median_band_left and right_center[0]<median_band_right:
        caution_lines.append(right_line)
    # if lines[0] != [0,0,0,0] and lines[1]!= [0,0,0,0]:
    #     median_line = (lines[0]+lines[1])/2
    #     if (median_line[0] > median_band_left and median_line[0] < median_band_right) or\
    #     (median_line[2] > median_band_left and median_line[2] < median_band_right):
    #         if
    # else:
    #     if lines[0] == [0,0,0,0] and lines[1]!= [0,0,0,0]:
    #         caution_lines.append(lines[1])
    #     else:
    #         if lines[0] != [0,0,0,0] and lines[1]== [0,0,0,0]:
    #             caution_lines.append(lines[0])
    # if lines is not None:
    #     for line in lines:
    #         for x1, y1, x2, y2 in line:
    #             if y1 > y2:
    #                 x, y = [x1, y1]
    #             else:
    #                 x, y = [x2, y2]
    #             if x > median_band_left and x < median_band_right:
    #                 if y > median_band:
    #                     caution_lines.append([[x1, y1, x2, y2]])
                # median_point = [(x1+x2)/2, (y1+y2)/2]
                # if median_point[0]>median_band_left and median_point[0]<median_band_right:
                #     caution_lines.append([[x1,y1,x2,y2]])
    return caution_lines

def dash_solid(average_lines):

    left_line = average_lines[0]
    right_line = average_lines[1]

    left_average_norm = average_lines[2]
    right_average_norm = average_lines[3]

    print("左长： "+str(left_average_norm))
    print("右长： " + str(right_average_norm))

def yellow_detection(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 0, 100])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    res = cv2.bitwise_and(img, img, mask=mask)
    return res



def yellow_enhance(img_rgb):
    '''
    该函数将rgb中淡黄色转换成白色，主要分三步：
    step1:convert rgb to hsv
    step2:create a lower/upper range of hsv
    step3:create a mask
    '''
    img_hsv = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([40,100,20])
    upper_yellow = np.array([100,255,255])
    mask = cv2.inRange(img_hsv,lower_yellow,upper_yellow)
    gray = grayscale(img_rgb)
    return weighted_img(mask,gray,a=1.,b=1.,r=0.)



def pipeline(img ,vertices ,low_thres ,hi_thres ,hline_show):
    # convert to grayscale + enhance yellow-ish tone
    gray = yellow_enhance(img)
    yellow_res = yellow_detection(img)
    # remove /cleanup noise
    gray_blur = gaussian_blur(gray ,3)
    # dilate features for large gap between edges line
    gray =cv2.dilate(gray, (3, 3), iterations=10)

    edges = canny(gray, 94, 63)
    # edges = canny(gray, 64, 38)
    # 白天：edges = canny(gray, 94, 63)
    # 夜晚（光线好）：edges = canny(gray, 70, 28)
    # 夜晚（光线差）：edges = canny(gray, 48, 24)
    imshape = img.shape
    masked = region_of_interest(edges, vertices)

    h_lines = hough_lines(masked, rho=1, theta=np.pi / 180, threshold=30,
                          min_line_len=10, max_line_gap=10)
    # Angle High Pass filter
    h_lines = bypass_angle_filter(h_lines, low_thres, hi_thres)
    avg_hlines = average_lines(img, h_lines, int(img.shape[0] * 0.65), img.shape[0])
    dash_solid(avg_hlines)

    if hline_show['caution'] == 'on':
        clines_img = np.zeros(imshape, dtype=np.uint8)
        clines = locationDectection(avg_hlines, imshape[1] * 0.40, imshape[1] * 0.60)
        draw_lines(clines_img, clines, color=[0, 255, 0], thickness=15)
    else:
        clines_img = np.zeros(imshape, dtype=np.uint8)


    if hline_show['hlines'] == 'on':
        hlines_img = np.zeros(imshape, dtype=np.uint8)
        draw_lines(hlines_img, h_lines, color=[255, 0, 0], thickness=10)
    else:
        hlines_img = np.zeros(imshape, dtype=np.uint8)


    # averaging lines
    if hline_show['avg'] == 'on':
        avg_img = np.zeros(imshape, dtype=np.uint8)
        avg_line = [avg_hlines[0], avg_hlines[1]]
        draw_lines(avg_img, avg_line, color=[255, 0, 0], thickness=10)
    else:
        avg_img = np.zeros(imshape, dtype=np.uint8)
    # Display result of each step of the pipeline
    if hline_show['steps'] == 'on':
        _, ax = plt.subplots(2, 3, figsize=(20, 10))
        show_img(ax[0, 0], img, None, 'original_img')
        show_img(ax[0, 1], gray, 'gray', 'Apply grayscale')
        show_img(ax[0, 2], gray_blur, 'gray', 'Apply Gaussian Blur')
        show_img(ax[1, 0], edges, 'gray', 'Apply Canny')
        show_img(ax[1, 1], masked, 'gray', 'Apply mask')
        show_img(ax[1, 2], yellow_res, 'rgb', 'yellow')
        plt.show()
    img_all_lines = weighted_img(hlines_img, img, a=1., b=0.8, r=0.)
    img_all_lines = weighted_img(avg_img, img_all_lines, a=1., b=0.8, r=0.)
    img_all_lines = weighted_img(clines_img, img_all_lines, a=1., b=0.8, r=0.)
    return img_all_lines


img_name = 'test_images/shadow1.jpg'
# reading in an imag
img = cv2.imread(img_name)
hline_show = {'hlines': 'off', 'avg': 'on', 'steps': 'on', 'caution':'on'}
imshape = img.shape
print(imshape)
# (1080, 1920, 4)
# vertices = np.array([[(imshape[1]*0.49,imshape[0]*0.53),(imshape[1]*0.85,imshape[0]*0.65),\
#                          (imshape[1]*0.71,imshape[0]*0.85),(imshape[1]*0.08,imshape[0]*0.78),\
#                          ]], dtype=np.int32)
# day_bold
vertices = np.array([[(imshape[1]*0.39,imshape[0]*0.53),(imshape[1]*1,imshape[0]*0.68),\
                         (imshape[1]*0.62,imshape[0]*0.85),(imshape[1]*0.12,imshape[0]*0.77),\
                         ]], dtype=np.int32)
# (此位置控制左上角越小越向左，此位置控制左上角越小越向上)，(此位置控制右上角越大越向上, 此位置控制右上角越大越靠左)，
# (此位置控制右下角越大越往右，此位置控制右下角越大越往下)，(此位置控制左下角)
low_thres, hi_thres = [15,80]
lines_img = pipeline(img, vertices, low_thres, hi_thres, hline_show)
plt.imshow(lines_img)
plt.show()