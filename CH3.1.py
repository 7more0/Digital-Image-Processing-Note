import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
'''
    Spatial filtering 
    Include:
        Gaussian filtering
        median filtering
        unsharp masking
        Laplace operator
        sobel operator
        canny algorithm
'''


def apply_filter(img, filter):
    # apply filter
    img = img.astype('float32')
    f_img = cv2.filter2D(img, -1, filter)
    return f_img


def gaussian_gen(s, size):
    '''
        generate gaussian filter mask
    :param s: Standard deviation
    :param size: mask size
    :return: mask
    '''
    step = int((size-1)/2)
    gaussian = np.zeros((size, size))
    for x in range(-step, step+1):
        for y in range(-step, step+1):
            gaussian[x+step, y+step] = math.exp(-(x*x+y*y)/(2*s*s))
    gaussian = gaussian*(1/2*math.pi*s*s)
    su = np.sum(gaussian)
    gaussian = (gaussian/su).astype('float32')

    return gaussian


def med_blur(img, sizes):
    # run different sizes of median blur
    new_imgs = []
    for size in sizes:
        new_img = cv2.medianBlur(img, size)
        new_imgs.append(new_img)
    return new_imgs


def unsharp_masking(img, k):
    # unsharp masking
    img_n = cv2.GaussianBlur(img, (5, 5), 0)
    mask = img.astype('float32')-img_n
    img_g = img.astype('float32')+k*mask
    img_g = img_g-np.min(img_g)
    img_g = (img_g*255/np.max(img_g)).astype(int)
    mask -= np.min(mask)
    mask = (mask*255/np.max(mask)).astype(int)
    return img_g, mask


def sobel(img):
    '''
        apply sobel operator on image
    :param img:
    :return:
    '''
    sobel_x = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype='float32')
    sobel_y = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype='float32')
    m1 = np.abs(apply_filter(img, sobel_x))
    m2 = np.abs(apply_filter(img, sobel_y))
    m = m1+m2
    m = m-np.min(m)
    m = (m*255/np.max(m)).astype(int)
    return m


def laplace_(img):
    '''
        apply laplace operator on image
    :param img:
    :return:
    '''
    laplace = np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]], dtype='float32')
    d2f = (np.array(apply_filter(img, laplace)))
    img_g = img.astype('float32')+d2f
    img_g = img_g-np.min(img_g)
    img_g = (img_g*255/np.max(img_g)).astype(int)
    d2f = d2f-np.min(d2f)
    d2f = (d2f*255/np.max(d2f)).astype(int)
    return img_g, d2f


def NMS(M, dx, dy):
    # 非最大值抑制
    d = np.copy(M)
    w, h = M.shape
    NMS = np.copy(d)
    NMS[0, :] = NMS[w - 1, :] = NMS[:, 0] = NMS[:, h - 1] = 0

    for i in range(1, w - 1):
        for j in range(1, h - 1):
            # 梯度为0则非边缘点
            if M[i, j] == 0:
                NMS[i, j] = 0
            else:
                grad_x = dx[i, j]
                grad_y = dy[i, j]
                grad = d[i, j]
                # y
                if np.abs(grad_y) > np.abs(grad_x):
                    weight = np.abs(grad_x) / np.abs(grad_y)  # 权重
                    grad2 = d[i - 1, j]
                    grad4 = d[i + 1, j]
                    if grad_x * grad_y > 0:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]
                    else:
                        grad1 = d[i - 1, j + 1]
                        grad3 = d[i + 1, j - 1]
                # x
                else:
                    weight = np.abs(grad_y) / np.abs(grad_x)
                    grad2 = d[i, j - 1]
                    grad4 = d[i, j + 1]
                    if grad_x * grad_y > 0:
                        grad1 = d[i + 1, j - 1]
                        grad3 = d[i - 1, j + 1]
                    else:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]
                # 亚像素插值
                grad1 = weight * grad1 + (1 - weight) * grad2
                grad2 = weight * grad3 + (1 - weight) * grad4
                if grad >= grad1 and grad >= grad2:
                    # 边缘
                    NMS[i, j] = grad
                else:
                    NMS[i, j] = 0
    return NMS


def canny(img, th1, th2):
    # canny算法
    # 阈值参数th1, th2 为与最大值的比
    sobel_x = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype='float32')
    sobel_y = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype='float32')
    w, h, _ = img.shape
    img = cv2.GaussianBlur(img, (5, 5), 0)
    dx = apply_filter(img, sobel_x)
    dy = apply_filter(img, sobel_y)
    d_img = np.sqrt(np.power(dx, 2)+np.power(dy, 2))
    NMS_img = NMS(d_img[:, :, 0], dx[:, :, 0], dy[:, :, 0])        # one channel
    th1 = th1*np.max(NMS_img)
    th2 = th2*np.max(NMS_img)
    print("threshold:{}<x<{}".format(th1, th2))
    new_img = np.zeros((w, h))
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            if NMS_img[i, j] < th1:
                new_img[i, j] = 0
            elif NMS_img[i, j] > th2:
                new_img[i, j] = 1
            elif (NMS_img[i - 1, j - 1:j + 1] < th2).any() or (NMS_img[i + 1, j - 1:j + 1].any() or (NMS_img[i, [j - 1, j + 1]] < th2).any()):
                new_img[i, j] = 1

    return np.stack([new_img]*3, axis=2)


def canny_cv2(img, th1, th2):
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    img_c = cv2.Canny(img[:, :, 0], th1, th2)
    img_c = np.stack([img_c]*3, axis=2)
    plt.subplot(121)
    plt.title('original img')
    plt.imshow(img)
    plt.subplot(122)
    plt.title('processed img')
    plt.imshow(img_c)
    plt.show()
    return img_c

