import cv2
import numpy as np
import math
'''
    Include image image gray level transformation
    image mean variance
    image interpolation (nearest bilinear bicubic)
    image horizontal wrap and rotation
'''

def show(window, img):
    cv2.imshow(window, img)
    cv2.waitKey(0)


def set_g_scale(img, n):
    # 图像灰度级数量修改
    if n == 256:
        return img
    img = img[:, :, 0]
    step = 256//n
    half_step = 0.5*step
    g_levels = [k*step for k in range(1, n+1)]      # 灰度区间
    g = [level-half_step for level in g_levels]     # 标准灰度值
    g[np.argmin(g)] = 0
    g[np.argmax(g)] = 255
    for l, level in enumerate(g_levels):
        for r, row in enumerate(img):
            for c, col in enumerate(row):
                if (level-step)<col and col<level:
                    img[r, c] = g[l]
    img = np.stack([img, img, img], axis=2)
    return img


def mean_img(img):
    # 计算图像灰度均值
    mean = 0
    img = img[:, :, 0]
    size = np.shape(img)
    for row in img:
        for col in row:
            mean = mean + col
    mean = mean/(size[0]*size[1])
    return mean


def s_img(img, mean):
    # 计算图像灰度值方差
    s = 0
    img = img[:, :, 0]
    size = np.shape(img)
    for row in img:
        for col in row:
            s = s + pow((col-mean), 2)
    s = s/(size[0]*size[1])
    return s


def nearest(img, size):
    # 最近邻内插
    o_row, o_col, channels = img.shape
    row, col = size
    z_img = np.zeros((row, col, channels), dtype=np.uint8)

    def to_int(x, max):
        # 整数转换 控制边界
        if (x-int(x)) <= 0.5:
            x = int(x)
        else:
            x = int(x)+1
        if x >= max:
            x -= 1
        return x

    for r in range(row):
        for c in range(col):
            x = r*o_row/row
            y = c*o_col/col
            x = to_int(x, o_row)
            y = to_int(y, o_col)
            z_img[r, c, :] = img[x, y, :]

    return z_img


def bilinear(img, size):
    # 双线性内插
    o_row, o_col, channels = img.shape
    row, col = size
    z_img = np.zeros((row, col, channels), dtype=np.uint8)

    def to_int(a, max):
        a = int(a)
        if a >= max-1:
            a -= 1
        return a

    for r in range(row):
        for c in range(col):
            f_x = r*o_row/row
            f_y = c*o_col/col
            x = to_int(f_x, o_row)
            y = to_int(f_y, o_col)
            u = f_x - x
            v = f_y - y
            z_img[r, c, :] = (1 - u) * (1 - v) * img[x, y, :] + (1 - u) * v * img[x, y+1, :] \
                             + u * (1 - v) * img[x+1, y, :] + u * v * img[x+1, y+1, :]
    return z_img


def bicubic(x, a=-0.5):
    # 钟函数
    x = abs(x)
    if x <= 1:
        w = (a+2)*pow(x, 3)-(a+3)*pow(x, 2)+1
    elif 1 < x < 2:
        w = a*pow(x, 3)-5*a*pow(x, 2)+8*a*x-4*a
    else:
        w = 0
    return w


def bicubic_interpolation(img, size):
    # 双三次内插
    o_row, o_col, channels = img.shape
    row, col = size
    z_img = np.zeros((row, col, channels), dtype=np.uint8)

    def to_int(x, max):
        x = int(x)
        if x >= max-3:
            x -= 2
        elif x<=1:
            x += 1
        return x

    for r in range(row):
        for c in range(col):
            f_x = r * o_row / row
            f_y = c * o_col / col
            x = to_int(f_x, o_row)
            y = to_int(f_y, o_col)
            bic_val = 0
            for i in range(x - 1, x + 3):
                for j in range(y - 1, y + 3):
                    w = bicubic(f_x - i) * bicubic(f_y - j)
                    bic_val += (img[i, j, :]*w).astype(np.float)
            bic_val = (bic_val/255).astype(np.uint8)
            z_img[r, c, :] = bic_val
    return z_img


def horizontal_shear(img, s):
    # 水平错切
    o_row, o_col, channels = img.shape
    z_img = np.zeros((o_row, int(o_col+s*o_row)+1, channels), dtype=np.uint8)
    for r in range(o_row):
        for c in range(o_col):
            x = r
            y = int(s*r+c)
            z_img[x, y, :] = img[r, c, :]
    return z_img


def rotate(img, degree):
    # 旋转
    o_row, o_col, channels = img.shape
    h_r = 0.5*o_row
    h_c = 0.5*o_col
    z_img = np.zeros((o_row, o_col, channels), dtype=np.uint8)
    for r in range(o_row):
        for c in range(o_col):
            x = int((r-h_r)*math.cos(degree)-(c-h_c)*math.sin(degree))+1
            y = int((r-h_r)*math.cos(degree)+(c-h_c)*math.sin(degree))+1
            x += int(0.5*o_row)
            y += int(0.5*o_col)
            if x<0 or y<0:
                continue
            if x >= o_row or y >= o_col:
                continue
            z_img[x, y, :] = img[r, c, :]
    return z_img

