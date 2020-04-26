import cv2
import numpy as np
'''
    Include:
    histogram equalization
    histogram specification
    local histogram equalization
    image segmentation based on gray-scale threshold 
'''


def equalize(hist, rang, T):
    # 分布函数计算
    grays = np.zeros(hist.shape[0])
    for ix in range(len(grays)):
        for iy in range(ix+1):
            # 累加所有小于ix灰度
            grays[ix] += hist[iy]
    grays = grays*(rang-1)/T
    return np.rint(grays)


def equalize_hist(img):
    # 灰度图像直方图均衡
    image = (img.copy())[:, :, 0]
    row, col = image.shape
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    grays = equalize(hist, 256, row*col)
    for r in range(row):
        for c in range(col):
            image[r, c] = grays[image[r, c]]
    image = np.stack([image, image, image], axis=2)
    return image


def hist_mapping(m_img, img):
    # map s to z
    row, col, channels = m_img.shape
    m_hist = cv2.calcHist([m_img], [0], None, [256], [0, 256])
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    m_gray = equalize(m_hist, 256, row*col)       # G(z)
    gray = equalize(hist, 256, row*col)       # s
    mapping = {}        # map of s to z
    mark = 0        # largest matched z
    for id_s in range(len(gray)):
        try:
            mapping[gray[id_s]]
            continue
        except:
            id_m = 0
            min = 1
            for z in range(mark, len(m_gray)):
                # find min z with G(z) closest to current s
                if min == 0:
                    break
                if np.fabs(gray[id_s]-m_gray[z]) <= min:
                    id_m = z
                    min = np.fabs(gray[id_s]-m_gray[z])
                    mark = z
            mapping[gray[id_s]] = id_m
    for idx in range(len(gray)):
        gray[idx] = mapping[gray[idx]]

    return gray


def hist_spec(m_img, img1):
    # 直方图匹配
    gray = hist_mapping(m_img, img1)
    row, col, channels = img1.shape
    img_proc = img1.copy()
    for r in range(row):
        for c in range(col):
            img_proc[r, c, :] = np.array([gray[img1[r, c, 0]]]*3)
    return img_proc


def local_hist_equ(img):
    # (2*step+1)^2 local hist equalization, strike = 1
    step = 3
    row, col, channels = img.shape
    new_img = (img.copy())[:, :, 0]
    img = img[:, :, 0]
    for r in range(step, row-(step+1)):
        for c in range(step, col-(step+1)):
            result = cv2.equalizeHist(img[r-step:r+(step+1), c-step:c+(step+1)])
            result = result[step, step]
            new_img[r, c] = result
    new_img = np.stack([new_img]*3, axis=2)
    return new_img


def calc_t(img, t0, tt):
    # calculate threshold
    img = img[:, :, 0]
    row, col = img.shape
    dt = 9999
    while abs(dt) > tt:
        low = 0
        low_n = 0
        high_n = 0
        high = 0
        for r in range(row):
            for c in range(col):
                if img[r, c] <= t0:
                    low += img[r, c]
                    low_n += 1
                else:
                    high += img[r, c]
                    high_n += 1
        low /= low_n
        high /= high_n
        t = 0.5*(high+low)
        dt = t-t0
        t0 = t
    return round(t0)


def seg_img(img, rang):
    # 基于灰度阈值图像分割（2分割）
    low, high = rang
    row, col, channels = img.shape
    img1 = np.zeros((row, col, channels))
    img2 = np.zeros((row, col, channels))
    for r in range(row):
        for c in range(col):
            if low<=img[r, c, 0]<=high:
                img1[r, c, :] = img[r, c, :]
            else:
                img2[r, c, :] = img[r, c, :]
    return img1, img2

