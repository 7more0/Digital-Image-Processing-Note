import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math


'''
    Filtering in frequency domain.
    # apply low-pass/high-pass gaussian/butterworth filter in frequency domain
    # calculate power spectrum
    # apply laplace arithmetic/un-sharp masking in frequency domain
    # convert spacial mask to frequency mask
    # convert frequency mask to spacial mask
'''


def dft(img):
    # 图像填充 DFT计算
    img = img[:, :, 0]
    w, h = img.shape
    # 填充
    p_img = np.zeros((2*w, 2*h))
    p_img[:w, :h] = img

    dft_img = cv2.dft(np.float32(p_img), flags=cv2.DFT_COMPLEX_OUTPUT)

    return np.fft.fftshift(dft_img)


def show_dft(dft_img):
    # dft_img = cv2.log(1 + cv2.magnitude(dft_img[:, :, 0], dft_img[:, :, 1]))
    dft_img = cv2.magnitude(dft_img[:, :, 0], dft_img[:, :, 1])
    cv2.normalize(dft_img, dft_img, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(np.around(dft_img))


def calc_distances(w, h):
    # 计算到模板中心的距离
    x, y = np.meshgrid(np.arange(h), np.arange(w))
    x = x.astype('float32')
    y = y.astype('float32')
    x = np.power(x-(w-1)/2, 2)
    y = np.power(y-(h-1)/2, 2)
    distance = np.sqrt(x+y)
    return distance


def calc_P(dft, distance, r):
    # 功率谱计算
    center_x, center_y = distance.shape
    center_x = int(center_x/2)
    center_y = int(center_y/2)
    dft = np.power(dft[:, :, 0], 2)+np.power(dft[:, :, 1], 2)
    Pa = 0
    P = 0
    for k in range(dft.shape[0]):
        Pa += dft[:, k]
    Pa = np.sum(Pa)
    for u in range(-r, r+1):
        for v in range(-r, r+1):
            if distance[center_x+u, center_y+v] < r:
                P += dft[center_x+u, center_y+v]
    return P/Pa


def lp_mask_gen(d0, distance, type, n=1):
    # 生成低通滤波器
    if type == 'butterworth':
        H = 1+np.power((distance/d0), 2*n)
        H = 1/H
    elif type == 'gaussian':
        H = np.exp(-np.power(distance, 2)/(2*d0*d0))
    return H


def hp_mask_gen(d0, distance, type, n=1):
    # 生成高通滤波器
    if type == 'butterworth':
        # avoid divide 0 mistake
        distance += 0.001
        H = 1+np.power((1/distance)*d0, 2*n)
        H = 1/H
    elif type == 'gaussian':
        H = np.exp(-np.power(distance, 2)/(2*d0*d0))
        H = 1-H
    return H


def idft(dft, normalize=True):
    # IDFT计算
    f = np.fft.fftshift(dft)
    f = cv2.idft(f)
    f = np.real(f)
    if normalize:
        cv2.normalize(f, f, 0, 255, cv2.NORM_MINMAX)
    return np.round(f).astype('float64')[:, :, 0]


def apply_filter(img, type, d0, n=2):
    '''
        apply filter on img
    :param type: filter type, 'h' or 'l'
    :param d0: power radius
    :param n: for butterworth filters
    '''
    w, h, _ = img.shape
    p = 2*w
    q = 2*h
    img_dft = dft(img)
    distance = calc_distances(p, q)
    if type[0] == 'l':
        mask = lp_mask_gen(d0, distance, type[1], n=2)
        show_mask = mask*255
    elif type[0] =='h':
        mask = hp_mask_gen(d0, distance, type[1], n=2)
        show_mask = mask * 255
    new_img_dft = np.zeros((p, q, 2))
    new_img_dft[:, :, 0] = img_dft[:, :, 0] * mask
    new_img_dft[:, :, 1] = img_dft[:, :, 1] * mask
    new_img = idft(new_img_dft)[:w, :h]
    Pt = calc_P(img_dft, distance, d0)
    plt.subplot(221)
    plt.title('original img')
    plt.imshow(img, 'gray')
    plt.subplot(222)
    plt.title('mask')
    plt.imshow(show_mask, 'gray')
    plt.subplot(223)
    plt.title('dft of new img')
    plt.imshow(show_dft(new_img_dft), 'gray')
    plt.subplot(224)
    plt.title('new img(alpha={}%)'.format(Pt*100))
    plt.imshow(new_img, 'gray')
    plt.show()
    return new_img, Pt


def laplce(img):
    '''
        apply laplace arithmetic/un-sharp masking in frequency domain
    '''
    w, h, _ = img.shape
    p = 2*w
    q = 2*h
    nor_img = img.astype('float32')/255
    img_dft = dft(nor_img)
    distance = calc_distances(p, q)
    H = -4*pow(math.pi, 2)*np.power(distance, 2)
    new_img_dft = np.zeros((p, q, 2))
    new_img_dft[:, :, 0] = img_dft[:, :, 0] * H
    new_img_dft[:, :, 1] = img_dft[:, :, 1] * H
    laplace_img = idft(new_img_dft, normalize=False)[:w, :h]
    laplace_img /= (np.max(laplace_img)-np.min(laplace_img))
    g_img = img[:, :, 0] - laplace_img*255
    for idx in range(w):
            for idy in range(h):
                if g_img[idx, idy] < 0:
                    g_img[idx, idy] = 0
                elif g_img[idx, idy] > 255:
                    g_img[idx, idy] = 255
    g_img -= np.min(g_img)
    g_img = (g_img*255/np.max(g_img)).astype(int)
    laplace_img = -laplace_img
    laplace_img -= np.min(laplace_img)
    laplace_img = (laplace_img*255/np.max(laplace_img)).astype('int')
    plt.subplot(131)
    plt.title('original img')
    plt.imshow(img[:, :, 0], 'gray')
    plt.subplot(132)
    plt.title('laplace img')
    plt.imshow(np.stack([laplace_img]*3, axis=2))
    plt.subplot(133)
    plt.title('sharped img')
    plt.imshow(g_img, 'gray')
    plt.show()


def unsharp(img):
    '''
        apply laplace arithmetic/un-sharp masking in frequency domain
    '''
    img = img.astype('float32')
    w, h, _ = img.shape
    p = 2*w
    q = 2*h
    distance = calc_distances(p, q)
    gaussian = lp_mask_gen(100, distance, 'gaussian')
    img_dft = dft(img)
    lp_img_dft = np.zeros((p, q, 2))
    lp_img_dft[:, :, 0] = img_dft[:, :, 0]*gaussian
    lp_img_dft[:, :, 1] = img_dft[:, :, 1]*gaussian
    lp_img = idft(lp_img_dft, normalize=False)[:w, :h]
    lp_img /= np.max(lp_img)
    mask = img[:, :, 0]/255-lp_img
    cv2.normalize(mask, mask, 255, 0, cv2.NORM_MINMAX)
    new_img = img.astype('float32')[:, :, 0]+mask
    for idx in range(w):
            for idy in range(h):
                if new_img[idx, idy] < 0:
                    new_img[idx, idy] = 0
                elif new_img[idx, idy] > 255:
                    new_img[idx, idy] = 255
    new_img -= np.min(new_img)
    new_img = (new_img*255/np.max(new_img)).astype('float32')
    plt.subplot(131)
    plt.title('original img')
    plt.imshow(img.astype(int))
    plt.subplot(132)
    plt.title('mask')
    plt.imshow(mask, 'gray')
    plt.subplot(133)
    plt.title('sharped img')
    plt.imshow(new_img, 'gray')
    plt.show()


def re_center(w, h):
    # reset spectrum center
    x, y = np.meshgrid(np.arange(h), np.arange(w))
    x = x.astype('float32')
    y = y.astype('float32')
    res = x+y
    for i in range(w):
        for j in range(h):
            res[i, j] = pow(-1, res[i, j])
    return res


def spa_to_fre(img, hxy, p, q):
    # 空域模板求频域响应
    w, h, _ = img.shape
    center = re_center(p, q)
    wh, hh = hxy.shape
    # 空域模板填充
    n_hxy = np.zeros((p, q))
    n_hxy[w//2 - wh // 2:w//2 + wh // 2 + 1, h//2 - hh // 2:h//2 + hh // 2 + 1] = hxy
    n_hxy = n_hxy * center
    Huv = cv2.dft(np.float32(n_hxy), flags=cv2.DFT_COMPLEX_OUTPUT)
    # H(u, v)*(-1)^(u,v)
    Huv[:, :, 0] = Huv[:, :, 0] * center
    Huv[:, :, 1] = Huv[:, :, 1] * center
    return Huv


def space_to_frequency(img, hxy):
    '''
        space --> frequency
    :param hxy: spacial mask
    :return:
    '''
    w, h, _ = img.shape
    p = 2 * w
    q = 2 * h
    # (-1)^(x+y)频谱中心化
    center = re_center(p, q)
    wh, hh = hxy.shape
    # 空域模板填充
    n_hxy = np.zeros((p, q))
    n_hxy[w - wh // 2:w + wh // 2 + 1, h - hh // 2:h + hh // 2 + 1] = hxy
    n_hxy = n_hxy * center
    Huv = cv2.dft(np.float32(n_hxy), flags=cv2.DFT_COMPLEX_OUTPUT)
    img_dft = dft(img)
    fre_img_dft = np.zeros((p, q, 2))
    # H(u, v)*(-1)^(u,v)
    Huv[:, :, 0] = Huv[:, :, 0] * center
    Huv[:, :, 1] = Huv[:, :, 1] * center
    # 复数乘法
    fre_img_dft[:, :, 0] = img_dft[:, :, 0] * Huv[:, :, 0] - img_dft[:, :, 1] * Huv[:, :, 1]
    fre_img_dft[:, :, 1] = img_dft[:, :, 1] * Huv[:, :, 0] + img_dft[:, :, 0] * Huv[:, :, 1]
    fre_img = (idft(fre_img_dft)[:w, :h])       # 频域结果
    spa_img = cv2.filter2D(img.astype('float64'), -1, hxy)      # 空域结果
    cv2.normalize(spa_img, spa_img, 0, 255, cv2.NORM_MINMAX)
    spa_img = spa_img.astype(int)
    plt.subplot(131)
    plt.title('original img')
    plt.imshow(img, 'gray')
    # plt.subplot(222)
    # plt.title('H(u,v)')
    # plt.imshow(Huv[:, :, 1], 'gray')
    plt.subplot(132)
    plt.title('frequency')
    plt.imshow(fre_img, 'gray')
    plt.subplot(133)
    plt.title('space')
    plt.imshow(spa_img, 'gray')
    plt.show()


def frequency_to_space(img, Huv, wh, hh):
    '''
        frequency --> space
    :param Huv: frequency mask
    :param wh: space mask width (The size of the spacial mask should be able to contain the main energy of the spacial response)
    :param hh: space mask height
    '''
    w, h, _ = img.shape
    p = 2 * w
    q = 2 * h
    pad_img = np.zeros((w+wh-1, h+hh-1))
    pad_img[wh//2:w+wh//2, hh//2:h+hh//2] = img[:, :, 0].astype('float32')      # pad edges of image
    H = np.zeros((p, q, 2))
    H[:, :, 0] = Huv
    H[:, :, 1] = 0      # generate frequency mask
    fre_img_dft = np.zeros((p, q, 2))
    img_dft = dft(img)
    fre_img_dft[:, :, 0] = img_dft[:, :, 0]*H[:, :, 0]
    fre_img_dft[:, :, 1] = img_dft[:, :, 1]*H[:, :, 0]
    fre_img = idft(fre_img_dft)[:w, :h]
    Hxy = np.fft.fftshift(cv2.idft(np.fft.fftshift(H)))[w-wh//2:w+wh//2+1, h-hh//2:h+hh//2+1, 0]
    Hxy = Hxy/(np.max(Hxy)-np.min(Hxy))     # generate spacial mask
    spa_img = cv2.filter2D(pad_img, -1, Hxy)[wh//2:w+wh//2, hh//2:h+hh//2]
    spa_img -= np.min(spa_img)
    spa_img = (spa_img*255/np.max(spa_img)).astype(int)
    plt.subplot(131)
    plt.title('original img')
    plt.imshow(img, 'gray')
    plt.subplot(132)
    plt.title('frequency')
    plt.imshow(fre_img, 'gray')
    plt.subplot(133)
    plt.title('space')
    plt.imshow(spa_img, 'gray')
    plt.show()


if __name__ == '__main__':
    img = cv2.imread('./imgs/test5.jpg')
    # hxy = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
    hxy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # spacial mod
    space_to_frequency(img, hxy)
    d0 = 100
    wh, hh = 15, 15
    h, w, _ = img.shape
    p, q = 2*h, 2*w
    distance = calc_distances(p, q)
    Huv = lp_mask_gen(d0, distance, 'gaussian', n=2)
    frequency_to_space(img, Huv, wh, hh)

