import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
'''
    Image recovery.
    # add gaussian/salt_pepper noise to image
    # apply motion degrade to image
    # geometric mean filtering, arithmetic mean filtering, (inverse) harmonic mean filtering.
    # maximum/minimum/median value filtering, median point filtering
    # adaptive local noise reduction filtering, adaptive median filtering
    # wiener filtering, constrained least squares filtering
'''


def dft(img, pad=True):
    # 图像填充 DFT计算
    img = img[:, :, 0]
    w, h = img.shape
    # 填充
    if pad is True:
        p_img = np.zeros((2*w, 2*h))
        p_img[:w, :h] = img
    else:
        p_img = img

    dft_img = cv2.dft(np.float32(p_img), flags=cv2.DFT_COMPLEX_OUTPUT)

    return np.fft.fftshift(dft_img)


def idft(dft, normalize=True):
    # IDFT计算
    f = np.fft.fftshift(dft)
    # f = cv2.idft(dft)
    f = cv2.idft(f)
    f = np.real(f)
    # f = cv2.magnitude(f[:, :, 0], f[:, :, 1])
    if normalize:
        # f -= np.min(f)
        # f = f*255/np.max(f)
        cv2.normalize(f, f, 255, 0, cv2.NORM_MINMAX)
    return np.round(f).astype('int')[:, :, 0]


def calc_uv(w, h, a, b):
    v, u = np.meshgrid(np.arange(w), np.arange(h))
    u = u.astype('float32')
    v = v.astype('float32')
    u = u-(w-1)/2
    v = v-(h-1)/2
    # 计算uv坐标网格 向上取整 绝对值>=1
    u = np.ceil(np.abs(u))*u/np.abs(u)
    v = np.ceil(np.abs(v))*v/np.abs(v)
    uv = a * u + b * v
    return uv, abs(math.ceil(u[0, 0]*a)), abs(math.ceil(v[0, 0]*b))


def gaus_noi(image, sigma, mean=0):
    # 图片叠加高斯噪声
    image = image.astype('float32')
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, image.shape[:2])
    # gauss = gauss.reshape(row, col, ch)
    n_img = image[:, :, 0] + gauss
    # 范围限制
    for idx in range(row):
        for idy in range(col):
            if n_img[idx, idy] < 0:
                n_img[idx, idy] = 0
            elif n_img[idx, idy] > 255:
                n_img[idx, idy] = 255
    n_img = np.stack([n_img]*3, axis=2)
    return n_img.astype('int')


def sp_noi(image, sp_ratio=0.5, ratio=0.2):
    '''
    图片叠加椒盐噪声
    :param sp_ratio: ratio of salt(ratio) and pepper(1-sp_ratio)
    :param ratio: noise amount
    '''
    row, col, ch = image.shape
    out = np.copy(image)
    num_salt = np.ceil(ratio*row*col*sp_ratio)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    out[coords[0], coords[1], :] = 255
    num_pepper = np.ceil(ratio*row*col*(1-sp_ratio))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    out[coords[0], coords[1], :] = 0
    return out


def apply_filter(img, filter):
    # apply filter and show new img
    img = img.astype('float32')
    f_img = cv2.filter2D(img, -1, filter)
    return f_img


def geo_mean(img, size=(3, 3)):
    # 几何均值滤波
    img = img.astype('float64')
    h, w, _ = img.shape
    gh, gw = size
    mn = gh*gw
    n_img = np.copy(img)
    for ix in range(1, h-size[0]//2):
        for iy in range(1, w-size[1]//2):
            val = np.prod(img[ix-gh//2:ix+gh//2+1, iy-gw//2:iy+gw//2+1, 0])
            n_img[ix, iy] = pow(val, 1/mn)
    return n_img.astype(int)


def inv_har_mean(img, q=0, size=(3, 3)):
    # 逆谐波均值滤波
    img = img.astype('float64')
    h, w, _ = img.shape
    gh, gw = size
    n_img = np.zeros((h, w))
    for ix in range(1, h - size[0] // 2):
        for iy in range(1, w - size[1] // 2):
            val_m = np.sum(np.power(img[ix - gh // 2:ix + gh // 2 + 1, iy - gw // 2:iy + gw // 2 + 1, 0], q+1))
            val_d = np.sum(np.power(img[ix - gh // 2:ix + gh // 2 + 1, iy - gw // 2:iy + gw // 2 + 1, 0], q))+0.0001
            n_img[ix, iy] = val_m/val_d
    return n_img.astype(int)


def min_med_max(img, type, size=(3, 3)):
    '''
        最大最小值、中值、中点滤波
    :param type: 'min', 'max', 'med' or 'med_p'
    '''
    w, h, _ = img.shape
    gh, gw = size
    n_img = np.zeros((w, h))
    if type == 'min':
        for ix in range(1, w - size[0] // 2):
            for iy in range(1, h - size[1] // 2):
                n_img[ix, iy] = np.min(img[ix - gh // 2:ix + gh // 2 + 1, iy - gw // 2:iy + gw // 2 + 1, 0])
    elif type == 'max':
        for ix in range(1, h - size[0] // 2):
            for iy in range(1, w - size[1] // 2):
                n_img[ix, iy] = np.max(img[ix - gh // 2:ix + gh // 2 + 1, iy - gw // 2:iy + gw // 2 + 1, 0])
    elif type == 'med':
        n_img = cv2.medianBlur(img.astype('float32'), size[0])
        for runtimes in range(2):
            n_img = cv2.medianBlur(n_img, size[0])
        n_img = n_img[:, :, 0]
    elif type == 'med_p':
        for ix in range(1, h - size[0] // 2):
            for iy in range(1, w - size[1] // 2):
                n_img[ix, iy] = 0.5*(np.max(img[ix - gh // 2:ix + gh // 2 + 1, iy - gw // 2:iy + gw // 2 + 1, 0])+
                                     np.min(img[ix - gh // 2:ix + gh // 2 + 1, iy - gw // 2:iy + gw // 2 + 1, 0]))
    return np.stack([n_img]*3, axis=2).astype('uint8')


def ada_noi_red(img, dn, size=(3, 3)):
    '''
        自适应局部降噪滤波
    :param dn: 噪声方差
    '''
    img = img.astype('float64')
    h, w, _ = img.shape
    gh, gw = size
    n_img = np.copy(img[:, :, 0])
    for ix in range(1, h - size[0] // 2):
        for iy in range(1, w - size[1] // 2):
            Sxy = img[ix - gh // 2:ix + gh // 2 + 1, iy - gw // 2:iy + gw // 2 + 1, 0]
            ml = np.sum(Sxy)/pow(gh, 2)
            dl = np.sum(np.power(Sxy-ml, 2))/pow(gh, 2)+0.0001
            if dn>dl:
                 dl=dn
            n_img[ix, iy] = img[ix, iy, 0]-(dn/dl)*(img[ix, iy, 0]-ml)
    for idx in range(h):
        for idy in range(w):
            if n_img[idx, idy] < 0:
                n_img[idx, idy] = 0
            elif n_img[idx, idy] > 255:
                n_img[idx, idy] = 255
    return n_img.astype(int)


def ada_med(img, max_size=15):
    '''
        自适应中值滤波
    :param img:
    :param max_size: max calculating box size
    '''
    img = img.astype('float64')
    h, w, _ = img.shape
    n_img = np.copy(img[:, :, 0])
    for ix in range(3, h - 3):
        for iy in range(3, w - 3):
            for s in range(5, max_size+1, 2):
                Sxy = img[ix - s // 2:ix + s // 2 + 1, iy - s // 2:iy + s // 2 + 1, 0]
                zmin = np.min(Sxy)
                zmax = np.max(Sxy)
                zmed = np.median(Sxy)
                if zmin<zmed<zmax:
                    if zmin<img[ix, iy, 0]<zmax:
                        n_img[ix, iy] = img[ix, iy, 0]
                        break
                    else:
                        n_img[ix, iy] = zmed
                        break
            if s>=max_size:
                n_img[ix, iy] = zmed
    return n_img


def img_deg(img, a=0.1, b=0.1, T=1):
    '''
        motion blur.
    '''
    img = img.astype('float64')
    w, h, _ = img.shape
    # Huv为复数 频域发生频移 空域相移 变换图像不填充
    p, q = w, h
    img_dft = dft(img, pad=False)
    # 计算退化传递函数Huv
    Huv = np.zeros((p, q, 2))
    uv, u_m, v_m = calc_uv(p, q, a, b)        # 原点在中心
    Apt = T * np.sin(math.pi * uv) / ((math.pi * uv)+0.000001)
    Huv[:, :, 0] = Apt*np.cos(math.pi*uv)
    Huv[:, :, 1] = -Apt*np.sin(math.pi*uv)
    plt.subplot(121)
    plt.imshow(Huv[:, :, 0], 'gray')
    plt.subplot(122)
    plt.imshow(Huv[:, :, 1], 'gray')
    plt.show()
    r_img_dft = np.zeros((p, q, 2))
    r_img_dft[:, :, 0] = img_dft[:, :, 0] * Huv[:, :, 0] - img_dft[:, :, 1] * Huv[:, :, 1]
    r_img_dft[:, :, 1] = img_dft[:, :, 1] * Huv[:, :, 0] + img_dft[:, :, 0] * Huv[:, :, 1]
    r_img = idft(r_img_dft)[:w, :h]
    plt.imshow(r_img, 'gray')
    plt.show()
    return np.stack([r_img]*3, axis=-1), Huv


def wiener(img, Huv, K):
    '''
        维纳滤波
    :param Huv: 退化函数估计
    :param K: 1/SNR估计
    '''
    w, h, _ = img.shape
    Huv = Huv[:, :, 0]+1j*Huv[:, :, 1]
    img_dft = dft(img, pad=False)
    img_dft = img_dft[:, :, 0]+1j*img_dft[:, :, 1]
    Huv2 = Huv*np.conj(Huv)
    Fuv = (img_dft/(Huv+0.000001))*(Huv2/(Huv2+K))
    r_img_dft = np.stack([np.real(Fuv), np.imag(Fuv)], axis=2)
    r_img = idft(r_img_dft)[:w, :h]
    plt.imshow(r_img, 'gray')
    plt.show()
    return np.stack([r_img]*3, axis=2).astype('uint8')


def con_le_squares(img, Huv, gama, dg=5*pow(10, -6), a=pow(10, 2), delta2=10):
    '''
        约束最小二乘方滤波
    :param Huv: 退化函数估计
    :param gama: 恢复参数
    :param dg: γ调节单位
    :param a: 精确度因子
    :param delta2: 噪声功率谱
    '''
    w, h, _ = img.shape
    padding = 0
    p, q = w, h
    # p, q = 2*w, 2*h
    nanda2 = w*h*(delta2+0)
    pxy = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])
    Puv = spa_to_fre(img, pxy, p, q)
    Puv = Puv[:, :, 0]+1j*Puv[:, :, 1]
    Huv = Huv[:, :, 0]+1j*Huv[:, :, 1]
    img_dft = dft(img, pad=False)
    # img_dft = dft(img)
    img_dft = img_dft[:, :, 0]+1j*img_dft[:, :, 1]
    for runtimes in range(128):
        # 动态调整gama值
        Fuv = np.conj(Huv)*img_dft/(np.conj(Huv)*Huv+gama*np.conj(Puv)*Puv)
        # calculate r=g-Hf'
        Ruv = img_dft-Huv*Fuv
        rxy = idft(np.stack([np.real(Ruv), np.imag(Ruv)], axis=2), normalize=False)
        r2 = np.sum(np.power(rxy, 2))
        if r2 < nanda2-a:
            gama += dg
        elif r2 > nanda2+a:
            gama -= dg
        else:
            break
    print('gama={}/nruntime={}'.format(gama, runtimes))
    r_img_dft = np.stack([np.real(Fuv), np.imag(Fuv)], axis=2)
    r_img = idft(r_img_dft)[:w, :h]
    plt.imshow(r_img, 'gray')
    plt.show()
    return np.stack([r_img] * 3, axis=2).astype('uint8')


if __name__ == '__main__':
    img = cv2.imread('')
    delta = 10
    # image motion blur
    new_img, Huv = img_deg(img, a=0.01, b=0.01)
    # add gaussian noise to img
    noi_img = gaus_noi(new_img, delta)
    # wiener filtering
    new_img = wiener(noi_img, Huv, 5*pow(10, -3))
    # Constrained least squares filtering
    new_img0 = con_le_squares(noi_img, Huv, 5*pow(10, -3), dg=2*pow(10, -6), delta2=delta)
    plt.subplot(131)
    plt.title('img with noise')
    plt.imshow(noi_img, 'gray')
    plt.subplot(132)
    plt.title('wiener')
    plt.imshow(new_img, 'gray')
    plt.subplot(133)
    plt.title('least squares')
    plt.imshow(new_img0, 'gray')
    plt.show()

