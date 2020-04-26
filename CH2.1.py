import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
    Image alignment based on ORB feature point detection.
'''


def img_alig(o_img1, o_img2, match_p_num):
    '''

    :param img1:
    :param img2: img to align
    :param match_p_num: number of feature points used in image alignment
    :return:
    '''

    img1 = cv2.cvtColor(o_img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(o_img2, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(100)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    # result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:99], img2, (0, 255, 0), flags=2)
    # plt.imshow(result)
    # plt.show()


    points1 = []
    points2 = []
    # matches = matches.random
    sel_matches = []

    for k in range(1, match_p_num):
       sel_matches.append(matches[2*k])

    # result = cv2.drawMatches(img1, kp1, img2, kp2, sel_matches, img2, (0, 255, 0), flags=2)
    # cv2.imwrite('./task2/match.jpg', result)
    # plt.imshow(result)
    # plt.show()

    for pair in sel_matches:
        points1.append(list(kp1[pair.trainIdx].pt))
        points2.append(list(kp2[pair.queryIdx].pt))
    # points1 = np.array(points1).astype(int)
    # points2 = np.array(points2).astype(int)
    # print(points1)
    # print(points2)

    for l in range(len(points1)):
        points1[l].append(1)
        points2[l].append(1)
    points1 = np.array(points1).astype(int)
    points2 = np.array(points2).astype(int)
    # P = np.matrix(points2.transpose())
    # Q = np.matrix(points1.transpose())
    # H = Q*(P.transpose())*(np.linalg.inv(P*P.transpose()))

    # print(P)
    # print(Q)
    
    H, s = cv2.findHomography(points2, points1, cv2.RANSAC)
    print(H)
    w_img = cv2.warpPerspective(o_img2, H, (img1.shape[1], img1.shape[0]))
    # w_img = cv2.cvtColor(w_img, cv2.COLOR_RGB2BGR)
    plt.subplot(121)
    plt.imshow(o_img1)
    plt.subplot(122)
    plt.imshow(w_img)
    plt.show()
    # cv2.imwrite('./w_img2.jpg', w_img)

