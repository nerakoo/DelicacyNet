# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def imshow(image):
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

def rgb2hsi(image):
    b, g, r = cv.split(image)
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    eps = 1e-6

    img_i = (r + g + b) / 3

    img_h = np.zeros(r.shape, dtype=np.float32)
    img_s = np.zeros(r.shape, dtype=np.float32)
    min_rgb = np.zeros(r.shape, dtype=np.float32)

    min_rgb = np.where((r <= g) & (r <= b), r, min_rgb)
    min_rgb = np.where((g <= r) & (g <= b), g, min_rgb)
    min_rgb = np.where((b <= g) & (b <= r), b, min_rgb)
    img_s = 1 - 3*min_rgb/(r+g+b+eps)

    num = ((r-g) + (r-b))/2
    den = np.sqrt((r-g)**2 + (r-b)*(g-b))
    theta = np.arccos(num/(den+eps))
    img_h = np.where((b-g) > 0, 2*np.pi - theta, theta)
    img_h = np.where(img_s == 0, 0, img_h)

    img_h = img_h/(2*np.pi)
    temp_s = img_s - np.min(img_s)
    temp_i = img_i - np.min(img_i)
    img_s = temp_s/np.max(temp_s)
    img_i = temp_i/np.max(temp_i)

    image_hsi = cv.merge([img_h, img_s, img_i])
    return img_h, img_s, img_i, image_hsi


# HSI到RGB的变换
def hsi2rgb(image_hsi):
    eps = 1e-6
    img_h, img_s, img_i = cv.split(image_hsi)

    image_out = np.zeros((img_h.shape[0], img_h.shape[1], 3))
    img_h = img_h*2*np.pi
    print(img_h)

    img_r = np.zeros(img_h.shape, dtype=np.float32)
    img_g = np.zeros(img_h.shape, dtype=np.float32)
    img_b = np.zeros(img_h.shape, dtype=np.float32)

    img_b = np.where((img_h >= 0) & (img_h < 2 * np.pi / 3), img_i * (1 - img_s), img_b)
    img_r = np.where((img_h >= 0) & (img_h < 2 * np.pi / 3),
                     img_i * (1 + img_s * np.cos(img_h) / (np.cos(np.pi/3 - img_h))), img_r)
    img_g = np.where((img_h >= 0) & (img_h < 2 * np.pi / 3), 3 * img_i - (img_r + img_b), img_g)

    img_r = np.where((img_h >= 2*np.pi/3) & (img_h < 4*np.pi/3), img_i * (1 - img_s), img_r)
    img_g = np.where((img_h >= 2*np.pi/3) & (img_h < 4*np.pi/3),
                     img_i * (1 + img_s * np.cos(img_h-2*np.pi/3) / (np.cos(np.pi - img_h))), img_g)
    img_b = np.where((img_h >= 2*np.pi/3) & (img_h < 4*np.pi/3), 3 * img_i - (img_r + img_g), img_b)

    img_g = np.where((img_h >= 4*np.pi/3) & (img_h <= 2*np.pi), img_i * (1 - img_s), img_g)
    img_b = np.where((img_h >= 4*np.pi/3) & (img_h <= 2*np.pi),
                     img_i * (1 + img_s * np.cos(img_h-4*np.pi/3) / (np.cos(5*np.pi/3 - img_h))), img_b)
    img_r = np.where((img_h >= 4*np.pi/3) & (img_h <= 2*np.pi), 3 * img_i - (img_b + img_g), img_r)

    temp_r = img_r - np.min(img_r)
    img_r = temp_r/np.max(temp_r)

    temp_g = img_g - np.min(img_g)
    img_g = temp_g/np.max(temp_g)

    temp_b = img_b - np.min(img_b)
    img_b = temp_b/np.max(temp_b)

    image_out = cv.merge((img_r, img_g, img_b))
    # print(image_out.shape)
    return image_out

