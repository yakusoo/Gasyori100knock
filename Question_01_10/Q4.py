import cv2
import numpy as np

def BGR2GRAY(img):
    gray = np.zeros_like(img)
    red = img[:,:,2].copy()
    green = img[:,:,1].copy()
    blue = img[:,:,0].copy()
    gray = 0.2126*red + 0.7152*green + 0.0722*blue

    return gray

def otsu_binarization(img):
    max_sigma = 0
    max_t = 0
    H, W = img.shape

    for _th in range(1, 256):
        v0 = img[np.where(img < _th)]
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        w0 = len(v0)
        v1 = img[np.where(img >= _th)]
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        w1 = len(v1)
        sigma = w0 * w1 * ((m0 - m1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = _th

    print("threshold >>", max_t)
    th = max_t
    img[img < th] = 0
    img[img >= th] =255

    return img

img = cv2.imread("imori.jpg").astype(np.float)
gray = BGR2GRAY(img).astype(np.uint8)
out = otsu_binarization(gray)

cv2.imwrite('answer_4.jpg',out)
cv2.imshow('answer_4.jpg',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
