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

def Morphology_Erode(img, Ero_time):
    H, W = img.shape

    out = img.copy()

    MF = np.array(((0,1,0),(1,0,1),(0,1,0)), dtype=np.int)

    for i in range (Ero_time):
        tmp = np.pad(out,(1,1),'edge')
        for y in range (1, H+1):
            for x in range (1, W+1):
                if np.sum(MF * tmp[y-1:y+2,x-1:x+2]) < 255*4:
                    out[y-1,x-1] = 0

    return out

def Morphology_Dilate(img, Dil_time):
    H, W = img.shape

    out = img.copy()

    MF = np.array(((0,1,0),(1,0,1),(0,1,0)), dtype=np.int)

    for i in range (Dil_time):
        tmp = np.pad(out,(1,1),'edge')
        for y in range (1, H+1):
            for x in range (1, W+1):
                if np.sum(MF * tmp[y-1:y+2,x-1:x+2]) > 255:
                    out[y-1,x-1] = 255

    return out

img = cv2.imread("imori.jpg").astype(np.float)
gray = BGR2GRAY(img).astype(np.uint8)
otsu = otsu_binarization(gray)
out = Morphology_Erode(otsu,1)
out = Morphology_Dilate(out,1)

cv2.imwrite('answer_49.jpg',out)
cv2.imshow('answer_49.jpg',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
