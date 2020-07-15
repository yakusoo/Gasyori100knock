import cv2
import numpy as np

def BGR2GRAY(img):
    gray = np.zeros_like(img)
    red = img[:,:,2].copy()
    green = img[:,:,1].copy()
    blue = img[:,:,0].copy()
    gray = 0.2126*red + 0.7152*green + 0.0722*blue

    return gray

def binalization(img,th):
    img[img < th] = 0;
    img[img >= th] =255;

    return img

img = cv2.imread("imori.jpg").astype(np.float)
gray = BGR2GRAY(img).astype(np.uint8)
out = binarization(gray,128)

print(img.shape)
print(gray.shape)
cv2.imwrite('answer_3.jpg',out)
cv2.imshow('answer_3.jpg',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
