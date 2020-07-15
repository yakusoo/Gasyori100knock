import cv2
import numpy as np

def BGR2GRAY(img):
    gray = np.zeros_like(img)
    red = img[:,:,2].copy()
    green = img[:,:,1].copy()
    blue = img[:,:,0].copy()
    gray = 0.2126*red + 0.7152*green + 0.0722*blue
    out = gray.copy().astype(np.uint8)

    return out

def bi_linear(img, alpha):
    if len(img.shape) > 2:
        H, W, C = img.shape
    else:
        H, W = img.shape
        C = 1

    Ha


img = cv2.imread("imori.jpg").astype(np.float32)



out = out.astype(np.uint8)


cv2.imwrite('answer_72.jpg',out)
cv2.imshow('answer_72.jpg',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
