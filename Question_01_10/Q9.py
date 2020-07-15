import cv2
import numpy as np

def gaussian_filter(img, K_size, sigma):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    #zeropadding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad:pad+H,pad:pad+W] = img.copy().astype(np.float)

    #Kernel
    K = np.zeros((K_size,K_size),dtype=np.float)
    for x in range(-pad, -pad+K_size):
        for y in range(-pad, -pad+K_size):
            K[y+pad,x+pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()

    tmp = out.copy()

    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad+y,pad+x,c] = np.sum(K * tmp[y:y+K_size,x:x+K_size,c])

    out = np.clip(out, 0, 255)
    out = out[pad:pad+H,pad:pad+W].astype(np.uint8)

    return out


img = cv2.imread("imori_noise.jpg")

out = gaussian_filter(img, 3, 1.3)

cv2.imwrite('answer_9.jpg',out)
cv2.imshow('answer_9.jpg',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
