import cv2
import numpy as np

def median_filter(img, K_size):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    #zeropadding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad:pad+H,pad:pad+W] = img.copy().astype(np.float)

    tmp = out.copy()

    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad+y,pad+x,c] = np.median(tmp[y:y+K_size,x:x+K_size,c])

    out = np.clip(out, 0, 255)
    out = out[pad:pad+H,pad:pad+W].astype(np.uint8)

    return out


img = cv2.imread("imori_noise.jpg")

out = median_filter(img, 3)

cv2.imwrite('answer_10.jpg',out)
cv2.imshow('answer_10.jpg',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
