import cv2
import numpy as np

def BGR2GRAY(img):
    red = img[:,:,2].copy()
    green = img[:,:,1].copy()
    blue = img[:,:,0].copy()
    gray = 0.2126*red + 0.7152*green + 0.0722*blue

    return gray

def laplacian_filter(img, K_size):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    #zeropadding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad:pad+H,pad:pad+W] = img.copy().astype(np.float)

    K = [[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]

    tmp = out.copy()

    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad+y,pad+x,c] = np.sum(K * tmp[y:y+K_size,x:x+K_size,c])

    out = np.clip(out, 0, 255)
    out = out[pad:pad+H,pad:pad+W].astype(np.uint8)

    return out


img = cv2.imread("imori.jpg").astype(np.float)
img = BGR2GRAY(img)
out = laplacian_filter(img, 3)

cv2.imwrite('answer_17.jpg',out)
cv2.imshow('answer_17.jpg',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
