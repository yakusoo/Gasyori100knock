import cv2
import numpy as np

def BGR2GRAY(img):
    red = img[:,:,2].copy()
    green = img[:,:,1].copy()
    blue = img[:,:,0].copy()
    gray = 0.2126*red + 0.7152*green + 0.0722*blue

    return gray

def sobel_filter(img, K_size):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    #zeropadding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad:pad+H,pad:pad+W] = img.copy().astype(np.float)

    K_v = [[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]]
    K_h = [[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]]

    out_v = out.copy()
    out_h = out.copy()

    tmp_v = out.copy()
    tmp_h = out.copy()

    for y in range(H):
        for x in range(W):
            for c in range(C):
                out_v[pad+y,pad+x,c] = np.sum(K_v * tmp_v[y:y+K_size,x:x+K_size,c])
                out_h[pad+y,pad+x,c] = np.sum(K_h * tmp_h[y:y+K_size,x:x+K_size,c])

    out_v = np.clip(out_v, 0, 255)
    out_h = np.clip(out_h, 0, 255)
    out_v = out_v[pad:pad+H,pad:pad+W].astype(np.uint8)
    out_h = out_h[pad:pad+H,pad:pad+W].astype(np.uint8)

    return out_v, out_h


img = cv2.imread("imori.jpg").astype(np.float)
img = BGR2GRAY(img)
out_v,out_h = sobel_filter(img, 3)

cv2.imwrite('answer_16_v.jpg',out_v)
cv2.imshow('answer_16_v.jpg',out_v)
while cv2.waitKey(100) != 27:# loop if not get ESC
    if cv2.getWindowProperty('answer_16_v.jpg',cv2.WND_PROP_VISIBLE) <= 0:
        break
cv2.destroyWindow('answer_16_v.jpg')

cv2.imwrite('answer_16_h.jpg',out_h)
cv2.imshow('answer_16_h.jpg',out_h)
while cv2.waitKey(100) != 27:
    if cv2.getWindowProperty('answer_16_h.jpg',cv2.WND_PROP_VISIBLE) <= 0:
        break
cv2.destroyWindow('answer_16_h.jpg')
cv2.destroyAllWindows()
