import cv2
import numpy as np

def BGR2HSV(img_):
    img = img_.copy() / 255.
    hsv = np.zeros_like(img, dtype=np.float32)

    max_v = np.max(img, axis=2).copy()
    min_v = np.min(img, axis=2).copy()
    min_arg = np.argmin(img, axis=2)

    #H
    hsv[:,:,0][np.where(max_v == min_v)] = 0
    ind = np.where(min_arg == 0)
    hsv[:,:,0][ind] = 60 * (img[:,:,1][ind] - img[:,:,2][ind]) / (max_v[ind] - min_v[ind]) + 60
    ind = np.where(min_arg == 2)
    hsv[:,:,0][ind] = 60 * (img[:,:,0][ind] - img[:,:,1][ind]) / (max_v[ind] - min_v[ind]) + 180
    ind = np.where(min_arg == 1)
    hsv[:,:,0][ind] = 60 * (img[:,:,2][ind] - img[:,:,0][ind]) / (max_v[ind] - min_v[ind]) + 300

    hsv[:,:,1] = max_v - min_v
    hsv[:,:,2] = max_v

    return hsv

def HSV2BGR(_img,hsv):
    img = _img.copy() / 255.
    out = np.zeros_like(_img)

    max_v = np.max(img, axis=2).copy()
    min_v = np.min(img, axis=2).copy()

    H = hsv[:,:,0]
    S = hsv[:,:,1]
    V = hsv[:,:,2]

    C = S
    H_ = H / 60.
    X = C * (1 - np.abs(H_ % 2 - 1))
    Z = np.zeros_like(H)

    vals = [[Z,X,C],[Z,C,X],[X,C,Z],[C,X,Z],[C,Z,X],[X,Z,C]]

    for i in range(6):
        ind = np.where((i <= H_) & (H_ < (i+1)))
        out[:,:,0][ind] = (V - C)[ind] + vals[i][0][ind]
        out[:,:,1][ind] = (V - C)[ind] + vals[i][1][ind]
        out[:,:,2][ind] = (V - C)[ind] + vals[i][2][ind]

    out[np.where(max_v == min_v)] = 0
    out = np.clip(out, 0, 1)
    out = (out * 255).astype(np.uint8)

    return out

def color_tracking(hsv):
    mask = np.zeros_like(hsv)

    mask[np.where((hsv[:,:,0]>=180)&(hsv[:,:,0]<=260))] = 255

    return mask

img = cv2.imread("imori.jpg").astype(np.float32)
hsv = BGR2HSV(img)
out = color_tracking(hsv)

cv2.imwrite('answer_70.jpg',out)
cv2.imshow('answer_70.jpg',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
