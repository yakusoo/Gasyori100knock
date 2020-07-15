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

def color_tracking(hsv):
    mask = np.zeros_like(hsv[:,:,0])

    mask[np.where((hsv[:,:,0]>=180)&(hsv[:,:,0]<=260))] = 1

    return mask

def masking(img, mask):
    mask = 1 - mask
    out = img.copy()
    mask = np.tile(mask, [3,1,1]).transpose([1,2,0])
    out *= mask

    return out

def Morphology_Erode(img, Ero_time):
    H, W = img.shape

    out = img.copy()

    MF = np.array(((0,1,0),(1,0,1),(0,1,0)), dtype=np.int)

    for i in range (Ero_time):
        tmp = np.pad(out,(1,1),'edge')
        for y in range (1, H+1):
            for x in range (1, W+1):
                if np.sum(MF * tmp[y-1:y+2,x-1:x+2]) < 1*4:
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
                if np.sum(MF * tmp[y-1:y+2,x-1:x+2]) > 1:
                    out[y-1,x-1] = 1

    return out

def Morphology_Opening(img, time):
    out = Morphology_Erode(img, time)
    out = Morphology_Dilate(img, time)
    return out

def Morphology_Closing(img, time):
    out = Morphology_Dilate(img, time)
    out = Morphology_Erode(out, time)
    return out

img = cv2.imread("imori.jpg").astype(np.float32)

hsv = BGR2HSV(img / 255.)
print(hsv.shape)

mask = color_tracking(hsv)
print(mask.shape)

mask = Morphology_Closing(mask, 5)
mask = Morphology_Opening(mask, 5)

out = masking(img, mask)

out = out.astype(np.uint8)


cv2.imwrite('answer_72.jpg',out)
cv2.imshow('answer_72.jpg',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
