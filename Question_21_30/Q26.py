import cv2
import numpy as np

#Bi-linear interpolation
def BL_interpolation(img, ax=1.5, ay=1.5):
    H, W, C = img.shape

    aH = int(H * ay)
    aW = int(W * ax)

    #get position of resized image
    y = np.arange(aH).repeat(aW).reshape(aH, -1)
    x = np.tile(np.arange(aW),(aH, 1))

    #get position of original image
    y = (y / ay)
    x = (x / ax)

    ix = np.floor(x).astype(np.int)
    iy = np.floor(y).astype(np.int)
    print(ix.shape)
    print(ix)

    ix = np.minimum(ix, W-2)
    iy = np.minimum(iy, H-2)
    print(ix.shape)
    print(ix)
    print(W-2)

	# get distance
    dx = x - ix
    dy = y - iy
    print(dx.shape)

    dx = np.repeat(np.expand_dims(dx, axis=-1), 3, axis=-1)
    dy = np.repeat(np.expand_dims(dy, axis=-1), 3, axis=-1)
    print(dx.shape)

	# interpolation
    out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out

img = cv2.imread("imori.jpg").astype(np.float)

out = BL_interpolation(img)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("answer_26.jpg", out)
