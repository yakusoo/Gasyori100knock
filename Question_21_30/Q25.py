import cv2
import numpy as np

#Nearest Neighbor interpolation
def NN_interpolation(img, ax=1.5, ay=1.5):
    H, W, C = img.shape

    aH = int(H * ay)
    aW = int(W * ax)

    y = np.arange(aH).repeat(aW).reshape(aH, -1)
    x = np.tile(np.arange(aW),(aH, 1))

    y = np.round(y / ay).astype(np.int)
    x = np.round(x / ax).astype(np.int)

    out = img[y,x]
    
    out = out.astype(np.uint8)

    return out

img = cv2.imread("imori.jpg").astype(np.float)

out = NN_interpolation(img)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("answer_25.jpg", out)
