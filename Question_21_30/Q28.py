import cv2
import numpy as np

# Affine
def affine(_img, a, b, c, d, tx, ty):
    H, W, C = img.shape

    # temporary image
    img = np.zeros((H+2, W+2, C), dtype=np.float32)
    img[1:H+1,1:W+1] = _img

    # get new image shape
    H_new = np.round

    # get position of new image

    # get position of original image by affine

    # assgin pixcel to new image




# Read image
_img = cv2.imread("imori.jpg").astype(np.float32)

# Affine
out = affine(_img, a=1, b=0, c=1, d=1, tx=30, ty=-30)

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("answer_28.jpg", out)
