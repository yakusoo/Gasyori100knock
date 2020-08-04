import cv2
import numpy as np

# Affine
def affine(img, dx, dy):
    H, W, C = img.shape

    a = 1.
    b = dx / H
    c = dy / W
    d = 1.
    tx = 0.
    ty = 0.

    # temporary image
    img = np.zeros((H+2, W+2, C), dtype=np.float32)
    img[1:H+1,1:W+1] = _img

    # get new image shape
    H_new = np.ceil(dy + H).astype(np.int)
    W_new = np.ceil(dx + W).astype(np.int)
    out = np.zeros((H_new, W_new, C), dtype=np.float32)

    # get position of new image
    x_new = np.tile(np.arange(W_new), (H_new, 1))
    y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

    # get position of original image by affine
    adbc = a * d - b * c
    x = np.round((d * x_new - b * y_new) / adbc).astype(np.int) - tx + 1
    y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

    x = np.minimum(np.maximum(x, 0), W + 1).astype(np.int)
    y = np.minimum(np.maximum(y, 0), H + 1).astype(np.int)

    # assgin pixcel to new image
    out[y_new, x_new] = img[y, x]
    out = out.astype(np.uint8)

    return out

# Read image
_img = cv2.imread("imori.jpg").astype(np.float32)

# Affine
out1 = affine(_img, dx=30, dy=0)
out2 = affine(_img, dx=0, dy=30)
out3 = affine(_img, dx=30, dy=30)

# Save result
cv2.imwrite("answer_31_1.jpg", out1)
cv2.imshow("answer_31_1", out1)
while cv2.waitKey(100) != 27:# loop if not get ESC
    if cv2.getWindowProperty('answer_31_1',cv2.WND_PROP_VISIBLE) <= 0:
        break
cv2.destroyWindow('answer_31_1')
cv2.imwrite("answer_31_2.jpg", out2)
cv2.imshow("answer_31_2", out2)
while cv2.waitKey(100) != 27:# loop if not get ESC
    if cv2.getWindowProperty('answer_31_2',cv2.WND_PROP_VISIBLE) <= 0:
        break
cv2.destroyWindow('answer_31_2')
cv2.imwrite("answer_31_3.jpg", out3)
cv2.imshow("answer_31_3", out3)
while cv2.waitKey(100) != 27:# loop if not get ESC
    if cv2.getWindowProperty('answer_31_3',cv2.WND_PROP_VISIBLE) <= 0:
        break
cv2.destroyWindow('answer_31_3')
cv2.destroyAllWindow()
