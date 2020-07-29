import cv2
import numpy as np

# Affine
def affine(img, a, b, c, d, tx, ty):
    H, W, C = img.shape

    # temporary image
    img = np.zeros((H+2, W+2, C), dtype=np.float32)
    img[1:H+1,1:W+1] = _img

    # get new image shape
    H_new = np.round(H * d).astype(np.int)
    W_new = np.round(W * a).astype(np.int)
    out = np.zeros((H_new+1, W_new+1, C), dtype=np.float32)

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

    out = out[:H_new, :W_new]
    out = out.astype(np.uint8)

    return out

# Read image
_img = cv2.imread("imori.jpg").astype(np.float32)

# Affine
out = affine(_img, a=1.3, b=0, c=0, d=0.8, tx=0, ty=0)
out2 = affine(_img, a=1, b=0, c=0, d=1, tx=30, ty=-30)

# Save result
cv2.imwrite("answer_29_1.jpg", out)
cv2.imshow("answer_29_1", out)
while cv2.waitKey(100) != 27:# loop if not get ESC
    if cv2.getWindowProperty('answer_29_1',cv2.WND_PROP_VISIBLE) <= 0:
        break
cv2.destroyWindow('answer_29_1')
cv2.imwrite("answer_29_2.jpg", out2)
cv2.imshow("answer_29_2", out2)
while cv2.waitKey(100) != 27:# loop if not get ESC
    if cv2.getWindowProperty('answer_29_2',cv2.WND_PROP_VISIBLE) <= 0:
        break
cv2.destroyWindow('answer_29_2')
cv2.destroyAllWindow()
