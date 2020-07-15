import cv2
import numpy as np

def Alpha_blend(img1, img2, alpha):
    out = img1 * alpha + img2 * (1 - alpha)
    out = out.astype(np.uint8)
    return out

# Read image
img1 = cv2.imread("imori.jpg").astype(np.float32)
img2 = cv2.imread("thorino.jpg").astype(np.float32)
out = Alpha_blend(img1, img2, 0.6)

# Save result
cv2.imwrite("answer_60.jpg", out)
cv2.imshow("answer_60.jpg", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
