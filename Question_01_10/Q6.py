import cv2
import numpy as np

img = cv2.imread("imori.jpg")

out = img.copy()
out = out // 64 * 64 + 32

cv2.imwrite('answer_6.jpg',out)
cv2.imshow('answer_6.jpg',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
