import cv2
import numpy as np

img = cv2.imread("imori.jpg")
out = np.zeros_like(img)
out[:,:,(2,1,0)] = img[:,:,(0,1,2)].copy()

cv2.imwrite('answer_1.jpg',out)
cv2.imshow('answer_1.jpg',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
