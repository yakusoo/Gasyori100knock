import cv2
import numpy as np

img = cv2.imread("imori.jpg").astype(np.float)
gray = np.zeros_like(img)
red = img[:,:,2].copy()
green = img[:,:,1].copy()
blue = img[:,:,0].copy()
gray = 0.2126*red + 0.7152*green + 0.0722*blue
out = gray.copy().astype(np.uint8)

cv2.imwrite('answer_2.jpg',out)
cv2.imshow('answer_2.jpg',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
