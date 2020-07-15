import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("imori_dark.jpg")
plt.hist(img.ravel(),256,[0,256])
plt.show()
