import cv2
import numpy as np
import matplotlib.pyplot as plt

#histogram normalization
def hist_normalization(img,a,b):
    c = img.min()
    d = img.max()
    print(c,d)

    out = img.copy()

    out = (b - a) / (d -c) * (out - c) + a
    out[out < a] = a
    out[out > b] = b

    out = out.astype(np.uint8)

    return out

img = cv2.imread("imori_dark.jpg").astype(np.float)

out = hist_normalization(img, 0, 255)

# Display histogram
plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("answer_21_2.png")
plt.show()

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("answer_21_1.jpg", out)
