import cv2
import numpy as np
import matplotlib.pyplot as plt

#histogram manipulation
def hist_maniplation(img,m0,s0):
    m = np.mean(img)
    s = np.std(img)

    out = img.copy()

    out = s0 / s * (out - m) + m0
    out[out < 0] = 0
    out[out > 255] = 255

    out = out.astype(np.uint8)

    return out

img = cv2.imread("imori_dark.jpg").astype(np.float)

out = hist_maniplation(img, 128, 52)

# Display histogram
plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("answer_22_2.png")
plt.show()

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("answer_22_1.jpg", out)
