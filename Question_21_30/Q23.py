import cv2
import numpy as np
import matplotlib.pyplot as plt

#histogram equalization
def hist_equalization(img,  z_max=255):
    H, W, C = img.shape
    out = img.copy()

    S = H * W * C * 1.

    h_sum = 0.

    for i in range(1,255):
        ind = np.where(img == i)
        h_sum += len(img[ind])
        z_prime = z_max / S * h_sum
        out[ind] = z_prime

    out = out.astype(np.uint8)

    return out

img = cv2.imread("imori.jpg").astype(np.float)

out = hist_equalization(img)

# Display histogram
plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("answer_23_2.png")
plt.show()

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("answer_23_1.jpg", out)
