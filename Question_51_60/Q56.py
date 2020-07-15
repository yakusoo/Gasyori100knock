import cv2
import numpy as np

def Template_matching(img, template):
    H, W, C = img.shape
    ht, wt, ct = template.shape

    S = -1
    i, j = -1, -1

    for y in range(0,H-ht):
        for x in range(0,W-wt):
            _s = np.sum(img[y:y+ht,x:x+wt] * template)
            _s /= np.sqrt(np.sum(img[y:y+ht,x:x+wt]**2))*np.sqrt(np.sum(template**2))
            if _s > S:
                S = _s
                i, j = x, y

    out = img.copy()

    cv2.rectangle(out, (i,j), (i+wt,j+ht), (0,0,255), 1)
    out = out.astype(np.uint8)

    return out

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# Read templete image
template = cv2.imread("imori_part.jpg").astype(np.float32)

# Template matching
out = Template_matching(img, template)


# Save result
cv2.imwrite("answer_55.jpg", out)
cv2.imshow("answer_55.jpg", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
