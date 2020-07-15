import cv2
import numpy as np

def max_pooling(img, G):
    out = img.copy()

    H, W, C = img.shape

    nH = int(H / G)
    nW = int(W / G)

    for y in range(nH):
        for x in range(nW):
            for c in range(C):
                out[x*G:(x+1)*G,y*G:(y+1)*G,c] = np.max(out[x*G:(x+1)*G,y*G:(y+1)*G,c]).astype(np.int)
    return out


img = cv2.imread("imori.jpg")

out = max_pooling(img, 8)

cv2.imwrite('answer_8.jpg',out)
cv2.imshow('answer_8.jpg',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
