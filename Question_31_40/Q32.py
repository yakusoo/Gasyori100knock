import cv2
import numpy as np

# DFT hyper-parameters
K, L = 128, 128
channel = 3

def RGB2GRAY(img):
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray

def DFT(img):
    H, W = img.shape
    # prepare DFT coefficient
    G = np.zeros((L, K, channel), dtype=np.complex)

    # prepare processed index corresponding to original image positions
    x = np.tile(np.arange(W), (H, 1))
    y = np.arange(H).repeat(W).reshape(H, -1)

    # dft
    for c in range(channel):
        for k in range(W):
            for l in range(H):
                G[l, k, c] = np.sum( img[..., c] * np.exp(-2j * np.pi * (k * x / K + l * y / L))) / np.sqrt(K * L)

    return G

# IDFT
def IDFT(G):
    H, W, _ = G.shape
    out = np.zeros((H, W, channel), dtype=np.float32)

    # prepare processed index index corresponding to original image positions
    x = np.tile(np.arange(W), (H, 1))
    y = np.arange(H).repeat(W).reshape(H, -1)

    # idft
    for c in range(channel):
        for k in range(W):
            for l in range(H):
                out[l, k, c] = np.abs(np.sum( G[..., c] * np.exp(2j * np.pi * (k * x / W + l * y / H)))) / np.sqrt(W * H)

    # clipping
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out

# read image
img = cv2.imread("imori.jpg").astype(np.float)
gray = RGB2GRAY(img)

# dft
G = DFT(gray)

# write poser spectal to image
ps = (np.abs(G) / np.abs(G).max() * 255).astype(np.uint8)
cv2.imshow("out_ps.jpg", ps)

# IDFT
out = IDFT(G)

cv2.imwrite('answer32.jpg',out)
cv2.imshow('answer32.jpg',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
