import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import norm

class ImageSynthesizer:
    def __init__(self, kernlen=100, nsig=1):
        self.kernlen = kernlen
        self.nsig = nsig
        self.g_mask = self._create_vignetting_mask()

    def _create_gaussian_kernel(self):
        interval = (2 * self.nsig + 1.) / self.kernlen
        x = np.linspace(-self.nsig - interval / 2., self.nsig + interval / 2., self.kernlen + 1)
        kern1d = np.diff(norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel / kernel.max()
        return kernel

    def _create_vignetting_mask(self):
        kernel = self._create_gaussian_kernel()
        mask = np.dstack((kernel, kernel, kernel))
        return mask

    def synthesis(self, t, r, sigma):
        t_init = t
        t = np.power(t, 2.2)
        r = np.power(r, 2.2)
        t = np.clip(t * 0.2, 0, 1)  # Decrease the brightness of transmission layer.

        sz = int(2 * np.ceil(2 * sigma) + 1)
        r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
        blend = r_blur + t

        att = 1.08 + np.random.random() / 10.0

        for i in range(3):
            maski = blend[:, :, i] > 1
            mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
            r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
        r_blur[r_blur >= 1] = 1
        r_blur[r_blur <= 0] = 0

        h, w = r_blur.shape[0:2]
        neww = np.random.randint(0, 560 - w - 10)
        newh = np.random.randint(0, 560 - h - 10)
        alpha1 = self.g_mask[newh:newh + h, neww:neww + w, :]
        alpha2 = 1 - np.random.random() / 5.0
        r_blur_mask = np.multiply(r_blur, alpha1)
        blend = r_blur_mask + t * alpha2

        t = np.power(t, 1 / 2.2)
        r_blur_mask = np.power(r_blur_mask, 1 / 2.2)
        blend = np.power(blend, 1 / 2.2)
        blend[blend >= 1] = 1
        blend[blend <= 0] = 0

        return t_init, r_blur_mask, blend

