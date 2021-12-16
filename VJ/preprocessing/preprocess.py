import cv2
import os
import numpy as np

class Preprocess:
    
    def histogram_equalization_rgb(self, img):
        # Simple preprocessing using histogram equalization 
        # https://en.wikipedia.org/wiki/Histogram_equalization

        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # For Grayscale this would be enough:
        #img = cv2.equalizeHist(img)

        return img

    def brightness_correction(self, img, alpha, beta):
        new_image = np.zeros(img.shape, img.dtype)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                for c in range(img.shape[2]):
                    new_image[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)
        return new_image

    def resize(self, img, dsize):
        return cv2.resize(img, (dsize,dsize))
    
    # Sharpening an image 
    def edge_enhancment(self, img):
        kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
        image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
        return image_sharp
    
    # Normalize pixel values to [0-1] 
    def pixel_normalization(self, img):
        norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm_img

    def blur(self, img, type: str, ksize: int):
        kernel = (ksize,ksize)
        if type == 'mean':
            return cv2.blur(img, kernel)
        elif type == 'gaussian':
            return cv2.GaussianBlur(img, kernel, 0)
        elif type == 'median':
            return cv2.medianBlur(img, ksize)

    def change_contrast_brightness(self, img, alpha, beta):
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    