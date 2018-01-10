import tifffile as tiff
import numpy as np
import cv2 as cv

img = cv.imread('./result.png')
print(img)

data = np.random.rand(5, 301, 219)
tiff.imsave('/home/seung/UDnet/temp.tif', data)
