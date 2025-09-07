import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("/home/tushara/Documents/Ashoka/Lab 4/Data/exp1/thickness_measurement.png", cv2.IMREAD_GRAYSCALE)

plt.imshow(img)
plt.show()
