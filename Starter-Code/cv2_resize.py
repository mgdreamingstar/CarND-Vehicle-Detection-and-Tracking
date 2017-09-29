import cv2
import matplotlib.pyplot as plt
import os
os.getcwd()
img = cv2.imread(R'..\examples\output_bboxes.png')
type(img)
print(img.shape)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
img_resize = cv2.resize(img,(np.int(img.shape[1]/2),np.int(img.shape[0]/2)))
plt.imshow(cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB))
img_resize.shape
