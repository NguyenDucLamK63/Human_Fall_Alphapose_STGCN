import cv2
import numpy as np
import glob

img_array = []
for filename in sorted(glob.glob('/home/duclam/Documents/dataset_action/UP-Fall_detection_dataset/Subject1Activity7Trial1Camera1/*.png')):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    print(filename)
    img_array.append(img)
    print(size)
    # print(img_array)

out = cv2.VideoWriter('/home/duclam/Documents/dataset_action/UP-Fall_detection_dataset/Subject1Activity7Trial1Camera1_222.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
# out.write(img_array)
print(len(img_array))
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()