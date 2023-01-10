# # list = [2,3,6]
# # list_2 = []
# # for i in range(3):
# #     list_2.append(i)
# #     print(i)
# # print(list_2)
# # if list_2 == [0,1,2] :
# #     print("pass")
# # if list == [2,3,6] :
# #     print("ok")
# # for i in range(6):
# # 	if list_2 == [0,1,2] :
# #         print("pass")
# #     else :
# # -*- coding: utf-8 -*-

# from keras.models import load_model
# import numpy as np
# import os
# import cv2
# import time
# import onnxruntime


# batch_size = 1 #Numero de muestras para cada batch (grupo de entrada)

# def load_test():
#  X_test = []
#  images_names = []
#  image_path = './src/Samples_cropped'
#  print('Read test images')
 
#  for imagen in [imagen for imagen in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, imagen))]:
#   imagenes = os.path.join(image_path, imagen)
#   print(imagenes)
#   img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (224, 224))
#   # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#   X_test.append(img)
#   images_names.append(imagenes)
#  return X_test, images_names

# def read_and_normalize_test_data():
#     test_data, images_names = load_test()
#     test_data = np.array(test_data, copy=False, dtype=np.float32)
#     return test_data, images_names
  

# test_data, images_names = read_and_normalize_test_data()
# y=test_data

# session = onnxruntime.InferenceSession('/home/maicg/Documents/FaceQnet/model.onnx', providers=['CUDAExecutionProvider'])


# t1 = time.time()
# results = session.run(['main_output'], {'base_input': y})
# t2 = time.time()
# tg = (t2-t1)/94
# print(tg)
# print(results)
# # #Loading one of the the pretrained models

# # # model = load_model('FaceQnet.h5')

# # model = load_model('FaceQnet_v1.h5')

# # #See the details (layers) of FaceQnet
# # # print(model.summary())

# # #Loading the test data
# # test_data, images_names = read_and_normalize_test_data()
# # y=test_data

# # #Extract quality scores for the samples
# # t1 = time.time()
# # score = model.predict(y, batch_size=batch_size, verbose=1)
# # predictions = score
# # t2 = time.time()
# # tg = (t2-t1)/5

# # #Saving the quality measures for the test images
# # fichero_scores = open('scores_quality_me.txt','w')
# # i=0



# # #Saving the scores in a file
# # fichero_scores.write("img;score\n")

# # for item in predictions:
# #  fichero_scores.write("%s" % images_names[i])
# #  #Constraining the output scores to the 0-1 range
# #  #0 means worst quality, 1 means best quality
# #  if float(predictions[i])<0:
# #   predictions[i]='0'
# #  elif float(predictions[i])>1:
# #   predictions[i]='1'
# #  fichero_scores.write(";%s\n" % predictions[i])
# #  i=i+1



# # # print(tg)
# import cv2
# img = "/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_2/Data/ouput/test_cty_23_11_2person/img_000024.jpg"
# img = cv2.imread(img, cv2.IMREAD_COLOR)
# print(img.shape)
# img = img.swapaxes(1, 2).swapaxes(0, 1)
# print(img.shape)
# # img = cv2.resize(img, (112, 112))
# # print(img.shape)
import torch 
seq_len = torch.randint(1, 5, (1,)).item()
print(seq_len)
x = torch.randn(1, 2, 3, seq_len)
print(x.shape)
print("xxxxxxxxxxxxxxxxx :", x)