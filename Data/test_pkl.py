import os
import cv2
import time
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import argparse

from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from fn import vis_frame_fast

if __name__ == '__main__':
    # frame = cv2.imread("/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_2/Data/ouput/video (1)/img_000059.jpg", cv2.IMREAD_COLOR)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # detector = TinyYOLOv3_onecls()
    # bb = detector.detect(frame)[0, :4].numpy().astype(int)
    # print("bb : ", bb)
    # obj = pd.read_pickle(r'/home/duclam/Documents/dataset_action/Le2i_FDD_fall/Home/Home_FDD_init.pkl') #576 , 15 non ==> bỏ non và mỗi class thì lưu lại dạng 30 frame
    obj = pd.read_pickle(r'/home/duclam/Documents/dataset_action/Le2i_FDD_fall/Home/Home_FDD_fix_20_fall_101.pkl')
    # obj = pd.read_pickle(r'/home/duclam/Documents/dataset_action/train.pickle')
    # obj = pd.read_pickle(r'/home/duclam/Documents/dataset_action/Le2i_FDD_fall/Cafe_17_test/Cafe_FDD_17.pkl')
    print(obj) #[0:10882][0:30][0:14][0:3] , float 64 : Home_1  
    # print(obj) #[0:10882][0:30][0:14][0:3] , float 64 : Home_1 
    # print(obj) #[0:3087][0:273][0:21][0:3] 