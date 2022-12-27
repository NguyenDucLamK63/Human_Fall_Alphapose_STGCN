"""
This script to extract skeleton joints position and score.
"""

import os
import cv2
import time
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q

from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from fn import vis_frame_fast
import argparse

from Track.Tracker import Detection, Tracker

def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                    kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))
if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                            help='Backbone model for SPPE FastPose model.')
    par.add_argument('--detection_input_size', type=int, default=384,
                        help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--device', type=str, default='cuda',
                        help='Device to run model on cpu or cuda.')
    par.add_argument('--pose_input_size', type=str, default='224x160',
                        help='Size of input in pose model must be divisible by 32 (h, w)')
    args = par.parse_args()
    save_path = '/home/duclam/Documents/dataset_action/Le2i_FDD_fall/Lecture_room/test/video/Fall_FDD_lecture_room_2_pose_224_160_384.csv'

    video_folder = '/home/duclam/Documents/dataset_action/Le2i_FDD_fall/Lecture_room/test/video'
    annot_file = '/home/duclam/Documents/dataset_action/Le2i_FDD_fall/Lecture_room/test/video/Fall_FDD_lecture_room_2.csv'
    annot_folder = ''  # bounding box annotation for each frame.
    
    # DETECTION MODEL.
    device = args.device
    inp_dets = args.detection_input_size #384
    # khởi tạo device và image size và khởi tạo mạng darNet cho YoloV3
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)


    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1])) # đưa về dạng (224,160)
    #khoi tao tham so pose va load model pose
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 30
    # tracker = Tracker(max_age=max_age, n_init=3)
    resize_fn = ResizePadding(inp_dets, inp_dets)
    # with score.
    columns = ['video', 'frame', 'Nose_x', 'Nose_y', 'Nose_s', 'LShoulder_x', 'LShoulder_y', 'LShoulder_s',
            'RShoulder_x', 'RShoulder_y', 'RShoulder_s', 'LElbow_x', 'LElbow_y', 'LElbow_s', 'RElbow_x',
            'RElbow_y', 'RElbow_s', 'LWrist_x', 'LWrist_y', 'LWrist_s', 'RWrist_x', 'RWrist_y', 'RWrist_s',
            'LHip_x', 'LHip_y', 'LHip_s', 'RHip_x', 'RHip_y', 'RHip_s', 'LKnee_x', 'LKnee_y', 'LKnee_s',
            'RKnee_x', 'RKnee_y', 'RKnee_s', 'LAnkle_x', 'LAnkle_y', 'LAnkle_s', 'RAnkle_x', 'RAnkle_y',
            'RAnkle_s', 'label']


    def normalize_points_with_size(points_xy, width, height, flip=False):
        points_xy[:, 0] /= width
        points_xy[:, 1] /= height
        if flip:
            points_xy[:, 0] = 1 - points_xy[:, 0]
        return points_xy


    annot = pd.read_csv(annot_file)
    vid_list = annot['video'].unique()
    for vid in vid_list:
        # khởi tạo tham số Tracker và theo dõi người
        tracker = Tracker(max_age=max_age, n_init=3)
        print("video : ", vid)
        print(f'Process on: {vid}')
        df = pd.DataFrame(columns=columns)
        cur_row = 0

        # Bounding Boxs Labels.
        # video_annot = pd.read_csv(os.path.join(annot_folder, vid.split('.')[0]) + '.txt',
        #                           header=None, names=[1, 2, 3, 4, 5, 6])
        # video_annot = video_annot.dropna().reset_index(drop=True)
        # video_annot_frames = video_annot.iloc[:, 0].tolist()

        # Pose Labels.
        frames_label = annot[annot['video'] == vid].reset_index(drop=True)

        cap = cv2.VideoCapture(os.path.join(video_folder, vid))
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cam_source = os.path.join(video_folder, vid)
        print("cam_source :", cam_source)
        if type(cam_source) is str and os.path.isfile(cam_source):
            # Use loader thread with Q for video file.
            cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
        else:
            # Use normal thread loader for webcam.
            cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                            preprocess=preproc).start()

        # assert frames_count == len(video_annot_frames), 'frame count not equal! {} and {}'.format(
        #     frames_count, len(video_annot_frames))
        # Bounding Boxs Labels.
        # annot_file = os.path.join(annot_folder, vid.split('.')[0], '.txt')
        # annot = None
        # if os.path.exists(annot_file):
        #     annot = pd.read_csv(annot_file, header=None,
        #                               names=['frame_idx', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
        #     annot = annot.dropna().reset_index(drop=True)

        #     assert frames_count == len(annot), 'frame count not equal! {} and {}'.format(frames_count, len(annot))

        fps_time = 0
        i = 1
        f = 0
        while cam.grabbed(): #Return True nếu cam đọc được frame
            f += 1
            #Lấy frame trong queue
            frame = cam.getitem()
            image = frame.copy()
            print("Frame : ",f , '\n')
            cls_idx = int(frames_label[frames_label['frame'] == i]['label'])
            # Detect humans bbox in the frame with detector model.
            detected = detect_model.detect(frame, need_resize=False, expand_bb=10) #384x384
            # bbox = detected[0, :4]
            print("detected :", detected)
            tracker.predict()
            # Merge two source of predicted bbox together.
            for track in tracker.tracks: #[]
                det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
                detected = torch.cat([detected, det], dim=0) if detected is not None else det
            detections = []  # List of Detections object for tracking., Lưu lại các bbox và keypoint (13x3)
            # print("type: ", type(detected))
            # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
            # Merge two source of predicted bbox together.
            # print("detected[:, 0:4] :" , detected[:, 0:4])
            if detected is not None:
                result = []
                # detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
                # Predict skeleton pose of each bboxs.
                result = pose_model.predict(frame, detected[:, 0:4], detected[:, 4]) # 13x2 (float32)
                print("poses : ", result)

                # Create Detections object.
                detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                        np.concatenate((ps['keypoints'].numpy(),
                                                        ps['kp_score'].numpy()), axis=1),
                                        ps['kp_score'].mean().numpy()) for ps in result]
                tracker.update(detections)
                # bbox = track.to_tlbr().astype(int)
                if len(result) > 0:
                    pt_norm = normalize_points_with_size(result[0]['keypoints'].numpy().copy(),
                                                        frame_size[0], frame_size[1])
                    pt_norm = np.concatenate((pt_norm, result[0]['kp_score']), axis=1)
                    #idx = result[0]['kp_score'] <= 0.05
                    #pt_norm[idx.squeeze()] = np.nan
                    row = [vid, i, *pt_norm.flatten().tolist(), cls_idx]
                    scr = result[0]['kp_score'].mean()
                else:
                    row = [vid, i, *[np.nan] * (13 * 3), cls_idx]
                    scr = 0.0

                df.loc[cur_row] = row
                cur_row += 1

                # VISUALIZE.
                frame = vis_frame_fast(frame, result)
                # try :
                #     frame = cv2.rectangle(frame, (detected[:, 0:1], detected[:, 1:2]), (detected[:, 2:3], detected[:, 3:4]), (0, 255, 0), 2)
                # except:
                #     continue
                frame = cv2.putText(frame, 'Frame: {}, Pose: {}, Score: {:.4f}'.format(i, cls_idx, scr),
                                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                frame = cv2.putText(frame, 'video : {}'.format(vid),
                                    (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                frame = frame[:, :, ::-1]
                fps_time = time.time()
                i += 1

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cam.stop()

        if os.path.exists(save_path):
            df.to_csv(save_path, mode='a', header=False, index=False)
        else:
            df.to_csv(save_path, mode='w', index=False)
        cv2.destroyAllWindows()