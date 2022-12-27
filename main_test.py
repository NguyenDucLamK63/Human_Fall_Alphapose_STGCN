import os
import cv2
import time
import torch
import argparse
import numpy as np

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

#source = '../Data/test_video/test7.mp4'
#source = '../Data/falldata/Home/Videos/video (2).avi'  # hard detect
# source = '../Data/falldata/Home/Videos/video (1).avi'
source = '/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_ver3_yolov5/Fall_1_persion_1_qt.mp4'
#source = 2


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
    par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
                        help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=384,
                        help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='224x160',
                        help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                        help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                        help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                        help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default='',
                        help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                        help='Device to run model on cpu or cuda.')
    args = par.parse_args()

    device = args.device

    # DETECTION MODEL.
    inp_dets = args.detection_input_size #384
    # khởi tạo device và image size và khởi tạo mạng darNet cho YoloV3
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    # Chia input size tách thanh sau 2 phân biệt bằng dấu x
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1])) # đưa về dạng (224,160)
    #khoi tao tham so pose va load model pose
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 30
    # khởi tạo tham số Tracker và theo dõi người
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    #khoi tao model nhận diện hành động
    action_model = TSSTG()
    # Chua hieu (Khoi tao ham de resize anh, tao anh 3 chieu)
    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()
    #frame_size = cam.frame_size
    #scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]
    # Lưu video output
    outvid = False
    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))

    fps_time = 0
    f = 0
    # Trả về True nếu đọc được frame trong queue
    fall_down = np.zeros((50,), dtype = np.int) 
    # Lying_Down = np.zeros((50,), dtype = np.int) 
    temp = np.zeros((50,), dtype = np.int) 
    list_action_1 = []
    list_action_2 = []
    list_action_3 = []
    fall = np.zeros((50,), dtype = np.int) 
    flag_1 = False
    flag_2 = False
    flag_3 = False
    k = np.zeros((50,), dtype = np.int)
    # bbox_1 = [0,0,0,0] 
    # bbox_2 = [0,0,0,0]
    # bbox_3 = [0,0,0,0]
    while cam.grabbed(): #Return True nếu cam đọc được frame
        f += 1
        #Lấy frame trong queue
        frame = cam.getitem()
        image = frame.copy()
        print("Frame : ",f , '\n')
        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10) #384x384
        print("detected :", detected)
        # print("type: ", type(detected))
        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks: #[]
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det
        detections = []  # List of Detections object for tracking., Lưu lại các bbox và keypoint (13x3)
        # print("detected[:, 0:4] :" , detected[:, 0:4])
        if detected is not None:
            # detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4]) # 13x2 (float32)
            print("poses : ", poses)

            # Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]
            # print("detections[0] : ",detections[0])
            # print("detections[1] : ",detections[1])
            # VISUALIZE.
            if args.show_detected:
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue
            i = i + 1
            if i == 1 :
                bbox_1 = track.to_tlbr().astype(int)
            if i == 2 :
                bbox_2 = track.to_tlbr().astype(int)
            if i == 3 :
                bbox_3 = track.to_tlbr().astype(int)
            # track_id = track.track_id
            track_id = i #track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            action_war = ""
            clr_war = (0,0,0) #black
            clr = (0, 255, 0) #green
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32) # 30x13x3
                out = action_model.predict(pts, frame.shape[:2]) #384x383 (30x13x3) (action) 
                # t : inputs sequence (time steps).,
                # v : number of graph node (body parts).,
                # c : channel (x, y, score).,
                action_name = action_model.class_names[out[0].argmax()]
                action_label =out[0].argmax()
                if i == 1 :
                    if (action_label == 6):
                        if temp[i] == 2:
                            list_action_1.append(temp[i])
                        fall_down[i] = fall_down[i] + 1
                    if(action_label != temp[i]):
                        fall[i] = fall_down[i]
                        print(f'LamND: fall[{i}] : {fall[i]}')
                        if 10 <= fall[i] <= 40 :
                            if len(list_action_1) !=0 :
                                if action_label == 3 :
                                    list_action_1.append(temp[i])
                                    list_action_1.append(action_label)
                                    print("list_action :", list_action_1)
                                    if list_action_1 == [2,6,3]:
                                        print("Nguoi nay dang nam !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                        flag_1 = False
                                        list_action_1 = []
                                elif  (action_label == 0) | (action_label == 1) | (action_label == 2):
                                    flag_1 = False
                                    print("Vao fall 3")
                                else :
                                    list_action_1 = []
                                    flag_1 = True
                                    k[i]=0 
                            elif (action_label == 0) | (action_label == 1) | (action_label == 2):
                                flag_1 = False
                            else:
                                flag_1 = True 
                                k[i]=0 
                        fall_down[i] = 0
                    temp[i] = action_label
                    # print('LamND: Flag_1 : ', flag_1)
                
                if i == 2 :
                    if (action_label == 6):
                        if temp[i] == 2:
                            list_action_2.append(temp[i])
                        fall_down[i] = fall_down[i] + 1
                    if(action_label != temp[i]):
                        fall[i] = fall_down[i]
                        print(f'LamND: fall[{i}] : {fall[i]}')
                        if 10 <= fall[i] <= 50 :
                            if len(list_action_2) !=0 :
                                if action_label == 3 :
                                    list_action_2.append(temp[i])
                                    list_action_2.append(action_label)
                                    print("list_action :", list_action_2)
                                    if list_action_2 == [2,6,3]:
                                        print("Nguoi nay dang nam !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                        flag_2 = False
                                        list_action_2 = []
                                elif  (action_label == 0) | (action_label == 1) | (action_label == 2):
                                    flag_2 = False
                                else :
                                    list_action_2 = []
                                    flag_2 = True
                                    k[i]=0 
                            elif (action_label == 0) | (action_label == 1) | (action_label == 2):
                                flag_2 = False
                            else:
                                flag_2 = True 
                                k[i]=0 
                        fall_down[i] = 0
                    temp[i] = action_label
                    # print('LamND: Flag_2 : ', flag_2)
                
                if i == 3 :
                    if (action_label == 6):
                        if temp[i] == 2:
                            list_action_3.append(temp[i])
                        fall_down[i] = fall_down[i] + 1
                    if(action_label != temp[i]):
                        fall[i] = fall_down[i]
                        print(f'LamND: fall[{i}] : {fall[i]}')
                        # print('LamND: fall[i] : ', fall[i])
                        if 10 <= fall[i] <= 50 :
                            if len(list_action_3) !=0 :
                                if action_label == 3 :
                                    list_action_3.append(temp[i])
                                    list_action_3.append(action_label)
                                    print("list_action :", list_action_3)
                                    if list_action_3 == [2,6,3]:
                                        print("Nguoi nay dang nam !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                        flag_3 = False
                                        list_action_3 = []
                                elif  (action_label == 0) | (action_label == 1) | (action_label == 2):
                                    flag_3 = False
                                else :
                                    list_action_3 = []
                                    flag_3 = True
                                    k[i]=0 
                            elif (action_label == 0) | (action_label == 1) | (action_label == 2):
                                flag_3 = False
                            else:
                                flag_3 = True 
                                k[i]=0 
                        fall_down[i] = 0
                    temp[i] = action_label
                    # print('LamND: Flag_3 : ', flag_3)
                # action = '{}: {:.2f}%'.format(action_name, (out[0].max() * 100) + 20)
                action = '{}:'.format(action_name)
                # print('action_name : ',action_name)
                print("LamND : action : ", action,"\n")
                if action_name == 'Fall Down':
                    clr = (255, 0, 0) #red
                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)
            # print('LamND: bbox_1[0] : ', bbox_1[0])
            # print('LamND: bbox_2[0] : ', bbox_2[0])
            # print('LamND: bbox_3[0] : ', bbox_3[0])
            # VISUALIZE.
            if track.time_since_update == 0:
                if args.show_skeleton:
                    frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)
                # frame = cv2.putText(frame, action_war, (bbox[0] + 30, bbox[1] + 45), cv2.FONT_HERSHEY_COMPLEX,
                #                     0.4, clr_war, 1)
            
                if flag_1 == True :
                    frame = cv2.putText(frame, 'Co nguoi nga !!!', (bbox_1[0] - 5, bbox_1[1] - 10), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr_war, 1)
                    if k[i] == 40 :
                        flag_1 = False
                        k[i]=0
                if flag_2 == True :
                    frame = cv2.putText(frame, 'Co nguoi nga !!!', (bbox_2[0] - 5, bbox_2[1] - 10), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr_war, 1)
                    if k[i] == 40 :
                        flag_2 = False
                        k[i]=0
                if flag_3 == True :
                    frame = cv2.putText(frame, 'Co nguoi nga !!!', (bbox_3[0] - 5, bbox_3[1] - 10), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr_war, 1)
                    if k[i] == 40 :
                        flag_3 = False
                        k[i]=0
            k[i] = k[i] + 1               
        FPS = 1.0 / (time.time() - fps_time)
        print("FPS : ", FPS, '\n')
        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = frame[:, :, ::-1]
        fps_time = time.time()

        if outvid:
            writer.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clear resource.
    cam.stop()
    if outvid:
        writer.release()
    cv2.destroyAllWindows()
