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
source = '/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_2/lam_fall_cty_6_12_1_persion.mp4'
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
    par.add_argument('--pose_backbone', type=str, default='resnet101',
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
    fall_down = 0
    Lying_Down = 0
    while cam.grabbed():
        f += 1
        #Lấy frame trong queue
        frame = cam.getitem()
        image = frame.copy()
        print("Frame : ",f , '\n')
        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10) #384x384
        # print("type: ", type(detected))

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            # print("/////////////////////////////////////////////////////////////////////////////////////////")
            # print("track", track)
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.
        # print("detected[:, 0:4] :" , detected[:, 0:4])
        if detected is not None:
            # print("True")
            # detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4]) # 13x2 (float32)
            # print("detected[:, 4] :", detected[:, 4])
            # for i in range(0, 10):
            #     print("xxx: ", detected[:, 0:i])

            # Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]
            # print("detections", detections)

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

            track_id = track.track_id
            # print("track_id : ", track_id)
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0) #green
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32) # 30x13x3
                out = action_model.predict(pts, frame.shape[:2]) #384x383 (30x13x3) (action) 
                # t : inputs sequence (time steps).,
                # v : number of graph node (body parts).,
                # c : channel (x, y, score).,
                # print("out :", out , '\n')
                action_name = action_model.class_names[out[0].argmax()]
                # print("Action_name raw :" , action_name)
                action_label =out[0].argmax()
                if (action_label == 6):
                    fall_down = fall_down + 1
                    print("fall_down = ",fall_down)
                if (action_label == 3):
                    Lying_Down = Lying_Down + 1
                    print("Lying Down = ",Lying_Down)
                # if (action_label != 6):
                #     fall_down = 0
                # if (action_label != 3):
                #     Lying_Down = 0
                    # action_name = "Fall Down"
                # if  1<=fall_down & Lying_Down >=1 :
                #     action_name = "Fall Down"
                #     clr = (255, 0, 0) #red
                    # print("fall_downfall_downfall_downfall_downfall_downfall_downfall_downfall_downfall_down")
                    # action = '{}:'.format(action_name)
                # if  15<=fall_down & Lying_Down >=1 :
                #     action_name = "Lying Down"
                #     clr = (255, 200, 0)
                    # print("Lying DownLying DownLying DownLying DownLying DownLying DownLying DownLying DownLying Down")
                    # action = '{}:'.format(action_name)
                # action = '{}: {:.2f}%'.format(action_name, (out[0].max() * 100) + 20)
                action = '{}:'.format(action_name)
                # print('action_name : ',action_name)
                # print("action : ", action,"\n")
                if action_name == 'Fall Down':
                    clr = (255, 0, 0) #red
                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)

            # VISUALIZE.
            if track.time_since_update == 0:
                if args.show_skeleton:
                    frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)
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
