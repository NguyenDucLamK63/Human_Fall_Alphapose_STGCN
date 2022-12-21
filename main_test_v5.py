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

### Detect V5
import sys
sys.path.insert(0, './yolov5')
import torch.backends.cudnn as cudnn
from pathlib import Path
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh,xywh2xyxy, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

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
    # par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
    #                     help='Source of camera or video file path.')
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
    # par.add_argument('--device', type=str, default='cuda',
    #                     help='Device to run model on cpu or cuda.')
    par.add_argument('--yolo_model', nargs='+', type=str, default='yolov5l.pt', help='model.pt path(s)')
    # par.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    par.add_argument('--source', type=str, default='/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_ver3_yolov5/Fall_1_persion_1_qt.mp4', help='source')  # file/folder, 0 for webcam
    par.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    par.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[384], help='inference size h,w')
    par.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    par.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    par.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    par.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    par.add_argument('--show-vid', action='store_false', help='display tracking video results')
    par.add_argument('--save-vid', action='store_true', help='save video tracking results')
    par.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    par.add_argument('--classes', default='0',nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    par.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    par.add_argument('--augment', action='store_true', help='augmented inference')
    par.add_argument('--evaluate', action='store_true', help='augmented inference')
    par.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    par.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    par.add_argument('--visualize', action='store_true', help='visualize features')
    par.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    par.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    par.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    par.add_argument('--name', default='exp', help='save results to project/name')
    par.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    args = par.parse_args()
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand

    device = args.device

    # DETECTION MODEL.
    inp_dets = args.detection_input_size #384
    # khởi tạo device và image size và khởi tạo mạng darNet cho YoloV3
    # detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    #YoloV5
    out, source, yolo_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        args.output, args.source, args.yolo_model, args.show_vid, args.save_vid, \
        args.save_txt, args.imgsz, args.evaluate, args.half, args.project, args.name, args.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(args.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=args.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit) #luong xu ly video
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

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

    cam_source = args.source
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

    #YoloV5
    f = 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset): #doc video, trich xuat ra cac frame xu ly va dua vao yoloV5
        f = f+1
        print("frame : ", f)
        frame = cam.getitem()
        img_raw = img.copy()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=args.augment) #Thuc hien detect, tim vat theo trong anh
        # Apply NMS, xu ly bounding box
        # preds = non_max_suppression(pred, args.conf_thres, args.iou_thres, args.classes, args.agnostic_nms, max_det=args.max_det)
        preds = non_max_suppression(pred)
        for i, detected in enumerate(preds):  # detections per image
            detected = detected.cpu()
            print(detected)
            tracker.predict()
            # Merge two source of predicted bbox together.
            for track in tracker.tracks: #[]
                det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
                det = torch.cat([det[:, :5], det[:, 6:]], dim=1)
                detected = torch.cat([detected, det], dim=0) if detected is not None else det
            print(detected.numel())
            detections = []  # List of Detections object for tracking., Lưu lại các bbox và keypoint (13x3)
            print("detected[:, 0:4] :" , detected[:, 0:4])
            if detected.numel() != 0:
                # detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
                # Predict skeleton pose of each bboxs.
                poses = pose_model.predict(frame, detected[:,0:4], detected[:, 4]) # 13x2 (float32)
                # print("detected[:, 4] :", detected[:, 4])
                # for i in range(0, 10):
                #     print("xxx: ", detected[:, 0:i])

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
                    # print("action_name :", action_name)
                    if i == 1 :
                        if (action_label == 6):
                            if temp[i] == 2:
                                list_action_1.append(temp[i])
                            fall_down[i] = fall_down[i] + 1
                        if(action_label != temp[i]):
                            fall[i] = fall_down[i]
                            # print(f'LamND: fall[{i}] : {fall[i]}')
                            if 10 <= fall[i] <= 40 :
                                if len(list_action_1) !=0 :
                                    if action_label == 3 :
                                        list_action_1.append(temp[i])
                                        list_action_1.append(action_label)
                                        # print("list_action :", list_action_1)
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
                    # print("LamND : action : ", action,"\n")
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
