# from tool import darknet2pytorch
from Detection.Models import Darknet
import torch

# load weights from darknet format
model = Darknet('/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_2/Models/yolo-tiny-onecls/yolov3.cfg')
model.load_darknet_weights('/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_2/Models/yolo-tiny-onecls/yolov3.weights')

# save weights to pytorch format
torch.save(model.state_dict(), '/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_2/Models/yolo-tiny-onecls/yolov3.pth')

# # reload weights from pytorch format
# model_pt = Darknet('path/to/cfg/yolov4-416.cfg', inference=True)
# model_pt.load_state_dict(torch.load('path/to/save/yolov4-pytorch.pth'))