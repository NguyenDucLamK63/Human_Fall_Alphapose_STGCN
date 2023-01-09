# timf ddaauf vao dau ra : 13112112 
import torch
from DetectorLoader import TinyYOLOv3_onecls
from Convert_onnx.convert_onnx import convert_ONNX_1output, convert_ONNX_2output, convert_ONNX_yolov3, load_state_dict
from Detection.Models import Darknet


def main():
    # convert to onnx
    # Let's load the model we just created and test the accuracy per label 
    inp_dets = 384
    device = 0
    config_file = 'Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg'
    weight_file = 'Models/yolo-tiny-onecls/best-model.pth'
    detect_model = Darknet(config_file)
    load_state_dict(detect_model, torch.load(weight_file))
    # model = ResNet(num_layers=100, feature_dim=512)
    # path_detect = '/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_2/Models/yolo-tiny-onecls/best-model.pth'
    # load_state_dict(detect_model, torch.load(path_detect))
    detect_model.eval()
    print("doneeeee")
 
    # Conversion to ONNX 
    convert_ONNX_yolov3(detect_model) 

    # # convert Quality model to onnx
    # model_quality = FaceQuality(512 * 7 * 7)
    # path_quality = './weights/quality.pth'
    # load_state_dict(model_quality, torch.load(path_quality))
    # model_quality.eval()
    # convert_ONNX_1output(model_quality)

main()