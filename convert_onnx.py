# timf ddaauf vao dau ra : 13112112 
import torch
from DetectorLoader import TinyYOLOv3_onecls
from Convert_onnx.convert_onnx import convert_ONNX_yolov3 , convert_ONNX_alphapose , convert_ONNX_STGCN
from Detection.Models import Darknet
from SPPE.src.main_fast_inference import InferenNet_fast, InferenNet_fastRes50,InferenNet_fastRes50_onnx
from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph
from torch.autograd import Variable
def main():
    # convert to onnx
    # Let's load the model we just created and test the accuracy per label 
    # inp_dets = 384
    # device='cuda'
    # config_file = 'Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg'
    # weight_file = 'Models/yolo-tiny-onecls/best-model.pth'
    # detect_model = Darknet(config_file)
    # # load_state_dict(detect_model, torch.load(weight_file))
    # detect_model.load_state_dict(torch.load(weight_file))
    # detect_model.eval()
    # print("doneeeee")
    # # Conversion to ONNX 
    # convert_ONNX_yolov3(detect_model) 

    # Pose 
    # inp_dets = 384
    # device='cuda'
    # config_file = 'Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg'
    # weight_file = 'Models/yolo-tiny-onecls/best-model.pth'
    # pose_model = InferenNet_fastRes50_onnx()
    # # load_state_dict(detect_model, torch.load(weight_file))
    # pose_model.eval()
    # print("doneeeee")
    # # Conversion to ONNX 
    # convert_ONNX_STGCN(pose_model)

    inp_dets = 384
    device= 'cuda'
    weight_file = '/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_2/Actionsrecognition/saved/chon_TSSTG_Mix_FDD_UR_100_32_0.01_percents_2_test_full_0.1_0.1/tsstg-model.pth'
    graph_args = {'strategy': 'spatial'}
    class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
                            'Stand up','Sit down','Fall Down', ]
    num_class = len(class_names)
    action_model = TwoStreamSpatialTemporalGraph(graph_args, num_class)
    dummy_input_1 = torch.randn(1 , 3, 30 , 14 )
    dummy_input_2 = torch.randn(1 , 2, 29 , 14 )
    dummy_input = [dummy_input_1, dummy_input_2]
    action_model.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
    example_output = action_model(dummy_input)
    print(example_output)
    action_model.eval()
    print("doneeeee")
    convert_ONNX_STGCN(action_model)
    # torch.onnx.export(action_model,         # model being run 
    #     args= dummy_input,       # model input (or a tuple for multiple inputs) 
    #     f = "/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_2/Models/onnx/STGCN_tsstg_1.onnx",       # where to save the model  
    #     # verbose=False,
    #     export_params=True,  # store the trained parameter weights inside the model file 
    #     opset_version=12,    # the ONNX version to export the model to 
    #     operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
    #     # do_constant_folding=True,  # whether to execute constant folding for optimization 
    #     input_names = ['input_action_1','input_action_2'],   # the model's input names 
    #     output_names = ['output_action'], # the model's output names 
    #     dynamic_axes={'input_action_1' : {1 : 'seq_len'}, "input_action_2": {3: "seq_len"},     # variable length axes 
    #                         'output_action' : {0 : 'batch_size'}}) 
    print("doneeeee")
    #")
    # Conversion to ONNX 
    # convert_ONNX_STGCN(action_model)
main()