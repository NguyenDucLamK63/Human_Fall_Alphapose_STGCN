from __future__ import division
import torch.onnx 
from torch.autograd import Variable
import onnxruntime
import cv2
import numpy as np


# load weight aldready
def load_state_dict(model, state_dict):
    all_keys = {k for k in state_dict.keys()}
    for k in all_keys:
        if k.startswith('module.'):
            state_dict[k[7:]] = state_dict.pop(k)
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    if len(pretrained_dict) == len(model_dict):
        print("all params loaded")
    else:
        not_loaded_keys = {k for k in pretrained_dict.keys() if k not in model_dict.keys()}
        print("not loaded keys:", not_loaded_keys)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

#Function to Convert to ONNX with 2 output
def convert_ONNX_2output(model): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = Variable(torch.randn(1, 3, 112, 112))

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "./onnx/Resnet2F.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input'],   # the model's input names 
         output_names = ['output_0', 'output_1'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

#Function to Convert to ONNX with 1 output
def convert_ONNX_1output(model): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = Variable(torch.randn(1, 25088))

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "./onnx/quality.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input'],   # the model's input names 
         output_names = ['output'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

def convert_ONNX_adaFace(model): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = Variable(torch.randn(1, 3, 112, 112))

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "./onnx/adaface_50w_2op.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input'],   # the model's input names 
         output_names = ['output'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')

def convert_ONNX_yolov3(model): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = Variable(torch.randn(1, 3, 384, 384))

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_2/Models/onnx/yolov3_tiny_use.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=11,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input_detect'],   # the model's input names 
         output_names = ['output_detect']) # the model's output names 
        #  dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
        #                         'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')

# def convert_ONNX_alphapose(model): 

#     # set the model to inference mode 
#     model.eval() 

#     # Let's create a dummy input tensor  
#     # seq_len = torch.randint(1, 20, (1,)).item()
#     # x = torch.randn(batch, model.input_channels, model.input_features, seq_len)
#     dummy_input_1 = Variable(torch.randn(1, 3, 224, 160))
#     dummy_input_2 = Variable(torch.randn(2, 3, 224, 160))
#     dummy_input_3 = Variable(torch.randn(3, 3, 224, 160))
#     # dummy_input_4 = Variable(torch.randn(4, 3, 224, 160))
#     # dummy_input_5 = Variable(torch.randn(5, 3, 224, 160))
#     dynamic_axes = {
#         "input_pose_1": {0: "batch"},
#         "input_pose_2": {0: "batch"},
#         "input_pose_3": {0: "batch"},
#         "output_pose_1": {0: "batch"},
#         "output_pose_2": {0: "batch"},
#         "output_pose_3": {0: "batch"},
#     }

#     # Export the model   
#     torch.onnx.export(model,         # model being run 
#          args= (dummy_input_1 , dummy_input_2,dummy_input_3),       # model input (or a tuple for multiple inputs) 
#          f = "/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_2/Models/onnx/alphapose_res50_multi.onnx",       # where to save the model  
#          export_params=True,  # store the trained parameter weights inside the model file 
#          opset_version=11,    # the ONNX version to export the model to 
#          do_constant_folding=True,  # whether to execute constant folding for optimization 
#          input_names = ['input_pose_1','input_pose_2','input_pose_3'],   # the model's input names 
#          output_names = ['output_pose_1','output_pose_2','output_pose_3' ], # the model's output names 
#          dynamic_axes= dynamic_axes)
#     print(" ") 
#     print('Model has been converted to ONNX')
def convert_ONNX_alphapose(model): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    # seq_len = torch.randint(1, 20, (1,)).item()
    # x = torch.randn(batch, model.input_channels, model.input_features, seq_len)
    dummy_input_1 = Variable(torch.randn(1 , 3, 224, 160 , requires_grad=True))
    # dummy_input_4 = Variable(torch.randn(4, 3, 224, 160))
    # dummy_input_5 = Variable(torch.randn(5, 3, 224, 160))

    # Export the model   
    torch.onnx.export(model,         # model being run 
         args= dummy_input_1 ,       # model input (or a tuple for multiple inputs) 
         f = "/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_2/Models/onnx/alphapose_res50_multi_requires_dynamic.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=11,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input_pose_1'],   # the model's input names 
         output_names = ['output_pose_1'], # the model's output names 
         dynamic_axes={'input_pose_1' : {0 : 'seq_len'},    # variable length axes 
                                'output_pose_1' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')

def convert_ONNX_STGCN(model): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    # seq_len = torch.randint(1, 20, (1,)).item()
    # x = torch.randn(batch, model.input_channels, model.input_features, seq_len)
    dummy_input_1 = Variable(torch.randn(1 , 3, 30 , 14 , requires_grad=True))
    dummy_input_2 = Variable(torch.randn(1 , 2, 29 , 14 , requires_grad=True))
    dummy_input = [dummy_input_1, dummy_input_2]
    # dummy_input_4 = Variable(torch.randn(4, 3, 224, 160))
    # dummy_input_5 = Variable(torch.randn(5, 3, 224, 160))

    # Export the model   
    torch.onnx.export(model,         # model being run 
         args= dummy_input,       # model input (or a tuple for multiple inputs) 
         f = "/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_2/Models/onnx/STGCN_tsstg_0.1_0.1.onnx",       # where to save the model  
         verbose=False,
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=12,    # the ONNX version to export the model to 
         operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input_action_1','input_action_2'],   # the model's input names 
         output_names = ['output_action'], # the model's output names 
         dynamic_axes={'input_action_1' : {1 : 'seq_len'}, "input_action_2": {3: "seq_len"},     # variable length axes 
                                'output_action' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')