import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin, IECore
import sys
import os
sys.path.append("../")
from model_list import get_face_model_regular, get_face_model, get_face_landmarks_model, get_gaze_model, get_head_pose_model, obtain_models

import pprint
import argparse
import sys
import cv2
import os

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def main(args, model_name):
    pp = pprint.PrettyPrinter(indent=4)
    model=args.model
    device=args.device
    image_path=args.image

    # Loading model
    model_weights=model+'.bin'
    model_structure=model+'.xml'
    
    # TODO: Load the model
    plugin = IECore()

    if CPU_EXTENSION and "CPU" in device:
        plugin.add_extension(CPU_EXTENSION, device)
    
    model = IENetwork(model_structure, model_weights)
    net = plugin.load_network(model, device, num_requests=1)

    input_name=next(iter(model.inputs))

    # Reading and Preprocessing Image
    input_img=cv2.imread(image_path)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    
    input_blob = next(iter(model.inputs))
    net_input_shape = model.inputs[input_blob].shape

    if model_name == "gaze_estimation":
        p_frame = np.random.randn(1,3)
    else:
        p_frame = preprocess_input(input_img, net_input_shape)

    # TODO: Run Inference and print the layerwise performance
    net.requests[0].infer({input_name: p_frame})
    pp.pprint(net.requests[0].get_perf_counts())

def preprocess_input(image, net_input_shape):
    '''
    Before feeding the data into the model for inference,
    you might have to preprocess it. This function is where you can do that.
    '''
    p_frame = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape(1, *p_frame.shape)

    return p_frame
    
def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="", required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--precision', default='FP16')
    parser.add_argument('--image', default=None)
    parser.add_argument('--threshold', default=0.6, type=float)
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = get_args()
    model_name = args.model
    prefix = "/home/workspace/ir_models/intel/"
    if args.model == "face":
        args.model = os.path.splitext(os.path.join(prefix, get_face_model(args.precision)))[0]
    elif args.model == "face_regular":
        args.model = os.path.splitext(os.path.join(prefix, get_face_model_regular(args.precision)))[0]
    elif args.model == "gaze_estimation":
        args.model = os.path.splitext(os.path.join(prefix, get_gaze_model(args.precision)))[0]
    elif args.model == "face_landmarks":
        args.model = os.path.splitext(os.path.join(prefix, get_face_landmarks_model(args.precision)))[0]
    elif args.model == "head_pose":
        args.model = os.path.splitext(os.path.join(prefix, get_head_pose_model(args.precision)))[0]

    if not os.path.isfile(args.model + ".xml"):
        print(args.model)
        raise Exception("The file does not exist")

    main(args, model_name)
