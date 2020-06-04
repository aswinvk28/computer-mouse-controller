import os
import numpy as np
import time
import sys
import argparse
import cv2
sys.path.append("../")
from model_list import get_face_model, get_face_landmarks_model, get_gaze_model, get_head_pose_model, obtain_models
from src.input_feeder import InputFeeder
from inference import Network

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="", required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--image', default=None)
    parser.add_argument('--iterations', default=None)
    parser.add_argument('--precision', default=None)
    parser.add_argument('--create_models', default=False)
    parser.add_argument('--threshold', default=0.6, type=float)
    
    args = parser.parse_args()

    return args

def infer(args):
    # Convert the args for color and confidence
    ### TODO: Initialize the Inference Engine
    model_paths, model_classes = obtain_models(args)
    input_feeder = InputFeeder(input_type='image', input_file=args.image)
    input_feeder.load_data()
    input_time_s, load_time_s, inference_time_s = [], [], []
    frame = input_feeder.cap
    for ii, model in enumerate(model_classes):
        start_time = time.time()
        net = Network()
        model_path = model_paths[ii]
        net.load_model(model_path, device='CPU', cpu_extension=CPU_EXTENSION)

        # Get and open video capture
        input_img = cv2.imread(args.image)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        end_time = time.time()
        load_time = end_time - start_time

        start_time = time.time()
        net_input_shape = net.get_input_shape()

        p_frame = preprocessing(input_img, net_input_shape)
        end_time = time.time()
        input_time = end_time - start_time

        args.iterations = int(args.iterations)

        start_time = time.time()
        for i in range(args.iterations):
            net.sync_inference(p_frame)
        end_time = time.time()
        inference_time = end_time - start_time

        input_time_s.append(input_time)
        load_time_s.append(load_time)
        inference_time_s.append(inference_time)

    return input_time_s, load_time_s, inference_time_s

def preprocessing(frame, net_input_shape, request_id=0):
    # TODO: Using the input image, run inference on the model for 10 iterations
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape(1, *p_frame.shape)

    return p_frame

if __name__=='__main__':

    args = get_args()
    input_time_s, load_time_s, inference_time_s = infer(args)

    for input_time, load_time, inference_time in zip(input_time_s, load_time_s, inference_time_s):
        print("Model...")
        print("Model Load Time is: ", load_time)
        print("Inference Time is: ", (inference_time) / args.iterations)
        print("Input/Output Time is: ", input_time)