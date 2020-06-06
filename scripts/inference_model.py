import os
import numpy as np
import time
import sys
import argparse
import cv2
sys.path.append("../")
from model_list import get_face_model_regular, get_face_model, get_face_landmarks_model, get_gaze_model, get_head_pose_model, obtain_models
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
    args.models = args.models.split(",")
    for ii, model in enumerate(model_classes):
        start_time = time.time()
        net = Network()
        model_path = model_paths[ii]
        if os.path.isfile(model_path):
            net.load_model(model_path, device='CPU', cpu_extension=CPU_EXTENSION)
        else:
            print(model_path, " does not exist")
            continue

        model_object = model(net)
        model_object.load_model(net)

        # Get and open video capture
        input_img = cv2.imread(args.image)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        end_time = time.time()
        load_time = end_time - start_time

        start_time = time.time()
        net_input_shape = net.get_input_shape()

        if args.models[ii] == "gaze_estimation":
            vector = np.random.randn(1,3)
            result = model_object.preprocess_input(frame, vector)
        else:
            p_frame = model_object.preprocess_input(frame)

        end_time = time.time()
        input_time = end_time - start_time

        args.iterations = int(args.iterations)

        start_time = time.time()
        for i in range(args.iterations):
            if args.models[ii] == "gaze_estimation":
                input_dict = dict(zip(list(result.keys()), list(result.values())))
                net.d_sync_inference(input_dict)
            else:
                net.sync_inference(p_frame)
        end_time = time.time()
        inference_time = end_time - start_time

        input_time_s.append(input_time)
        load_time_s.append(load_time)
        inference_time_s.append(inference_time)

    return args.models, input_time_s, load_time_s, inference_time_s

if __name__=='__main__':

    args = get_args()
    models, input_time_s, load_time_s, inference_time_s = infer(args)

    print("Precision: ", args.precision)
    for model, input_time, load_time, inference_time in zip(models, input_time_s, load_time_s, inference_time_s):
        print("Model...: ", model)
        print("Model Load Time is: ", load_time)
        print("Inference Time is: ", (inference_time) / args.iterations)
        print("Input/Output Time is: ", input_time)