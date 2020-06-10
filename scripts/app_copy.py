import argparse
import cv2
from Network import Network
import numpy as np
import os
import time

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--image', default=None)
    parser.add_argument('--iterations', default=None)
    parser.add_argument('--precision', default=None)
    
    args = parser.parse_args()

    return args

def draw_boxes(frame, boxes, result, args, width, height, y_pixel):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        xmin = int(box[3] * width)
        ymin = int(box[4] * height)
        xmax = int(box[5] * width)
        ymax = int(box[6] * height)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, args.th)
    return frame

def preprocess_outputs(frame, args, result, confidence_level=0.65):
        '''
        TODO: This method needs to be completed by you
        '''
        boxes = []
        confs = []
        height, width = frame.shape[:2]
        if len(result[0][0]) > 0:
            for res in result[0][0]:
                _, label, conf, xmin, ymin, xmax, ymax = res
                if conf > confidence_level:
                    xmin = int(xmin*width)
                    ymin = int(ymin*height)
                    xmax = int(xmax*width)
                    ymax = int(ymax*height)
                    boxes.append([xmin, ymin, xmax, ymax])
                    confs.append(conf)
        
        return boxes, confs, frame

def infer_on_video(args):
    # Convert the args for color and confidence
    ### TODO: Initialize the Inference Engine
    plugin = Network()

    model_weights = args.model+'.bin'
    model_structure = args.model+'.xml'

    ### TODO: Load the network model into the IE
    plugin.load_model(model_structure, args.device, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()

    # Get and open video capture
    input_img = cv2.imread(args.image)

    # Process frames until the video ends, or process is exited
    # Read the next frame
    ### TODO: Pre-process the frame
    p_frame = cv2.resize(input_img, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape(1, *p_frame.shape)

    start_time = time.time()
    for i in range(args.iterations):
        ### TODO: Perform inference on the frame
        plugin.async_inference(p_frame)

        ### TODO: Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()

    end_time = time.time()

    print("Model Load Time is: ", end_time - start_time)
    print("Inference Time is: ", (end_time - start_time) / args.iterations)
    # print("Input/Output Time is: ", (end_time - start_time) / args.iterations)


def main():
    args = get_args()
    args.iterations = int(args.iterations)
    args.model = os.path.join(args.prefix, args.model, args.precision, args.model)
    infer_on_video(args)


if __name__ == "__main__":
    main()
