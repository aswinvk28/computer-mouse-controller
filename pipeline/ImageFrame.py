from .Pipeline import Pipeline
import cv2
import sys
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ImageFrame(Pipeline):

    def __init__(self, model_type, model_class, objects, networks, objs, nets, logging=None):
        Pipeline.__init__(self, model_type, model_class, objects, networks, objs, nets, logging)
        self.idx = 0

    def exec_result_frame(self, obj, net, model_class, frames, request_id):
        self.exec_result(obj, net, model_class, frames, request_id)

    # preprocessing the model, executing the async inference
    def run(self, args, frames, model_classes):
        with ThreadPoolExecutor(max_workers=3) as executor:
            for counter in range(0,len(frames),args.batch_size):
                counter_array = [[counter]]
                counter_array = np.array(counter_array)
                counter_array = counter_array.flatten().tolist()
                for res in executor.map(self.exec_result_frame, [self.objects[self.idx]], 
                [self.networks[self.idx]],
                [model_classes.tolist()[self.idx]], 
                [frames[counter:counter+args.batch_size]], 
                counter_array):
                    pass

    # extract outputs for frame, exec bounding boxes fo frames
    def produce_outputs_frame(self, obj, net, model_class, frames, args, request_id):
        boxes_list = []
        gen_frames = []
        faces = []
        
        for ii, frame in enumerate(frames):
            outputs = obj.check_model(net, request_id=ii+request_id)
            output_name = obj.output_blob
            frame_copy = frame.copy()
            boxes, confs = model_class.preprocess_output(outputs, output_name, frame, confidence_level=args.conf)
            xmin, ymin, xmax, ymax = boxes[0]
            frame_copy, (xmin, ymin, xmax, ymax), frame[ymin:ymax,xmin:xmax]
            statuses = obj.wait(net, len(frames), request_id)
            if len(statuses) == len(frames):
                gen_frames.append(frame_copy)
                boxes_list.append(boxes[0])
                faces.append(frame[ymin:ymax,xmin:xmax])
        
        return gen_frames, faces, boxes_list

    def produce(self, args, frames, model_classes):

        gen_frames, faces, face_boxes = [], [], []

        # postprocessing the outputs, from face detection for all objects
        with ThreadPoolExecutor(max_workers=3) as executor:
            for counter in range(0,len(frames),args.produce_batch_size):
                counter_array = [[counter]]
                counter_array = np.array(counter_array)
                counter_array = counter_array.flatten().tolist()
                for ii, res in zip(list(range(1)),
                    executor.map(self.produce_outputs_frame, [self.objects[self.idx]], 
                    [self.networks[self.idx]], 
                    [model_classes.tolist()[self.idx]], 
                    [frames[counter:counter+args.produce_batch_size]], 
                    [args], 
                    counter_array)):
                    for f in res[0]:
                        gen_frames.append(f)
                    for f in res[1]:
                        faces.append(f)
                    for f in res[2]:
                        face_boxes.append(f)

        return gen_frames, faces, face_boxes