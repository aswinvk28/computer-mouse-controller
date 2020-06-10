from .Pipeline import Pipeline
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class Face(Pipeline):

    def __init__(self, model_type, model_class, objects, networks, objs, nets):
        self.model_type = model_type
        self.model_class = model_class
        self.objects = objects
        self.networks = networks
        self.objs = objs
        self.nets = nets
        self.idx = 1

    def exec_result_face(self, obj, net, model_class, frames, request_id):
        self.exec_result(obj, net, model_class, frames, request_id)

    # extract outputs for facial landmarks
    def produce_outputs_face(self, obj, net, model_class, frames, faces, face_boxes, args, 
    request_id):
        boxes_list, confs_list = [], []
        gen_frames = []
        cropped_left_eye = []
        cropped_right_eye = []
        left_eye, right_eye, nose, left_lip, right_lip = [], [], [], [], []
        const = 30
        output_name = obj.output_blob
        statuses = obj.wait(net, len(faces), request_id)
        if len(statuses) == len(frames):
            boxes_list, confs_list = [], []
            for ii, face in enumerate(faces):
                outputs = obj.check_model(net, request_id=ii+request_id)
                frame_copy = frames[ii].copy()
                boxes, _ = model_class.preprocess_output(outputs, output_name, face, confidence_level=args.conf)
                for ii, box in enumerate(boxes):
                    face_box = face_boxes[ii]
                    xmin, ymin, xmax, ymax = face_box
                    left_eye_x, left_eye_y, right_eye_x, right_eye_y, \
                    nose_x, nose_y, left_lip_x, left_lip_y, right_lip_x, right_lip_y = box
                    cropped_left_eye.append(
                        face[left_eye_y-const:left_eye_y+const,
                        left_eye_x-const:left_eye_x+const]
                    )
                    cropped_right_eye.append(
                        face[right_eye_y-const:right_eye_y+const,
                        right_eye_x-const:right_eye_x+const]
                    )
                    left_eye.append((left_eye_x, left_eye_y))
                    right_eye.append((right_eye_x, right_eye_y))
                    nose.append((nose_x, nose_y))
                    left_lip.append((left_lip_x, left_lip_y))
                    right_lip.append((right_lip_x, right_lip_y))
                gen_frames.append(frame_copy)
                
        return gen_frames, cropped_left_eye, cropped_right_eye, \
            left_eye, right_eye, nose, left_lip, right_lip

    def run(self, args, frames, faces, model_classes):
        # preprocessing the face and executing the landmarks detection
        with ThreadPoolExecutor(max_workers=4) as executor:
            for counter in range(0,len(frames),args.batch_size):
                counter_array = [[counter]]
                counter_array = np.array(counter_array)
                counter_array = counter_array.flatten().tolist()
                for res in executor.map(self.exec_result_face, 
                [self.objects[self.idx]], 
                [self.networks[self.idx]], 
                [model_classes.tolist()[self.idx]], 
                [faces[counter:counter+args.batch_size]], 
                counter_array):
                    pass

    def produce(self, args, frames, gen_frames, faces, face_boxes, 
    model_classes):
        
        batch_gen_frames, cropped_left, \
        cropped_right, left_eye, right_eye, \
        nose, left_lip, right_lip = [], [], [], [], [], [], [], []

        # postprocessing the outputs, from landmarks detection
        with ThreadPoolExecutor(max_workers=4) as executor:
            for counter in range(0,len(frames),args.produce_batch_size):
                counter_array = [[counter]]
                counter_array = np.array(counter_array)
                counter_array = counter_array.flatten().tolist()
                for ii, res in zip(list(range(1)),
                    executor.map(self.produce_outputs_face, [self.objects[self.idx]], 
                    [self.networks[self.idx]],
                    [model_classes.tolist()[self.idx]], 
                    [gen_frames[counter:counter+args.produce_batch_size]], 
                    [faces[counter:counter+args.produce_batch_size]], 
                    [face_boxes[counter:counter+args.produce_batch_size]], 
                    [args], 
                    counter_array)):
                    for f in res[0]:
                        batch_gen_frames.append(f)
                    for f in res[1]:
                        cropped_left.append(f)
                    for f in res[2]:
                        cropped_right.append(f)
                    for f in res[3]:
                        left_eye.append(f)
                    for f in res[4]:
                        right_eye.append(f)
                    for f in res[5]:
                        nose.append(f)
                    for f in res[6]:
                        left_lip.append(f)
                    for f in res[7]:
                        right_lip.append(f)

        return batch_gen_frames, cropped_left, \
        cropped_right, left_eye, right_eye, \
        nose, left_lip, right_lip