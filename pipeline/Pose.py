from .Pipeline import Pipeline
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class Pose(Pipeline):

    def __init__(self, model_type, model_class, objects, networks, objs, nets, logging=None):
        Pipeline.__init__(self, model_type, model_class, objects, networks, objs, nets, logging)
        self.idx = 2

    def exec_result_pose(self, obj, net, model_class, faces, request_id):
        self.exec_result(obj, net, model_class, faces, request_id)

    # extract outputs for pose
    def produce_outputs_pose(self, obj, net, model_class, frames, args, request_id):
        pose_angles_list = []
        output_name = obj.output_blob
        statuses = obj.wait(net, len(frames), request_id)
        if len(statuses) == len(frames):
            for ii, frame in enumerate(frames):
                outputs = obj.check_model(net, request_id=ii+request_id)
                matrix, _ = model_class.preprocess_output(outputs, output_name, frame, confidence_level=args.conf)
                pose_angles_list.append(matrix)
        return pose_angles_list

    def run(self, args, frames, faces, model_classes):
        # preprocessing the face and executing the landmarks detection
        with ThreadPoolExecutor(max_workers=2) as executor:
            for counter in range(0,len(frames),args.batch_size):
                counter_array = [counter]
                counter_array = np.array(counter_array)
                counter_array = counter_array.flatten().tolist()
                for res in executor.map(self.exec_result_pose, [self.objects[self.idx]], 
                [self.networks[self.idx]], 
                [model_classes.tolist()[self.idx]], 
                [faces[counter:counter+args.batch_size]], 
                counter_array):
                    pass

    def produce(self, args, frames, batch_gen_frames, model_classes):
        
        head_pose_angles = []
        
        # postprocessing the outputs, from landmarks detection
        with ThreadPoolExecutor(max_workers=2) as executor:
            for counter in range(0,len(frames),args.produce_batch_size):
                counter_array = [counter]
                counter_array = np.array(counter_array)
                counter_array = counter_array.flatten().tolist()
                for ii, res in zip(list(range(1)),
                    executor.map(self.produce_outputs_pose, [self.objects[self.idx]], 
                    [self.networks[self.idx]],
                    [model_classes.tolist()[self.idx]], 
                    [batch_gen_frames[counter:counter+args.produce_batch_size]], 
                    [args], 
                    counter_array)):
                    for f in res:
                        head_pose_angles.append(f)

        return head_pose_angles