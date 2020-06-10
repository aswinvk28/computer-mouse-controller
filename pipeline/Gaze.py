from .Pipeline import Pipeline
import cv2
from concurrent.futures import ThreadPoolExecutor

class Gaze(Pipeline):

    def __init__(self, model_type, model_class, objects, networks, objs, nets):
        self.model_type = model_type
        self.model_class = model_class
        self.objects = objects
        self.networks = networks
        self.objs = objs
        self.nets = nets
        self.idx = 3

    def exec_result_gaze(self, obj, net, model_class, faces, cropped_left, 
    cropped_right, head_pose, request_id):
        self.exec_result_multi(obj, net, model_class, faces, cropped_left, 
    cropped_right, head_pose, request_id)

    # exec outputs for gaze
    def produce_outputs_gaze(self, obj, net, model_class, frames, request_id):
        outputs_list = []
        statuses = obj.wait(net, len(frames), request_id)
        if len(statuses) == len(frames):
            outputs = obj.check_model(net, request_id=request_id)
            outputs_list.append(outputs)
                
        return outputs_list

    def run(self, args, frames, faces, cropped_left, 
    cropped_right, head_pose_angles, model_classes):
        # preprocessing the gaze and executing the landmarks detection
        with ThreadPoolExecutor(max_workers=4) as executor:
            for counter in range(0,len(frames),args.batch_size):
                try:
                    for ii in executor.map(range(1)):
                        self.exec_result_gaze(self.objs['gaze_estimation'], 
                        self.nets['gaze_estimation'], 
                        model_classes.tolist()[self.idx], 
                        faces[counter:counter+args.batch_size], 
                        cropped_left[counter:counter+args.batch_size], 
                        cropped_right[counter:counter+args.batch_size], 
                        head_pose_angles[counter:counter+args.batch_size], counter)
                except Exception as e:
                    print(e.args)

    def produce(self, args, frames, batch_gen_frames,
    model_classes):
        gaze_vector = []
        # post process gaze vector
        for counter in range(0,len(frames),args.produce_batch_size):
            res = self.produce_outputs_gaze(self.objs['gaze_estimation'], 
            self.nets['gaze_estimation'], model_classes.tolist()[self.idx], 
            batch_gen_frames[counter:counter+args.produce_batch_size], counter)
            for f in res[0]:
                gaze_vector.append(f)

        return gaze_vector