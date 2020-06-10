from .Pipeline import Pipeline
import cv2
from concurrent.futures import ThreadPoolExecutor

class Gaze(Pipeline):

    def __init__(self, model_type, model_class, objects, networks, objs, nets, logging=None):
        Pipeline.__init__(self, model_type, model_class, objects, networks, objs, nets, logging)
        self.idx = 3

    def exec_result_gaze(self, obj, net, model_class, faces, cropped_left, 
    cropped_right, head_pose, request_id):
        self.exec_result_multi(obj, net, model_class, faces, cropped_left, 
    cropped_right, head_pose, request_id)

    # exec outputs for gaze
    def produce_outputs_gaze(self, args, obj, net, model_class, frames, request_id):
        statuses = obj.wait(net, len(frames), request_id)
        outputs = []
        if len(statuses) == len(frames):
            for ii in range(len(frames)):
                outputs.append(obj.check_model(net, request_id=ii+request_id))
            return outputs

    def run(self, args, frames, faces, cropped_left, 
    cropped_right, head_pose_angles, model_classes, cntr=None):
        # preprocessing the gaze and executing the landmarks detection
        with ThreadPoolExecutor(max_workers=3) as executor:
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
                    raise e

    def produce(self, args, frames, batch_gen_frames,
    model_classes, cntr=None):
        gaze_vector = []
        # post process gaze vector
        for counter in range(0,len(frames),args.produce_batch_size):
            res = self.produce_outputs_gaze(args, self.objs['gaze_estimation'], 
            self.nets['gaze_estimation'], model_classes.tolist()[self.idx], 
            batch_gen_frames[counter:counter+args.produce_batch_size], counter)
            if res is not None:
                for f in res:
                    for g in f['gaze_vector']:
                        gaze_vector.append(g)

        return gaze_vector