import cv2
import sys
import os
from concurrent.futures import ThreadPoolExecutor
sys.path.append("../")
from scripts.Network import Network

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

class Pipeline:

    def __init__(self, model_type, model_class):
        self.model_type = model_type
        self.model_class = model_class
        self.objects = {}
        self.networks = {}
        self.objs = {}
        self.nets = {}

    def load_network(self, args, model_class, model_type, model_path):
        if not os.path.isfile(model_path):
            print("Model path does not exist: ", model_path)
            return None, None
        net = Network()
        net.load_model(model_path, device=args.device, cpu_extension=CPU_EXTENSION, args=args)
        c = model_class
        m = c(net)
        m.load_model(net)
        return m, net, model_type

    def exec_result(self, obj, net, model_class, frames, request_id):
        p_frames = [obj.preprocess_input(frame) for frame in frames]
        obj.predict(net, p_frames, request_id=request_id)

    def exec_result_multi(self, obj, net, model_class, faces, cropped_left, 
    cropped_right, head_pose, request_id):
        p_frames = []
        for i,l,r,p in zip(list(range(len(cropped_left))), cropped_left, cropped_right, head_pose):
            ro = obj.preprocess_input_gaze(left=l, right=r, vector=p)
            if ro is not None:
                p_frames.append(ro)
        obj.predict_dict(net, p_frames, request_id=request_id)

    # load network by existence of model file path
    def run(self, model_classes, models_array, model_paths, args):
        with ThreadPoolExecutor(max_workers=4) as executor:
            for ii, res in zip(list(range(len(model_classes))),
            executor.map(self.load_network, [args]*len(model_classes), 
            model_classes.tolist(), models_array.tolist(), 
            model_paths)):
                if res[0] is not None:
                    self.objects[ii] = res[0]
                    self.networks[ii] = res[1]
                    self.objs[res[2]] = res[0]
                    self.nets[res[2]] = res[1]

    def draw_boxes(self, frame_copy, face_box):
        xmin, ymin, xmax, ymax = face_box
        cv2.rectangle(frame_copy, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

    def draw_markers(self, frame_copy, face_box, 
    left_eye_x, left_eye_y, right_eye_x, right_eye_y, 
    nose_x, nose_y, left_lip_x, left_lip_y, 
    right_lip_x, right_lip_y, color=(0,255,0), 
    thickness=5, radius=5):
        xmin, ymin, xmax, ymax = face_box
        cv2.circle(frame_copy, (xmin+left_eye_x, ymin+left_eye_y), radius, color, thickness)
        cv2.circle(frame_copy, (xmin+right_eye_x, ymin+right_eye_y), radius, color, thickness)
        cv2.circle(frame_copy, (xmin+nose_x, ymin+nose_y), radius, color, thickness)
        cv2.circle(frame_copy, (xmin+left_lip_x, ymin+left_lip_y), radius, color, thickness)
        cv2.circle(frame_copy, (xmin+right_lip_x, ymin+right_lip_y), radius, color, thickness)

    def finalize_pipeline(self, out, frame, batch_gen_frames, face_boxes, 
    left_eye, right_eye, nose, left_lip, right_lip, 
    gaze_vector):
        try:
            for ii, frame in enumerate(batch_gen_frames):
                xmin, ymin, xmax, ymax = face_boxes[ii]
                left_eye_x, left_eye_y = left_eye[ii]
                right_eye_x, right_eye_y = right_eye[ii]
                nose_x, nose_y = nose[ii]
                left_lip_x, left_lip_y = left_lip[ii]
                right_lip_x, right_lip_y = right_lip[ii]
                self.draw_markers(frame, face_boxes[ii], left_eye_x, left_eye_y, right_eye_x, right_eye_y, 
                nose_x, nose_y, left_lip_x, left_lip_y, right_lip_x, right_lip_y)
                self.draw_boxes(frame, face_boxes[ii])
                if ii < len(gaze_vector):
                    cv2.putText(frame, str(gaze_vector[ii]), 
                    (15, 40), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
                out.write(frame)
        except Exception as e:
            raise e