import argparse
from concurrent.futures import ThreadPoolExecutor
from src.input_feeder import InputFeeder
import threading
import cv2
import os
import numpy as np
import time
from scripts.Network import Network
from model_list import get_face_model_regular, get_face_model, get_face_landmarks_model, get_gaze_model, get_head_pose_model, obtain_models

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser()
    m_desc = "The list of tagged models: face_regular,head_pose,facial_landmarks,gaze_estimation"
    parser.add_argument("--models", help=m_desc, required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--precision', default='FP16')
    parser.add_argument('--conf', default=0.6, type=float)
    parser.add_argument('--prefix', default="/home/workspace/ir_models/intel/")
    parser.add_argument("--input_file", default='bin/demo.mp4')
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--produce_batch_size", default=64, type=int)
    parser.add_argument("--video_len", default=595, type=int)
    parser.add_argument("--output_path", default="", type=str)
    parser.add_argument("--input_type", default="video", type=str)
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    models_array = args.models.split(",")
    model_paths, model_classes = obtain_models(args)
    input_feeder = InputFeeder(input_type=args.input_type, 
    input_file=args.input_file)
    input_feeder.load_data()
    counter = 0
    width = int(input_feeder.cap.get(3))
    height = int(input_feeder.cap.get(4))
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    
    # generate outputs for face detection input
    def exec_result_frame(obj, net, model_class, model_type, frames, request_id):
        if (model_type == "face") or (model_type == "face_regular"):
            p_frames = [obj.preprocess_input(frame) for frame in frames]
            obj.predict(net, p_frames, request_id=request_id)

    # generate outputs for facial landmarks input
    def exec_result_face(obj, net, model_class, model_type, faces, request_id):
        if (model_type == "facial_landmarks"):
            p_frames = [obj.preprocess_input(face) for face in faces]
            obj.predict(net, p_frames, request_id=request_id)

    # generate outputs for facial landmarks input
    def exec_result_pose(obj, net, model_class, model_type, faces, request_id):
        if (model_type == "head_pose"):
            p_frames = [obj.preprocess_input(face) for face in faces]
            obj.predict(net, p_frames, request_id=request_id)

    # generate outputs for gaze estimation
    def exec_result_gaze(obj, net, model_class, model_type, faces, cropped_left, 
    cropped_right, head_pose, request_id):
        if model_type == "gaze_estimation":
            p_frames = []
            for i,l,r,p in zip(list(range(len(cropped_left))), cropped_left, cropped_right, head_pose):
                ro = obj.preprocess_input_gaze(left=l, right=r, vector=p)
                if ro is not None:
                    p_frames.append(ro)
            obj.predict_dict(net, p_frames, request_id=request_id)

    # extract outputs for frame, exec bounding boxes fo frames
    def produce_outputs_frame(obj, net, model_class, model_type, frames, args, request_id):
        boxes_list = []
        gen_frames = []
        faces = []
        
        for ii, frame in enumerate(frames):
            if (model_type == "face") or (model_type == "face_regular"):
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

    # extract outputs for facial landmarks
    def produce_outputs_face(obj, net, model_class, model_type, frames, faces, face_boxes, args, 
    request_id):
        boxes_list, confs_list = [], []
        gen_frames = []
        cropped_left_eye = []
        cropped_right_eye = []
        left_eye, right_eye, nose, left_lip, right_lip = [], [], [], [], []
        const = 30
        if (model_type == "facial_landmarks"):
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

    # extract outputs for pose
    def produce_outputs_pose(obj, net, model_class, model_type, frames, args, request_id):
        pose_angles_list = []
        if (model_type == "head_pose"):
            output_name = obj.output_blob
            statuses = obj.wait(net, len(frames), request_id)
            if len(statuses) == len(frames):
                for ii, frame in enumerate(frames):
                    outputs = obj.check_model(net, request_id=ii+request_id)
                    matrix, _ = model_class.preprocess_output(outputs, output_name, frame, confidence_level=args.conf)
                    pose_angles_list.append(matrix)
        return pose_angles_list

    # exec outputs for gaze
    def produce_outputs_gaze(obj, net, model_class, model_type, frames, request_id):
        outputs_list = []
        if model_type == "gaze_estimation":
            statuses = obj.wait(net, len(frames), request_id)
            if len(statuses) == len(frames):
                outputs = obj.check_model(net, request_id=request_id)
                outputs_list.append(outputs)
                
        return outputs_list

    def load_network(args, model_class, model_type, model_path):
        if not os.path.isfile(model_path):
            print("Model path does not exist: ", model_path)
            return None, None
        net = Network()
        net.load_model(model_path, device=args.device, cpu_extension=CPU_EXTENSION, args=args)
        c = model_class
        m = c(net)
        m.load_model(net)
        return m, net, model_type
    
    objects = {}
    networks = {}
    objs = {}
    nets = {}

    model_classes = np.array(model_classes)
    models_array = np.array(models_array)
    
    # load network by existence of model file path
    with ThreadPoolExecutor(max_workers=4) as executor:
        for ii, res in zip(list(range(len(model_classes))),
        executor.map(load_network, [args]*len(model_classes), 
        model_classes.tolist(), models_array.tolist(), 
        model_paths)):
            if res[0] is not None:
                objects[ii] = res[0]
                networks[ii] = res[1]
                objs[res[2]] = res[0]
                nets[res[2]] = res[1]
    
    frames = []
    counter = 0
    start_time = time.time()

    while True:
        ret, frame = input_feeder.cap.read()
        if not ret:
            break
        frames.append(frame)
    
    end_time = time.time()

    print(len(frames), " Frame(s) loading time: ", end_time - start_time)
    
    # filtering the models by presence of output
    idxs = [ii for ii, model in enumerate(model_classes) if ii in list(networks.keys())]
    model_classes = model_classes[idxs]
    models_array = models_array[idxs]

    start_time = time.time()
    # preprocessing the model, executing the async inference
    with ThreadPoolExecutor(max_workers=4) as executor:
        for counter in range(0,len(frames),args.batch_size):
            counter_array = [[counter]]*len(list(networks.values()))
            counter_array = np.array(counter_array)
            counter_array = counter_array.flatten().tolist()
            for res in executor.map(exec_result_frame, list(objects.values()), 
            list(networks.values()), 
            model_classes.tolist(), 
            models_array.tolist(), 
            [frames[counter:counter+args.batch_size]]*len(list(objects.keys())), 
            counter_array):
                pass

    print("Preprocess and exec async for face detection: ", time.time() - start_time)

    # for each n batches and for each batch_size
    gen_frames = []
    faces = []
    face_boxes = []

    start_time = time.time()
    # postprocessing the outputs, from face detection for all objects
    with ThreadPoolExecutor(max_workers=4) as executor:
        for counter in range(0,len(frames),args.produce_batch_size):
            counter_array = [[counter]]*len(list(networks.values()))
            counter_array = np.array(counter_array)
            counter_array = counter_array.flatten().tolist()
            for ii, res in zip(list(range(len(list(networks.keys())))),
                executor.map(produce_outputs_frame, list(objects.values()), 
                list(networks.values()),
                model_classes.tolist(), 
                models_array.tolist(), 
                [frames[counter:counter+args.produce_batch_size]]*len(model_classes), 
                [args]*len(model_classes), 
                counter_array)):
                for f in res[0]:
                    gen_frames.append(f)
                for f in res[1]:
                    faces.append(f)
                for f in res[2]:
                    face_boxes.append(f)

    print("Post-process face detection: ", time.time() - start_time)

    start_time = time.time()
    # preprocessing the face and executing the landmarks detection
    with ThreadPoolExecutor(max_workers=4) as executor:
        for counter in range(0,len(frames),args.batch_size):
            counter_array = [[counter]]*len(list(networks.values()))
            counter_array = np.array(counter_array)
            counter_array = counter_array.flatten().tolist()
            for res in executor.map(exec_result_face, list(objects.values()), 
            list(networks.values()), 
            model_classes.tolist(), 
            models_array.tolist(), 
            [faces[counter:counter+args.batch_size]]*len(model_classes), 
            counter_array):
                pass

    print("Preprocess and exec async for facial landmarks: ", time.time() - start_time)

    batch_gen_frames = []
    cropped_left = []
    cropped_right = []
    left_eye, right_eye, nose, left_lip, right_lip = [], [], [], [], []

    start_time = time.time()
    # postprocessing the outputs, from landmarks detection
    with ThreadPoolExecutor(max_workers=4) as executor:
        for counter in range(0,len(frames),args.produce_batch_size):
            counter_array = [[counter]]*len(list(networks.values()))
            counter_array = np.array(counter_array)
            counter_array = counter_array.flatten().tolist()
            for ii, res in zip(list(range(len(list(networks.keys())))),
                executor.map(produce_outputs_face, list(objects.values()), 
                list(networks.values()),
                model_classes.tolist(), 
                models_array.tolist(), 
                [gen_frames[counter:counter+args.produce_batch_size]]*len(model_classes), 
                [faces[counter:counter+args.produce_batch_size]]*len(model_classes), 
                [face_boxes[counter:counter+args.produce_batch_size]]*len(model_classes), 
                [args]*len(model_classes), 
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

    print("Post-process facial landmarks: ", time.time() - start_time)

    start_time = time.time()
    # preprocessing the face and executing the landmarks detection
    with ThreadPoolExecutor(max_workers=4) as executor:
        for counter in range(0,len(frames),args.batch_size):
            counter_array = [[counter]]*len(list(networks.values()))
            counter_array = np.array(counter_array)
            counter_array = counter_array.flatten().tolist()
            for res in executor.map(exec_result_pose, list(objects.values()), 
            list(networks.values()), 
            model_classes.tolist(), 
            models_array.tolist(), 
            [faces[counter:counter+args.batch_size]]*len(model_classes), 
            counter_array):
                pass

    print("Preprocess and exec async head pose: ", time.time() - start_time)

    head_pose_angles = []
    start_time = time.time()
    # postprocessing the outputs, from landmarks detection
    with ThreadPoolExecutor(max_workers=4) as executor:
        for counter in range(0,len(frames),args.produce_batch_size):
            counter_array = [[counter]]*len(list(networks.values()))
            counter_array = np.array(counter_array)
            counter_array = counter_array.flatten().tolist()
            for ii, res in zip(list(range(len(list(networks.keys())))),
                executor.map(produce_outputs_pose, list(objects.values()), 
                list(networks.values()),
                model_classes.tolist(), 
                models_array.tolist(), 
                [batch_gen_frames[counter:counter+args.produce_batch_size]]*len(model_classes), 
                [args]*len(model_classes), 
                counter_array)):
                for f in res:
                    head_pose_angles.append(f)

    print("Post-process head pose: ", time.time() - start_time)

    gen_frames = None

    start_time = time.time()
    # preprocessing the gaze and executing the landmarks detection
    with ThreadPoolExecutor(max_workers=4) as executor:
        for counter in range(0,len(frames),args.batch_size):
            try:
                for ii in executor.map(range(len(models_array))):
                    exec_result_gaze(objs['gaze_estimation'], 
                    nets['gaze_estimation'], model_classes.tolist()[len(model_classes)-1], 
                    models_array.tolist()[len(models_array)-1], 
                    faces[counter:counter+args.batch_size], 
                    cropped_left[counter:counter+args.batch_size], 
                    cropped_right[counter:counter+args.batch_size], 
                    head_pose_angles[counter:counter+args.batch_size], counter)
            except Exception as e:
                print(e.args)

    print("Preprocess and exec async gaze estimation: ", time.time() - start_time)

    faces = None

    gaze_vector = []
    start_time = time.time()
    # post process gaze vector
    for counter in range(0,len(frames),args.produce_batch_size):
        res = produce_outputs_gaze(objs['gaze_estimation'], 
        nets['gaze_estimation'], model_classes.tolist()[len(model_classes)-1], 
        models_array.tolist()[len(models_array)-1], 
        batch_gen_frames[counter:counter+args.produce_batch_size], counter)
        for f in res[0]:
            gaze_vector.append(f)

    print("Post-process gaze: ", time.time() - start_time)

    start_time = time.time()

    try:
        for ii, frame in enumerate(batch_gen_frames):
            xmin, ymin, xmax, ymax = face_boxes[ii]
            left_eye_x, left_eye_y = left_eye[ii]
            right_eye_x, right_eye_y = right_eye[ii]
            nose_x, nose_y = nose[ii]
            left_lip_x, left_lip_y = left_lip[ii]
            right_lip_x, right_lip_y = right_lip[ii]
            draw_markers(frame, face_boxes[ii], left_eye_x, left_eye_y, right_eye_x, right_eye_y, 
            nose_x, nose_y, left_lip_x, left_lip_y, right_lip_x, right_lip_y)
            draw_boxes(frame, face_boxes[ii])
            if ii < len(gaze_vector):
                cv2.putText(frame, str(gaze_vector[ii]), 
                (15, 40), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
            out.write(frame)
    except Exception as e:
        raise e
    
    print("Post-process video writing and painting time: ", time.time() - start_time)
    
    
    out.release()
    input_feeder.close()
    cv2.destroyAllWindows()

def draw_markers(frame_copy, face_box, left_eye_x, left_eye_y, right_eye_x, right_eye_y, 
nose_x, nose_y, left_lip_x, left_lip_y, right_lip_x, right_lip_y, color=(0,255,0), 
thickness=5, radius=5):
    xmin, ymin, xmax, ymax = face_box
    cv2.circle(frame_copy, (xmin+left_eye_x, ymin+left_eye_y), radius, color, thickness)
    cv2.circle(frame_copy, (xmin+right_eye_x, ymin+right_eye_y), radius, color, thickness)
    cv2.circle(frame_copy, (xmin+nose_x, ymin+nose_y), radius, color, thickness)
    cv2.circle(frame_copy, (xmin+left_lip_x, ymin+left_lip_y), radius, color, thickness)
    cv2.circle(frame_copy, (xmin+right_lip_x, ymin+right_lip_y), radius, color, thickness)

def draw_boxes(frame_copy, face_box):
    xmin, ymin, xmax, ymax = face_box
    cv2.rectangle(frame_copy, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

def draw_axis():
    pass

def deliver_color(idx):
    c = (255,255,255)
    if (idx == 0):
        c = (0,255,0)
    elif (idx == 1):
        c = (0,255,255)
    elif (idx == 2):
        c = (0,0,255)
    elif (idx == 3):
        c = (255,0,0)
    elif (idx == 4):
        c = (255,255,0)
    elif (idx == 5):
        c = (255,255,255)
    return c


if __name__ == "__main__":

    main()
