import argparse
from concurrent.futures import ThreadPoolExecutor
from src.input_feeder import InputFeeder
import threading
import cv2
import os
import numpy as np
import time
from pipeline.Pipeline import Pipeline
from pipeline.Face import Face
from pipeline.Gaze import Gaze
from pipeline.ImageFrame import ImageFrame
from pipeline.Pose import Pose
from scripts.Network import Network
from model_list import get_face_model_regular, get_face_model, get_face_landmarks_model, get_gaze_model, get_head_pose_model, obtain_models

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser()
    m_desc = "The list of tagged models: face_regular,head_pose,facial_landmarks,gaze_estimation"
    parser.add_argument("--face", help=m_desc, default="face", required=True)
    parser.add_argument("--gaze", help=m_desc, default="gaze_estimation", required=True)
    parser.add_argument("--landmarks", help=m_desc, default="facial_landmarks", required=True)
    parser.add_argument("--pose", help=m_desc, default="head_pose", required=True)
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

def extract_model_type(args):
    return args.face, args.landmarks, args.pose, args.gaze

def main():
    args = get_args()
    models = extract_model_type(args)
    face, landmarks, pose_estimation, gaze = models
    model_paths, model_classes = obtain_models(args, models)
    input_feeder = InputFeeder(input_type=args.input_type, 
    input_file=args.input_file)
    input_feeder.load_data()
    counter = 0
    width = int(input_feeder.cap.get(3))
    height = int(input_feeder.cap.get(4))
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    
    models_array = list(models)

    model_classes = np.array(model_classes)
    models_array = np.array(models_array)
    
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

    print("Creating Image Frame pipeline: ")
    default_pipeline = Pipeline(model_type=None, model_class=None)
    image_frame = ImageFrame(model_type=face, model_class=model_classes, 
    objects=default_pipeline.objects, networks=default_pipeline.networks, 
    objs=default_pipeline.objs, nets=default_pipeline.nets)
    face_detection = Face(model_type=landmarks, model_class=model_classes.tolist()[1], 
    objects=default_pipeline.objects, networks=default_pipeline.networks, 
    objs=default_pipeline.objs, nets=default_pipeline.nets)

    # load the network objects
    default_pipeline.run(model_classes, models_array, model_paths, args)

    gaze_estimation = Gaze(model_type=gaze, model_class=model_classes.tolist()[3], 
    objects=default_pipeline.objects, networks=default_pipeline.networks, 
    objs=default_pipeline.objs, nets=default_pipeline.nets)

    # filtering the models by presence of output
    # idxs = [ii for ii, model in enumerate(model_classes) if ii in list(default_pipeline.networks.keys())]
    # model_classes = model_classes[idxs]
    # models_array = models_array[idxs]

    start_time = time.time()

    image_frame.run(args, frames, model_classes)

    print("Preprocess and exec async for face detection: ", time.time() - start_time)

    # for each n batches and for each batch_size
    start_time = time.time()

    gen_frames, faces, face_boxes = \
        image_frame.produce(args, frames, model_classes)

    print("Post-process face detection: ", time.time() - start_time)

    start_time = time.time()
    
    face_detection.run(args, frames, faces, model_classes)

    print("Preprocess and exec async for facial landmarks: ", time.time() - start_time)

    start_time = time.time()

    batch_gen_frames, cropped_left, \
        cropped_right, left_eye, right_eye, \
        nose, left_lip, right_lip = \
            face_detection.produce(args, frames, gen_frames, faces, face_boxes, 
    model_classes)

    print("Post-process facial landmarks: ", time.time() - start_time)

    start_time = time.time()

    pose_model = Pose(model_type=pose_estimation, 
    model_class=model_classes.tolist()[2], 
    objects=default_pipeline.objects, networks=default_pipeline.networks, 
    objs=default_pipeline.objs, nets=default_pipeline.nets)

    pose_model.run(args, frames, faces, model_classes)
    
    print("Preprocess and exec async head pose: ", time.time() - start_time)

    start_time = time.time()

    head_pose_angles = pose_model.produce(args, frames, 
    batch_gen_frames, model_classes)

    print("Post-process head pose: ", time.time() - start_time)

    gen_frames = None

    start_time = time.time()
    # preprocessing the gaze and executing the landmarks detection
    gaze_estimation.run(args, frames, faces, 
    cropped_left, cropped_right, 
    head_pose_angles, model_classes)

    print("Preprocess and exec async gaze estimation: ", time.time() - start_time)

    faces = None

    start_time = time.time()
    # post process gaze vector
    gaze_vector = gaze_estimation.produce(args, frames, batch_gen_frames,
    model_classes)

    print("Post-process gaze: ", time.time() - start_time)

    start_time = time.time()

    default_pipeline.finalize_pipeline(out, frame, 
    batch_gen_frames, face_boxes, 
    left_eye, right_eye, 
    nose, left_lip, right_lip, 
    gaze_vector)

    print("Post-process video writing and painting time: ", time.time() - start_time)
    
    out.release()
    input_feeder.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    main()
