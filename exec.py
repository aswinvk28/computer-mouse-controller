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
from model_list import obtain_models

import logging

logging.basicConfig(filename='app.log', filemode='w', level=logging.INFO) #optionally log into file


CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''

    face_desc = """
    face recognition model
    """
    gaze_desc = """
    gaze estimation model
    """
    landmarks_desc = """
    landmarks model
    """
    pose_desc = """
    head pose model
    """
    device_desc = """
    device model
    """
    precision_desc = """
    precision model
    """
    precisions_order_desc = """
    precisions order for all the models
    """
    conf_desc = """
    confidence interval for all the models
    """
    prefix_desc = """
    prefix for all the models
    """
    input_file_desc = """
    input file to run the app
    """
    batch_size_file_desc = """
    batch size file to run the app
    """
    produce_batch_size_file_desc = """
    produce batch size file to run the app
    """
    video_len_desc = """
    video len size file to run the app
    """
    time_execution_desc = """
    time of execution for the app
    """
    output_path_desc = """
    output path for the app
    """
    input_type_desc = """
    input type for the app
    """

    parser = argparse.ArgumentParser()
    m_desc = "The list of tagged models: face_regular,head_pose,facial_landmarks,gaze_estimation"
    parser.add_argument("--face", help=face_desc, default="face", required=True)
    parser.add_argument("--gaze", help=gaze_desc, default="gaze_estimation", required=True)
    parser.add_argument("--landmarks", help=landmarks_desc, default="facial_landmarks", required=True)
    parser.add_argument("--pose", help=pose_desc, default="head_pose", required=True)
    parser.add_argument('--device', help=device_desc, default='CPU')
    parser.add_argument('--precision', help=precision_desc, default='')
    parser.add_argument('--precisions_order', help=precisions_order_desc, default='INT1,FP16,FP16,FP16')
    parser.add_argument('--conf', help=conf_desc, default=0.6, type=float)
    parser.add_argument('--prefix', help=prefix_desc, default="/home/workspace/ir_models/intel/")
    parser.add_argument("--input_file", help=input_file_desc, default='bin/demo.mp4')
    parser.add_argument("--batch_size", help=batch_size_file_desc, default=64, type=int)
    parser.add_argument("--produce_batch_size", help=produce_batch_size_file_desc, default=64, type=int)
    parser.add_argument("--video_len", help=video_len_desc, default=595, type=int)
    parser.add_argument("--time_execution", help=time_execution_desc, default=3000, type=int)
    parser.add_argument("--output_path", help=output_path_desc, default="", type=str)
    parser.add_argument("--input_type", help=input_type_desc, default="video", type=str)
    
    args = parser.parse_args()

    return args

def convert_perf_time(args):

    import re
    
    perf_text = """595  Frame(s) loading time:  2.779841423034668
    Creating Image Frame pipeline: 
    Preprocess and exec async for face detection:  2.462164878845215
    Post-process face detection:  17.501105308532715
    Preprocess and exec async for facial landmarks:  0.11006045341491699
    Post-process facial landmarks:  1.4040935039520264
    Preprocess and exec async head pose:  0.13110589981079102
    Post-process head pose:  0.8902149200439453
    Preprocess and exec async gaze estimation:  0.00014662742614746094
    Post-process gaze:  0.020793676376342773
    Post-process video writing and painting time:  6.754056930541992"""

    result = re.split("\n", perf_text)

    order = ['frame_load', 'face:pre', 'face:post', 'facial:pre', 'facial:post', 
    'head:pre', 'head:post', 'gaze:pre', 'gaze:post', 'paint']

    def extract(r):
        l = r.split(" ")
        return l[len(l)-1]

    result = list(map(lambda x: float(x), 
    list(filter(lambda x: x.strip() != "", list(map(extract, result))))))

    return dict(zip(order, result))

def convert_perf_time_video_len(args, perf_stats, reference_video_len=None):
    per_frame_execution_time = 0.0

    per_frame_execution_time += perf_stats['frame_load'] / reference_video_len
    per_frame_execution_time += perf_stats['face:pre'] / reference_video_len
    per_frame_execution_time += perf_stats['face:post'] / (reference_video_len*0.7)
    per_frame_execution_time += perf_stats['facial:pre'] / (reference_video_len)
    per_frame_execution_time += perf_stats['facial:post'] / (reference_video_len*0.7)
    per_frame_execution_time += perf_stats['head:pre'] / reference_video_len
    per_frame_execution_time += perf_stats['head:post'] / (reference_video_len*0.7)
    per_frame_execution_time += perf_stats['gaze:pre'] / reference_video_len
    per_frame_execution_time += perf_stats['gaze:post'] / (reference_video_len*0.7)
    per_frame_execution_time += perf_stats['paint'] / reference_video_len

    return per_frame_execution_time

def extract_model_type(args):
    return args.face, args.landmarks, args.pose, args.gaze

def is_image():
    return ['.png', '.jpg', '.jpeg']

def is_video():
    return ['.mp4', '.avi']

def run_pipeline(logging, frames, models, model_classes, models_array, 
model_paths, args, start_time, out, save=False):

    end_time = time.time()

    face, landmarks, pose_estimation, gaze = models

    logging.info("""{f} Frame(s) loading time: {t}""".format(f=len(frames), 
    t=(end_time - start_time)))

    logging.info("Creating Image Frame pipeline: ")

    default_pipeline = Pipeline(model_type=None, model_class=None, objects={}, 
    networks={}, objs={}, nets={}, logging=logging)

    # load the network objects
    default_pipeline.run(model_classes, models_array, model_paths, args)

    image_frame = ImageFrame(model_type=face, model_class=model_classes, 
    objects=default_pipeline.objects, networks=default_pipeline.networks, 
    objs=default_pipeline.objs, nets=default_pipeline.nets)

    face_detection = Face(model_type=landmarks, model_class=model_classes.tolist()[1], 
    objects=default_pipeline.objects, networks=default_pipeline.networks, 
    objs=default_pipeline.objs, nets=default_pipeline.nets)

    gaze_estimation = Gaze(model_type=gaze, model_class=model_classes.tolist()[3], 
    objects=default_pipeline.objects, networks=default_pipeline.networks, 
    objs=default_pipeline.objs, nets=default_pipeline.nets)

    start_time = time.time()

    image_frame.run(args, frames, model_classes)

    logging.info("""Preprocess and exec async for face detection: {t}""".format(t=(time.time() - start_time)))

    # for each n batches and for each batch_size
    start_time = time.time()

    gen_frames, faces, face_boxes = \
        image_frame.produce(args, frames, model_classes)

    logging.info("""Post-process face detection: {t}""".format(t=(time.time() - start_time)))

    start_time = time.time()
    
    face_detection.run(args, frames, faces, model_classes)

    logging.info("""Preprocess and exec async for facial landmarks: {t}""".format(t=(time.time() - start_time)))

    start_time = time.time()

    batch_gen_frames, cropped_left, \
        cropped_right, left_eye, right_eye, \
        nose, left_lip, right_lip = \
            face_detection.produce(args, frames, gen_frames, faces, face_boxes, 
    model_classes)

    logging.info("""Post-process facial landmarks: {t}""".format(t=(time.time() - start_time)))

    start_time = time.time()

    pose_model = Pose(model_type=pose_estimation, 
    model_class=model_classes.tolist()[2], 
    objects=default_pipeline.objects, networks=default_pipeline.networks, 
    objs=default_pipeline.objs, nets=default_pipeline.nets)

    pose_model.run(args, frames, faces, model_classes)
    
    logging.info("""Preprocess and exec async head pose: {t}""".format(t=(time.time() - start_time)))

    start_time = time.time()

    head_pose_angles = pose_model.produce(args, frames, 
    batch_gen_frames, model_classes)

    logging.info("""Post-process head pose: {t}""".format(t=(time.time() - start_time)))

    gen_frames = None

    start_time = time.time()
    # preprocessing the gaze and executing the landmarks detection
    gaze_estimation.run(args, frames, faces, 
    cropped_left, cropped_right, 
    head_pose_angles, model_classes)

    logging.info("""Preprocess and exec async gaze estimation: {t}""".format(t=(time.time() - start_time)))

    faces = None

    start_time = time.time()
    # post process gaze vector
    gaze_vector = gaze_estimation.produce(args, frames, batch_gen_frames, 
    model_classes)

    logging.info("""Post-process gaze: {t}""".format(t=(time.time() - start_time)))

    start_time = time.time()
    ext = os.path.splitext(args.output_path)[1]

    if ext in is_video():
        default_pipeline.finalize_pipeline(out, frames, 
        args, batch_gen_frames, face_boxes, 
        left_eye, right_eye, 
        nose, left_lip, right_lip, 
        gaze_vector, save=save)
    else:
        frame = default_pipeline.write_pipeline(0, frames, batch_gen_frames, face_boxes, 
        left_eye, right_eye, nose, left_lip, right_lip, 
        gaze_vector)
        cv2.imwrite(out, frame)

    logging.info("""Post-process video writing and painting time: {t}""".format(t=(time.time() - start_time)))

    return batch_gen_frames

def main():
    args = get_args()
    models = extract_model_type(args)
    if args.precisions_order:
        precisions = args.precisions_order.split(",")
    else:
        precision = args.precision
    face, landmarks, pose_estimation, gaze = models
    model_paths, model_classes = obtain_models(args, models, precisions)
    input_feeder = InputFeeder(input_type=args.input_type, 
    input_file=args.input_file)
    input_feeder.load_data()
    counter = 0
    ext = os.path.splitext(args.output_path)[1]
    if ext in is_video():
        width = int(input_feeder.cap.get(3))
        height = int(input_feeder.cap.get(4))
    if ext in is_image():
        out = args.output_path
    else:
        out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    
    perf_stats = convert_perf_time(args)

    per_frame_execution_time = convert_perf_time_video_len(args, perf_stats, args.video_len)

    fps = 1 / per_frame_execution_time
    
    models_array = list(models)

    model_classes = np.array(model_classes)
    models_array = np.array(models_array)
    
    frames = []
    start_time = time.time()

    if (args.video_len <= args.produce_batch_size) or (args.video_len <= args.batch_size):
        raise Exception("Video len is less than the expected batch size")

    if args.input_type == "cam":

        input_cap = cv2.VideoCapture(args.input_file)

        args.batch_size = 4
        args.produce_batch_size = 4
        counter = 0

        total_frames = int(args.time_execution * fps)
        args.video_len = total_frames

        while True:
            ret, frame = input_cap.read()
            key_pressed = cv2.waitKey(60)
            if not ret:
                break
            frames.append(frame)
            if (counter+1) % args.batch_size == 0:
                batch_gen_frames = run_pipeline(logging, frames, 
                models, model_classes, models_array, 
                model_paths, args, start_time, out, save=False)

                for frame in batch_gen_frames:
                    cv2.imshow("window", frame)

                frames = []
            
            # Break if escape key pressed
            if key_pressed == 27:
                break
            
            counter += 1

            if counter >= total_frames:
                break

    else:
        if ext in is_video():
            while True:
                ret, frame = input_feeder.cap.read()
                if not ret:
                    break
                frames.append(frame)
        else:
            frames.append(cv2.imread(args.input_file))

        run_pipeline(logging, frames, models, model_classes, models_array, 
        model_paths, args, start_time, out, save=True)

    out.release()
    input_feeder.close()
    cv2.destroyAllWindows()

    logging.shutdown()

if __name__ == "__main__":

    main()
