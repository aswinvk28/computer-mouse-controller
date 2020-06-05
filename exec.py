import argparse
from concurrent.futures import ThreadPoolExecutor
from src.input_feeder import InputFeeder
import threading
from model_list import get_face_model, get_face_landmarks_model, get_gaze_model, get_head_pose_model, get_network, obtain_models

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="", required=True)
    parser.add_argument("--device", help="", default='CPU')
    parser.add_argument("--conf", help="", default=0.65)
    parser.add_argument("--batch_size", help="", default=32)
    parser.add_argument("--threshold", help="", default=2, type=int)

    args = parser.parse_args()

    return args

def run_models(frame, objects, args):

    def exec_result(obj, frame):
        obj.predict(frame)
        p_frame = obj.preprocess_input(frame)
        return p_frame

    def produce_outputs(obj, model_class, frame):
        outputs = obj.check_model()
        output_name = obj.output_name
        boxes, confs = model_class.preprocess_output(outputs, output_name, frame, confidence_level=0.5)
    
    height, width = frame.shape[:2]
    with ThreadPoolExecutor(max_workers=3) as executor:
        
        output_frames = []
        for res in executor.map(exec_result, (objects, [frame]*len(objects))):
            output_frames.append(res)

        output_result = []
        for ii, res in zip(list(range(len(objects))),
            executor.map(produce_outputs, (objects, [frame]*len(objects), [args.conf]*len(objects)))):
            boxes, confs = res
            output_result.append(res)

    return output_frames, output_result

def main():
    args = parse_args()
    model_paths, model_objects, model_classes = obtain_models(args)
    input_feeder = InputFeeder(input_type='video', args.input_file)
    input_feeder.load_data()
    threads = []
    counter = 0
    frames = []
    width = int(cap.get(3))
    height = int(cap.get(4))
    boxes_list = []
    confs_list = []
    out = cv2.VideoWriter("output_video.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    for frame in input_feeder.next_batch():
        if (counter+1) % 4 == 0:
            with ThreadPoolExecutor(max_workers=2) as executor:
                for result in executor.map(run_models, (frames, [model_classes]*len(frames), [args]*len(frames))):
                    output_frames.append(result[0])
                    output_result.append(result[1])
            # for each frame for each model
            for ii, frame in enumerate(output_frames):
                res = output_result[ii]
                for jj, model in enumerate(model_classes):
                    boxes, confs = res[jj]
                

        else:
            frames = []
        frames.append(frame)
        counter += 1
    input_feeder.close()
