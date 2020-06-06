import argparse
from concurrent.futures import ThreadPoolExecutor
from src.input_feeder import InputFeeder
import threading
import cv2
import os
from scripts.inference import Network
from model_list import get_face_model_regular, get_face_model, get_face_landmarks_model, get_gaze_model, get_head_pose_model, obtain_models

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="", required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--precision', default='FP16')
    parser.add_argument('--threshold', default=0.6, type=float)
    parser.add_argument("--input_file", help="", default='bin/demo.mp4')
    parser.add_argument("--conf", help="", default=0.65)
    parser.add_argument("--batch_size", help="", default=32)
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    models_array = args.models.split(",")
    model_paths, model_classes = obtain_models(args)
    input_feeder = InputFeeder(input_type='video', input_file=args.input_file)
    input_feeder.load_data()
    counter = 0
    width = int(input_feeder.cap.get(3))
    height = int(input_feeder.cap.get(4))
    boxes_list = []
    confs_list = []
    frame_dict = {}
    out = cv2.VideoWriter("output_video.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    
    def exec_result(obj, net, frames):
        p_frames = [obj.preprocess_input(frame) for frame in frames]
        obj.predict(net, p_frames)

    def produce_outputs(obj, net, model_class, frames, object_idx):
        output_name = obj.output_blob
        boxes_list, confs_list = [], []
        for ii, frame in enumerate(frames):
            outputs = obj.check_model(net, request_id=ii)
            boxes, confs = model_class.preprocess_output(outputs, output_name, frame, confidence_level=0.5)
            boxes_list.append(boxes)
            confs_list.append(confs)
        return boxes_list, confs_list

    def load_network(args, model_class, model_path):
        if not os.path.isfile(model_path):
            print("Model path does not exist: ", model_path)
            return None, None
        net = Network()
        net.load_model(model_path, device=args.device, cpu_extension=CPU_EXTENSION, args=args)
        c = model_class
        m = c(net)
        m.load_model(net)
        return m, net
    
    objects = {}
    networks = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for ii, res in zip(list(range(len(model_classes))),
        executor.map(load_network, [args]*len(model_classes), model_classes, model_paths)):
            if res[0] is not None:
                objects[ii] = res[0]
                networks[ii] = res[1]
    output_result = {}
    output_frames = []
    frames = []
    counter = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            ret, frame = input_feeder.cap.read()
            if not ret:
                break
            frames.append(frame)
        for counter in range(0,len(frames),args.batch_size):
            for res in executor.map(exec_result, list(objects.values()), 
            list(networks.values()), 
            [frames[counter:counter+args.batch_size]]*len(list(objects.keys()))):
                output_frames.append(res)
            model_classes = [model for ii,model in enumerate(model_classes) if ii in list(networks.keys())]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for counter in range(0,len(frames),args.batch_size):
            for ii, res in zip(list(range(len(list(networks.keys())))),
                executor.map(produce_outputs, list(objects.values()), list(networks.values()),
                model_classes, 
                [frames[counter:counter+args.batch_size]]*len(model_classes), 
                list(range(len(model_classes))))):
                output_result[ii] = res
    output_result_keys = sorted(output_result)
    try:
        for frame in frames:
            for key in output_result_keys:
                # to draw
                if models_array[key] == "face":
                    idx = 0
                    for boxes in output_result[key][0]:
                        c = deliver_color(idx)
                        for box in boxes:
                            xmin, ymin, xmax, ymax = box
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), c, 2)
                        idx += 1
                elif models_array[key] == "face_regular":
                    idx = 0
                    for boxes in output_result[key][0]:
                        c = deliver_color(idx)
                        for box in boxes:
                            xmin, ymin, xmax, ymax = box
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), c, 2)
                        idx += 1
                elif models_array[key] == "facial_landmarks":
                    for boxes in output_result[key][0]:
                        for box in boxes:
                            cv2.circle(frame, box[0], 2, (0,255,0))
                            cv2.circle(frame, box[1], 2, (0,255,0))
                            cv2.circle(frame, box[2], 2, (0,255,0))
                            cv2.circle(frame, box[3], 2, (0,255,0))
                            cv2.circle(frame, box[4], 2, (0,255,0))
                elif models_array[key] == "head_pose":
                    for matrixes in output_result[key][0]:
                        for matrix in matrixes:
                            y_pixel = 40
                            out_text = """ {v1} yaw {v2} pitch {v3} roll """.format(v1=matrix[0], v2=matrix[1], v3=matrix[2])
                            cv2.putText(frame, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            out.write(frame)
    except Exception as e:
        print(e.args)
    out.release()
    input_feeder.close()
    cv2.destroyAllWindows()

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
