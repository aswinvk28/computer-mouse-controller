from src.face_detection import FaceDetection
from src.face_landmarks_detection import FaceLandMarksDetection
from src.gaze_estimation import GazeEstimation
from src.head_pose_estimation import HeadPoseEstimation
import time

def get_face_model(precision='INT1'):
    return "face-detection-adas-binary-0001/"+precision+"/face-detection-adas-binary-0001.xml"

def get_face_model_regular(precision='FP16'):
    return "face-detection-adas-0001/"+precision+"/face-detection-adas-0001.xml"

def get_face_landmarks_model(precision='FP16'):
    return "landmarks-regression-retail-0009/"+precision+"/landmarks-regression-retail-0009.xml"

def get_gaze_model(precision='FP16'):
    return "gaze-estimation-adas-0002/"+precision+"/gaze-estimation-adas-0002.xml"

def get_head_pose_model(precision='FP16'):
    return "head-pose-estimation-adas-0001/"+precision+"/head-pose-estimation-adas-0001.xml"

def obtain_models(args, prefix="/home/workspace/ir_models/intel/"):
    models = args.models.split(",")
    model_paths = []
    model_classes = []
    model_path = None
    model_class = None
    for m in models:
        if m == "face":
            model_path = prefix + get_face_model(args.precision)
            model_class = FaceDetection
        if m == "face_regular":
            model_path = prefix + get_face_model_regular(args.precision)
            model_class = FaceDetection
        elif m == "head_pose":
            model_path = prefix + get_head_pose_model(args.precision)
            model_class = HeadPoseEstimation
        elif m == "facial_landmarks":
            model_path = prefix + get_face_landmarks_model(args.precision)
            model_class = FaceLandMarksDetection
        elif m == "gaze_estimation":
            model_path = prefix + get_gaze_model(args.precision)
            model_class = GazeEstimation
        model_paths.append(model_path)
        model_classes.append(model_class)

    return model_paths, model_classes