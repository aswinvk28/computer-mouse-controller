from src.face_detection import FaceDetection
from src.face_landmarks_detection import FaceLandMarksDetection
from src.gaze_estimation import GazeEstimation
from src.head_pose_estimation import HeadPoseEstimation
import time

def get_face_model():
    return "face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001.xml"

def get_face_landmarks_model():
    return "facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml"

def get_gaze_model():
    return "gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml"

def get_head_pose_model():
    return "head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml"

def obtain_models(args, prefix="/home/workspace/ir_models/intel/"):
    models = args.models.split(",")
    model_paths = []
    model_classes = []
    model_objects = []
    model = None
    for m in models:
        if m == "face":
            model_path = prefix + get_face_model()
            model_class = FaceDetection
        # elif m == "head_pose":
        #     model_path = prefix + get_head_pose_model()
        #     if args.create_models:
        #         model = HeadPoseEstimation(model_path, args.device)
        #     model_class = HeadPoseEstimation
        # elif m == "facial_landmarks":
        #     model_path = prefix + get_face_landmarks_model()
        #     if args.create_models:
        #         model = FaceLandMarksDetection(model_path, args.device)
        #     model_class = FaceLandMarksDetection
        # elif m == "gaze_estimation":
        #     model_path = prefix + get_gaze_model()
        #     if args.create_models:
        #         model = GazeEstimation(model_path, args.device)
        #     model_class = GazeEstimation
        model_paths.append(model_path)
        model_classes.append(model_class)

    return model_paths, model_classes