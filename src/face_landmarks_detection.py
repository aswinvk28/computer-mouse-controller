'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
import numpy as np
from .Adapter import Adapter
from concurrent.futures import ThreadPoolExecutor

class FaceLandMarksDetection(Adapter):
    '''
    Class for the Face landmarks Detection Model.
    '''
    @staticmethod
    def preprocess_output(outputs, output_name, frame, confidence_level=0.5):
        '''
        Preprocesses the output based on facial landmark points
        '''
        try:
            outputs = outputs[output_name]
            boxes = []
            height, width = frame.shape[:2]
            if len(outputs) > 0:
                for res in outputs:
                    left_eye_x, left_eye_y, \
                    right_eye_x, right_eye_y, nose_x, nose_y, \
                    left_lip_x, left_lip_y, right_lip_x, right_lip_y = tuple(res[:,0,0].tolist())
                    
                    left_eye_x = int(left_eye_x*width)
                    left_eye_y = int(left_eye_y*height)
                    right_eye_x = int(right_eye_x*width)
                    right_eye_y = int(right_eye_y*height)
                    nose_x = int(nose_x*width)
                    nose_y = int(nose_y*height)
                    left_lip_x = int(left_lip_x*width)
                    left_lip_y = int(left_lip_y*height)
                    right_lip_x = int(right_lip_x*width)
                    right_lip_y = int(right_lip_y*height)
                    boxes.append((left_eye_x, left_eye_y, right_eye_x, right_eye_y, \
                    nose_x, nose_y, left_lip_x, left_lip_y, right_lip_x, right_lip_y))
        except Exception as e:
            print(res.shape)
            raise e
            print(e.args)

        # coordinate points
        return boxes, []
