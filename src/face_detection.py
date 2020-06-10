import os
import cv2
import numpy as np
import sys
from .Adapter import Adapter
from concurrent.futures import ThreadPoolExecutor

class FaceDetection(Adapter):
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, net, threshold=0.60, args=None):
        Adapter.__init__(self, net, threshold, args)

    @staticmethod
    def preprocess_output(outputs, output_name, frame, confidence_level=0.5):
        '''
        preprocesses the outputs to bounding boxes 
        '''
        try:
            outputs = outputs[output_name]
            boxes = []
            confs = []
            height, width = frame.shape[:2]
            if len(outputs[0][0]) > 0:
                for res in outputs[0][0]:
                    _, label, conf, xmin, ymin, xmax, ymax = res
                    if conf > confidence_level:
                        xmin = int(xmin*width)
                        ymin = int(ymin*height)
                        xmax = int(xmax*width)
                        ymax = int(ymax*height)
                        boxes.append([xmin, ymin, xmax, ymax])
                        confs.append(conf)
        except Exception as e:
            print(e.args)
        
        # coordinate points
        return boxes, confs
