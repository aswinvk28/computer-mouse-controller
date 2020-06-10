'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
import numpy as np
from .Adapter import Adapter
from concurrent.futures import ThreadPoolExecutor
from openvino.inference_engine import IENetwork, IECore

class HeadPoseEstimation(Adapter):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, net, threshold=0.60, args=None):
        Adapter.__init__(self, net, threshold, args)
            
    @staticmethod
    def preprocess_output(outputs, output_name, frame, confidence_level=0.5):
        '''
        Preprocesses the output into a matrix of angles
        '''
        try:
            keys = ['angle_y_fc', 'angle_p_fc', 'angle_r_fc']
            output_keys = ['yaw', 'pitch', 'roll']

            matrix = np.zeros((outputs[keys[0]].shape[1],3))
            for ii,k in enumerate(keys):
                for idx in range(outputs[k].shape[1]):
                    matrix[idx,ii] = outputs[k][0,idx].item()

        except Exception as e:
            raise e

        # result matrix
        return matrix, []