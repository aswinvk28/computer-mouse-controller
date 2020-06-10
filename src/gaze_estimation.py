import os
import cv2
import numpy as np
from .Adapter import Adapter
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

class GazeEstimation(Adapter):
    '''
    Class for the Gaze Estimation Model.
    '''

    def __init__(self, net, threshold=0.60, args=None):
        Adapter.__init__(self, net, threshold, args)
        self.input_blob = list(iter(net.network.inputs))

    def load_model(self, net):
        '''
        load the network input shape
        '''
        self.net_input_shape = OrderedDict()
        for name in self.input_blob:
            self.net_input_shape[name] = net.network.inputs[name].shape
        
    def predict(self, net, batch_images, request_id=0):
        for ii, p_frame in enumerate(batch_images):
            head_pose_angles = np.random.randn(1,3) * 10.0
            p_frame['head_pose_angles'] = head_pose_angles
            net.exec_network.start_async(inputs=p_frame, request_id=ii)
        for i in range(len(batch_images)):
            status = net.wait(request_id=i)

    def predict_dict(self, net, batch_images, request_id=0):
        for ii, p_frame in enumerate(batch_images):
            net.exec_network.start_async(inputs=p_frame, request_id=ii+request_id)
    
    def preprocess(self, image, name):
        '''
        The function preprocesses the input image to the model input shape
        '''
        p_frame = None
        if len(image.shape) > 2:
            p_frame = cv2.resize(image, (self.net_input_shape[name][3], self.net_input_shape[name][2]))
            p_frame = p_frame.transpose((2,0,1))
            p_frame = p_frame.reshape(1, *p_frame.shape)

        return p_frame
        
    def preprocess_input(self, frame=None, vector=None):
        '''
        te function uses random data to produce outputs
        This class is used for counting the performance counts as well as 
        to execute the inference time, load time, etc
        '''
        result = OrderedDict()
        for ii, name in enumerate(self.input_blob):
            if ii == 0:
                result[name] = vector
            else:
                result[name] = self.preprocess(frame, name)

        return result

    def preprocess_input_gaze(self, left=None, right=None, vector=None):
        '''
        The function executes gaze input preprocess
        '''
        result = {}
        result["head_pose_angles"] = vector
        result["left_eye_image"] = self.preprocess(left, "left_eye_image")
        result["right_eye_image"] = self.preprocess(right, "right_eye_image")
        return result

    @staticmethod
    def preprocess_output(outputs, output_name, frame, confidence_level=0.5):
        '''
        preprocesses the output
        '''
        # coordinate vector
        return outputs[output_name], []
