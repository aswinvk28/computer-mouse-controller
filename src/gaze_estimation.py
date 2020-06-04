'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.device_name = device
        self.model_name = model_name
        self.model_bin = os.path.splitext(self.model_name)[0] + ".bin"

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        try:
            self.core = IECore()
            self.model = IENetwork()
            self.net = self.core.load_network(self.model, device_name=self.device_name)
            self.input_names = list(iter(self.model.inputs))
            self.output_name = next(iter(self.model.outputs))
            self.net_input_shape = {}
            for key in self.input_names:
                self.net_input_shape[key] = self.model.inputs[self.input_names[key]].shape
        except Exception as e:
            print("An exception occured in loading the model")

        return self.net

    def predict(self, left_image, right_image, head_pose_angles):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_dict = {self.input_names[0]: left_image, self.input_names[1]: right_image, 
        self.input_names[2]: head_pose_angles}
        self.net.infer(input_dict)

    def check_model(self, request_id=0):
        
        return self.net.requests[request_id].outputs

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        p_frame = cv2.resize(image, (self.net_input_shape[3], self.net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        return p_frame

    @staticmethod
    def preprocess_output(self, outputs, frame, confidence_level=0.5):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # coordinate vector
        return outputs[self.output_name], []
