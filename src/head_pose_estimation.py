'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class HeadPoseEstimation:
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
            self.input_name = next(iter(self.model.inputs))
            self.net_input_shape = self.model.inputs[self.input_name].shape
        except Exception as e:
            print("An exception occured in loading the model")

        return self.net

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_dict = {self.input_name: image}
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
        keys = ['angle_y_fc', 'angle_p_fc', 'angle_r_fc']
        output_keys = ['yaw', 'pitch', 'roll']

        matrix = np.zeros((outputs[keys[0]].shape[1],3))
        for ii,k in enumerate(keys):
            for idx in range(outputs[k].shape[1]):
                matrix[idx,ii] = outputs[k][0,idx,0].item()

        # result matrix
        return matrix, []