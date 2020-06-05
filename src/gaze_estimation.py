'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
import numpy as np
from collections import OrderedDict
from openvino.inference_engine import IENetwork, IECore

class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, net, threshold=0.60, args=None):
        
        self.network = net
        self.args = args
        # Get the input layer
        self.input_blob = list(iter(net.network.inputs))
        self.output_blob = next(iter(net.network.outputs))
            
    def load_model(self, net, num_requests=1):
        '''
        TODO: This method needs to be completed by you
        '''
        
        self.net_input_shape = OrderedDict()
        for name in self.input_blob:
            self.net_input_shape[name] = net.network.inputs[name].shape
        
    def predict(self, net, batch_images, request_id=0):
        '''
        TODO: This method needs to be completed by you
        '''
        for i in range(len(batch_images)):
            status = net.requests[i].wait(-1)

    def check_model(self, net, request_id=0):
        
        return net.network.requests[request_id].outputs

    def preprocess(self, image, name):
        p_frame = cv2.resize(image, (self.net_input_shape[name][3], self.net_input_shape[name][2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        return p_frame
        
    def preprocess_input(self, frame=None, vector=None):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        result = OrderedDict()
        for ii, name in enumerate(self.input_blob):
            if ii == 0:
                result[name] = vector
            else:
                result[name] = self.preprocess(frame, name)

        return result


    @staticmethod
    def preprocess_output(outputs, output_name, frame, confidence_level=0.5):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # coordinate vector
        return outputs[output_name], []
