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
    def __init__(self, net, threshold=0.60, args=None):
        
        self.network = net
        self.args = args
        # Get the input layer
        self.input_blob = next(iter(net.network.inputs))
        self.output_blob = next(iter(net.network.outputs))
            
    def load_model(self, net, num_requests=1):
        '''
        TODO: This method needs to be completed by you
        '''
        
        self.net_input_shape = net.network.inputs[self.input_blob].shape
        
    def predict(self, net, batch_images, request_id=0):
        '''
        TODO: This method needs to be completed by you
        '''
        for ii, p_frame in enumerate(batch_images):
            net.async_inference(p_frame, request_id=ii)
        for i in range(len(batch_images)):
            status = net.wait(request_id=i)

    def check_model(self, net, request_id=0):
        return net.exec_network.requests[request_id].outputs
        
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
    def preprocess_output(outputs, output_name, frame, confidence_level=0.5):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        try:
            keys = ['angle_y_fc', 'angle_p_fc', 'angle_r_fc']
            output_keys = ['yaw', 'pitch', 'roll']

            matrix = np.zeros((outputs[keys[0]].shape[1],3))
            for ii,k in enumerate(keys):
                print(outputs[k].shape)
                for idx in range(outputs[k].shape[1]):
                    matrix[idx,ii] = outputs[k][0,idx,0].item()

        except Exception as e:
            print(e.args)

        # result matrix
        return matrix, []