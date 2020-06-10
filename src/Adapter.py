import os
import cv2
import numpy as np
import sys

# adapter class object by inheritance
class Adapter:
    '''
    General Class for the Inference Models.
    '''

    def __init__(self, net, threshold=0.60, args=None):
        
        self.network = net
        self.args = args
        # Get the input layer
        self.input_blob = next(iter(net.network.inputs))
        self.output_blob = next(iter(net.network.outputs))
            
    def load_model(self, net):
        self.net_input_shape = net.network.inputs[self.input_blob].shape
        
    def predict(self, net, batch_images, request_id=0):
        for ii, p_frame in enumerate(batch_images):
            net.async_inference(p_frame, request_id=ii+request_id)

    def wait(self, net, length, request_id=0):
        statuses = []
        for i in range(length):
            status = net.wait(request_id=i+request_id)
            statuses.append(status)
        return statuses

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