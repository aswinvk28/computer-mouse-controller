'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
import numpy as np
import sys
sys.path.append("../")

# class objects created by delegation
class FaceDetection:
    '''
    Class for the Person Detection Model.
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
        for i in range(len(batch_images)):
            status = net.requests[i].wait(-1)

    def check_model(self, net, request_id=0):
        
        return net.network.requests[request_id].outputs
        
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
