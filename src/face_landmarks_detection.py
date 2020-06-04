'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
import numpy as np

class FaceLandMarksDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, net, threshold=0.60, args=None):
        
        self.network = net
        self.args = args
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

    def load_model(self, num_requests=1):
        '''
        TODO: This method needs to be completed by you
        '''
        
        self.net_input_shape = self.network.inputs[self.input_blob].shape
        
    def predict(self, net, batch_images, request_id=0):
        '''
        TODO: This method needs to be completed by you
        '''
        for i in range(len(batch_images)):
            status = net.requests[i].wait(-1)

    def check_model(self, request_id=0):
        
        return self.network.requests[request_id].outputs

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
            height, width = frame.shape[:2]
            if len(outputs) > 0:
                for res in outputs:
                    left_eye_x, left_eye_y, 
                    right_eye_x, right_eye_y, nose_x, nose_y, 
                    left_lip_x, left_lip_y, right_lip_x, right_lip_y = res
                    
                    boxes.append([(left_eye_x, left_eye_y), (right_eye_x, right_eye_y), 
                    (nose_x, nose_y), (left_lip_x, left_lip_y), (right_lip_x, right_lip_y)])
        except Exception as e:
            print(e.args)

        # coordinate points
        return boxes, []
