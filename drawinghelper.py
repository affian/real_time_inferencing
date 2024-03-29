# import the necessary packages
import os
import time
from threading import Thread
import requests
import json
from collections import deque

import numpy as np
import logging
import cv2
import queue

logger = logging.getLogger("DrawingHelper")

class DrawingHelper:
    """ Manages the drawing on screen using the inferenced frames. """
    def __init__(self, VID_HEIGHT, VID_WIDTH):
        
        # Thread management and queueing
        self.stopped = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.queue = queue.Queue()

        # Video dimensions
        self.VID_HEIGHT = VID_HEIGHT
        self.VID_WIDTH = VID_WIDTH

        # Drawing constants
        # Possibly replace with a namespace?
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.2        
        self.THICKNESS = 1
        self.BB_THICKNESS = 1
        self.FRAME_THRESHOLD = 600  # This should be an even number.

        # The different font and bbox colours- dependent on cell type.
        self.FONT_COLOUR_BLUE = (255, 98, 0)
        self.FONT_COLOUR_RED = (74, 39, 186)
        self.FONT_COLOUR_YELLOW = (51, 184, 253)

        self.BB_COLOUR_BLUE = (255, 98, 0)   # IBM blue 
        self.BB_COLOUR_RED = (74, 39, 186)
        self.BB_COLOUR_YELLOW = (51, 184, 253)       

        #nuclei labels
        self.cell_type_a = 'epithelial'
        self.cell_type_b = 'fibroblast'
        self.cell_type_c = 'lymphocyte'

        #confidence threshold 
        self.confidence_thresh_a = 0.99
        self.confidence_thresh_b = 0.98
        self.confidence_thresh_c = 0.99
        

    def start(self):
        # start the thread to read frames from the queue
        self.thread.start()
        return self

    def enqueue(self, item):
        self.queue.put_nowait(item)

    def update(self):
        try:
            # keep looping infinitely until the thread is stopped
            while True:
                # if the thread indicator variable is set, stop the thread
                if self.stopped:
                    return

                try:
                    inferred_frame = self.queue.get(block=True)
                    json_resp = inferred_frame['json_resp']
                    frame_data = inferred_frame['frame']

                    if json_resp:
                        current_letter = json_resp[0]['label']
                    else:
                        current_letter = ''
                        json_resp = [{'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0, 'confidence': 0, 'label': ''}]

                    frame_data = self.draw_bounding_box(json_resp, frame_data)

                    return frame_data, json_resp

                except queue.Empty:
                    logger.debug("Slept and nothing to do... Trying again.")
                    time.sleep(1)
                    continue

        except Exception as e:
            logger.error("Exception occurred = {}".format(e))

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def draw_bounding_box(self, json_resp, image):
        
        for i in range(len(json_resp)):
            #apply confidence thresholds to json response then execute
            if (json_resp[i]['confidence'] >= self.confidence_thresh_a) & (json_resp[i]['confidence'] >= self.confidence_thresh_b) & (json_resp[i]['confidence'] >= self.confidence_thresh_c): 

                #print('CONFIDENCE:', json_resp[i]['confidence'])

                x_max = json_resp[i]['xmax'] 
                y_max = json_resp[i]['ymax'] 

                x_min = json_resp[i]['xmin'] 
                y_min = json_resp[i]['ymin'] 

                #assign bounding box colours to each class
                if json_resp[i]['label'] == 'epithelial':
                    display_BB_colour = self.BB_COLOUR_BLUE

                elif json_resp[i]['label'] == 'fibroblast':
                    display_BB_colour = self.BB_COLOUR_RED

                elif json_resp[i]['label'] == 'lymphocyte':
                    display_BB_colour = self.BB_COLOUR_YELLOW

                else:
                    return(0)

                cv2.rectangle(image, (x_max,y_max), (x_min, y_min), display_BB_colour, self.BB_THICKNESS)

                #put text label on bounding box
                if json_resp[i]['label'] == 'epithelial':
                    display_label = self.cell_type_a

                elif json_resp[i]['label'] == 'fibroblast':
                    display_label = self.cell_type_b

                elif json_resp[i]['label'] == 'lymphocyte': 
                    display_label = self.cell_type_c

                else:
                    return(0)

                #assign corresponding font colours to each class
                if json_resp[i]['label'] == 'epithelial':
                    font_colour = self.FONT_COLOUR_BLUE

                elif json_resp[i]['label'] == 'fibroblast':
                    font_colour = self.FONT_COLOUR_RED

                elif json_resp[i]['label'] == 'lymphocyte':
                    font_colour = self.FONT_COLOUR_YELLOW

                else:
                    return(0)


                cv2.putText(image, display_label, (x_min, y_min), self.FONT, 0.35, font_colour, lineType=cv2.LINE_AA)
        return image


