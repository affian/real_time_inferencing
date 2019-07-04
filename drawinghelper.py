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
        self.FONT_SCALE = 1        
        self.THICKNESS = 1
        self.FONT_COLOUR = (255,255,255)
        self.BB_COLOUR = (255, 98, 0)   # IBM blue 
        self.BB_THICKNESS = 3
        self.RECTANGLE_BACKGROUND = (0, 0, 0)
        self.TEXT_HEIGHT = 75
        self.TEXT_WIDTH = int(self.VID_HEIGHT*0.95)
        self.WIDTH_FACTOR = 5

        # Letter tracking constants
        # NOTE: Alot of these values are chosen through trial and error for
        # a nice looking visual effect. Do not read too much into the values.
        self.FRAME_THRESHOLD = 60  # This should be an even number.
        self.WIDE_LETTER_LIST = ["g", "j", "k", "m", "u"]
        self.WIDE_LETTER_FACTOR = 11
        self.SLIM_LETTER_FACTOR = 7
        self.MAX_LETTERS_TO_DISPLAY = 20

        # Letter tracking variables
        self.letter_deque = deque(maxlen=self.FRAME_THRESHOLD)
        self.constant_letter_count = 0
        self.count_slim_letter = 0
        self.count_wide_letter = 0
        self.constant_letter_dict = {}

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

                    self.draw_bounding_box(json_resp, frame_data)

                    self.draw_letter(current_letter, frame_data)

                    self.letter_deque.append(current_letter)
                    shortened_letter_list = list(self.letter_deque)
                
                    unique_letters = []
                    letter_count_dict = {}

                    for letter in shortened_letter_list:
                        if letter not in unique_letters:
                            unique_letters.append(letter)

                        # Maps the letters which have appeared in the last self.FRAME_THRESHOLD number of frames
                        # to their respective counts.  e.g. {"a": 11, "b": 1} over the past 12 frames.
                        letter_count_dict[letter] = shortened_letter_list.count(letter)

                    value_count_list = []
                    for _, val in letter_count_dict.items():
                        # Equivalent to the letter_count_dict but with only the values.
                        value_count_list.append(val)

                    # If there are two unique letters that have been inferred recently we need to decide which
                    # one to display. Therefore we count the number of instances of each letter and select the
                    # one that appears most frequently.
                    if len(unique_letters) == 2:
                        for i in range(len(shortened_letter_list)):
                            shortened_letter_list[i] = max(letter_count_dict.keys(), key=(lambda key: letter_count_dict[key]))

                    # TODO: FIGURE OUT WHAT THE FUCK THIS DOES.
                    if ((len(value_count_list) == 1) or (len(value_count_list) == 2 and value_count_list[1] == 1 or value_count_list[0] == 1)
                        or (len(value_count_list) == 2 and value_count_list[0] > (self.FRAME_THRESHOLD // 2) or value_count_list[1] > ( self.FRAME_THRESHOLD // 2))):
                        self.constant_letter_count += 1
                    else:
                        self.constant_letter_count = 0

                    if len(unique_letters) < 2 and self.constant_letter_count == self.FRAME_THRESHOLD:
                        if unique_letters[0] in self.WIDE_LETTER_LIST:
                            self.count_slim_letter += 1
                            wide_dict_key = str(self.count_wide_letter * self.WIDE_LETTER_FACTOR + self.count_slim_letter * self.SLIM_LETTER_FACTOR)
                            self.constant_letter_dict[wide_dict_key] = unique_letters[0]
                        else:
                            self.count_wide_letter += 1
                            slim_dict_key = str(self.count_wide_letter * self.WIDE_LETTER_FACTOR + self.count_slim_letter * self.SLIM_LETTER_FACTOR)
                            self.constant_letter_dict[slim_dict_key] = unique_letters[0]

                    self.draw_constant_letters(self.constant_letter_dict, frame_data)
                    return frame_data

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
        x_max = json_resp[0]['xmax']
        y_max = json_resp[0]['ymax']

        x_min = json_resp[0]['xmin']
        y_min = json_resp[0]['ymin']

        cv2.rectangle(image, (x_max,y_max), (x_min, y_min), self.BB_COLOUR, self.BB_THICKNESS)
	
    def draw_letter(self, label, image):
        # Coords for drawing
        drawing_coords = (int(0.5 * self.VID_HEIGHT), int(0.8 * self.VID_WIDTH))
        cv2.putText(image, label, drawing_coords, self.FONT, 2.0, self.FONT_COLOUR, 2, cv2.LINE_AA)

    def draw_constant_letters(self, letter_dict, image):    
        # define our empty black box to put our spelt letters in at bottom of screen
        rect_top_left = (int(0.1 * self.VID_HEIGHT), int(0.85 * self.VID_WIDTH))
        rect_bottom_right = (int(0.9 * self.VID_HEIGHT), int(0.95 * self.VID_WIDTH))
        cv2.rectangle(image, rect_top_left, rect_bottom_right, self.RECTANGLE_BACKGROUND, cv2.FILLED)

        x_start = int(0.1 * self.VID_HEIGHT)
        y_start = int(self.VID_WIDTH - 56)
    
        for key, letter in letter_dict.items():
            x_position = int(x_start + np.int64(key) * self.WIDTH_FACTOR)
            cv2.putText(image, letter, (x_position, y_start), self.FONT, 2.0, self.FONT_COLOUR, 2, cv2.LINE_AA)
