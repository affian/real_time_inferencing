# import the necessary packages
import os
import time
from threading import Thread
import requests
import json

import logging
import cv2
import queue

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("InferenceHelper")

# common items for PowerAI Vision
POWERAI_BASEURL = "http://9.196.150.55:8009/inference" #8008 for asl

# TODO: Should figure out how to generate a CSRF token for PAIV endpoint
# cookiejar = {'x-auth-token': '6d047c1d-f056-4af3-9c83-f6906e570c30',
#              'csrftoken': 'hrjBGsgUnv23OSCnAVKdwc07mJ2IpoIJ'
#              }

class InferenceHelper:
    """
    This helper class queues frames from the webcam thread, and sends them to the 
    PAIV backend to be classified. 
    """

    def __init__(self, API_ID):
        """
        Args:
            API_ID: API endpoint for the inference model
        """

        logger.info(f"Configuring new inference helper for API {API_ID}")

        # initialize the variable used to indicate if the thread should be stopped
        self.stopped = False
        self.API_ID = API_ID
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

        # The queue receives frames from the webcam.
        self.queue = queue.Queue()

        # Initial placeholder JSON response
        self.json_resp = [{'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0, 'confidence': 0, 'label': ''}]

    def start(self):
        """
        Start the thread to read frames from the queue
        """
        self.thread.start()
        return self

    def infer_image(self, frame):
        """
        Send a frame for inference at the backend. 
        """
        endpoint = POWERAI_BASEURL + self.API_ID
        file = open(frame, 'rb')
        myfiles = {'files': (frame, file)}

        status_code = 0
        retry_count = 0
        resp_value = dict()
        objs = None

        while (status_code != 200) and (retry_count < 5) and (objs is None):
            try:
                if retry_count != 0:
                    logger.warning(f"retrying upload for {frame}, attempt {retry_count}")

                request = requests.post(endpoint, files=myfiles, verify=False)
                status_code = request.status_code
                retry_count = retry_count + 1
                file.close()
                resp_value = json.loads(request.text)

                if 'classified' in resp_value.keys():
                    objs = resp_value['classified']
            
            except Exception as e:
                logger.exception("Caught exception during inference API call.")
                pass
        
        if objs is None:
            objs = []

        return status_code, objs

    def enqueue(self, item):
        self.queue.put_nowait(item)

    def update(self):
        try:
            # keep looping infinitely until the thread is stopped
            while True:
                # if the thread indicator variable is set, stop the thread
                if self.stopped:
                    return
                #wait forever for something to act on

                try:
                    frame_data = self.queue.get(block=True) #timeout in seconds
                except queue.Empty:
                    logger.debug("Slept and nothing to do... Trying again.")
                    time.sleep(1)
                    continue

                # write frame to JPEG- used by infer_image
                cv2.imwrite(frame_data['name'], frame_data['frame'])
                (status_code, json_resp) = self.infer_image(frame_data['name'])

                if status_code == 200:
                    # TODO: Clean this up not sure I'm using the frame_data object outside of the json_resp.
                    frame_data['json_resp'] = json_resp
                    frame_data['type'] = 'inference_done' #this makes it officially ready for processing
                    logger.debug(f"NEW INFERENCE FOR FRAME {frame_data['name']}, status code = {status_code}, json = \"{json_resp}\"")
                    
                    # Set the json_resp parameter which will be used to draw on the frames in the main thread.
                    self.json_resp = json_resp

                #tidy up that file we wrote...
                try:
                    os.remove(frame_data['name'])
                except:
                    pass

        except Exception as e:
            logger.error(f"Exception occurred = {e}")

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
