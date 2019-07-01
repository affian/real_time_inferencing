# import the necessary packages
from threading import Thread
import cv2
import logging

logger = logging.getLogger("WebcamVideoStream")

class WebcamVideoStream:
    def __init__(self, src=1):
        # Initialise the video camera stream and the first frame from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # Initialise the variable used to indicate if the thread should be stopped
        self.stopped = False

        # Video dimensions
        self.VID_HEIGHT = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.VID_WIDTH = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        logger.debug(f"Video dimensions: {self.VID_HEIGHT} * {self.VID_WIDTH}")

    def start(self):
        """
        Start the thread to read frames from the video stream
        """
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        """
        Update the stream with the next frame from the video file
        """
        while True:
            if self.stopped:
                return
            
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        """
        Returns: 
            self.frame: The last frame which has been read from the stream.
        """
        return self.frame
    
    def stop(self):
        """
        Stop the thread
        """
        self.stopped = True