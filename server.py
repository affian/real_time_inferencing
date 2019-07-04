# import the necessary packages
from fps import FPS
from webcam import WebcamVideoStream
from inferencehelper import InferenceHelper
from drawinghelper import DrawingHelper

import logging
import time
from queue import Queue

import cv2

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("Server")

# Spawning the relevant threads
logger.info("Starting webcam thread")
webcam_thread = WebcamVideoStream(src=1).start()
fps = FPS().start()

logger.info("Starting inference thread")
INFERENCE_API = ""
inference_thread = InferenceHelper(INFERENCE_API).start()

#logger.info("Starting drawing thread")
drawing_thread = DrawingHelper(webcam_thread.VID_WIDTH, webcam_thread.VID_HEIGHT).start()

# Run until quit.
while(True):
	frame = webcam_thread.read()
	frame_name = "./output/frame{}.jpg".format(fps.current_frame_number())

	# Queue up a new frame for inference every time the queue is empty.
	# This prevents the inference thread being queued with every single frame
	if inference_thread.queue.empty():
		inference_thread.enqueue({'name': frame_name, 'frame': frame, 'type': 'inference_queued'})

	# Everytime the inference thread processes a frame it returns the json_resp
	# This then adds the frame and json_resp to the drawing thread queue.
	drawing_thread.enqueue({'frame': frame, 'json_resp': inference_thread.json_resp})

	# The update function returns the drawn frame data which is then displayed on the screen.
	drawn_frame = drawing_thread.update()
	#new_frame = cv2.resize(drawn_frame, (812, 586)) #resize the frame
	cv2.imshow("Coloprog Nuclei Detection", drawn_frame) 
	font = cv2.FONT_HERSHEY_SIMPLEX
	
	
	# Press q to break the loop, and terminate the cv2 window.
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	elif cv2.waitKey(1) & 0xFF == ord('c'):
		# Press c to restart the drawing thread to clear the screen.
		drawing_thread = DrawingHelper(webcam_thread.VID_WIDTH, webcam_thread.VID_HEIGHT).stop()
		drawing_thread = DrawingHelper(webcam_thread.VID_WIDTH, webcam_thread.VID_HEIGHT).start()
 
	# update the FPS counter
		fps.update()

 
fps.stop()
logger.info("Camera FPS: {}".format(fps.end_to_end_fps()))

# Cleaning up the windows
webcam_thread.stop()
cv2.waitKey(0)
cv2.destroyAllWindows()
