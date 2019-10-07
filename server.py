# import the necessary packages
from fps import FPS
from webcam import WebcamVideoStream
from inferencehelper import InferenceHelper
from drawinghelper import DrawingHelper
import numpy as np
import time
import logging
import time
from queue import Queue

import cv2


def similarity_factor(sim_list):
	sim_np = np.subtract(sim_list[0],sim_list[1])
	sim_np = np.absolute(sim_np)
	return np.mean(sim_np)

def save_file(frame):
	cv2.iwrite()
	pass



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

# Read and inference the first image
frame = webcam_thread.read()
frame_name = "./output/frame{}.jpg".format(fps.current_frame_number())
similarity_list = [frame]

if inference_thread.queue.empty():
	inference_thread.enqueue({'name': frame_name, 'frame': frame, 'type': 'inference_queued'})
		
drawing_thread.enqueue({'frame': frame, 'json_resp': inference_thread.json_resp})
drawn_frame, json_resp = drawing_thread.update()
i =0

# Run until quit.
while(True):
	i+=1
	frame = webcam_thread.read()
	frame_name = "./output/frame{}.jpg".format(fps.current_frame_number())
	similarity_list.append(frame)

	time.sleep(0.2)

	similarity = similarity_factor(similarity_list)
	#print(fps.current_frame_number(), similarity)
	# cv2.imwrite(frame_name, frame)
	
	# if similarity > 60 & i > 50:
	# 	#print("similarity", similarity)
	# 	print("Frame", fps.current_frame_number())
	# 	break

	#print(sim_mean_np)
	del similarity_list[0]
	
	if similarity < 102: # Check if image is stable.
		drawn_frame = drawing_thread.draw_bounding_box(json_resp, frame)
		new_frame = cv2.resize(drawn_frame, (1285, 690))
		#print(similarity)
	else: #if not, send image for inferencing
		#print(similarity)
		#time.sleep(0.1)
		#frame = webcam_thread.read()
		if inference_thread.queue.empty():
			inference_thread.enqueue({'name': frame_name, 'frame': frame, 'type': 'inference_queued'})
		
		drawing_thread.enqueue({'frame': frame, 'json_resp': inference_thread.json_resp})

		drawn_frame, json_resp = drawing_thread.update()

		new_frame = cv2.resize(drawn_frame, (1285, 690))



	cv2.imshow("ColoProg Nuclei Detection", new_frame) 
	font = cv2.FONT_HERSHEY_SIMPLEX
	fps.update()
	
	
	# Press q to break the loop, and terminate the cv2 window.
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

fps.stop()
logger.info("Camera FPS: {}".format(fps.end_to_end_fps()))

# Cleaning up the windows
webcam_thread.stop()
cv2.waitKey(0)
cv2.destroyAllWindows()
