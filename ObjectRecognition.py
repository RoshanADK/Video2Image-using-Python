

import os   # portable way of using OS dependent functionality
import logging   #  implements a flexible error logging system for applications.
import logging.handlers
import random

import numpy as numpy
import skvideo.io  # using a FFmpeg/LibAV backend to read and write videos and to parse metadata from videos.
import cv2
import matplotlib.pyplot as plt 

import utils  # User Defined Functionalities is available here 

# Github Issue #6078 to close OpenCL to compute images later
cv2.ocl.setUseOpenCL(False) # apparently it provides interaction between the Python and C++ environments (i.e. the cv2.cpp file).
random.seed(123)

# ================================================
#      FILE SOURCES
IMAGE_DIR = "./out"
VIDEO_SOURCE =  "input.mp4"
SHAPE = (720, 1280)  # Height x Width 
# ================================================

def train_bg_subtractor(inst, cap, num=500): # Inst =  self
	# BG subtractor need process some amount of frames to give result
	print("Training BG subtractor....")

	i = 0 
	for frame in cap:
		inst.apply(frame, None, 0.001) # self.apply equivalent 
		i += 1
		if i >= num:            # Taking only num(500) frames and processing them 
			return cap

def main():

	log = logging.getLogger("Main")   # Event Logger and handling the main function
	# Log halnu ko arko Kaam , for Debugging use log.debug , log.info for giving status etc so log saves events for future cross checking

	# creating MOG(Mixture of Gaussians) background subtractor with 500 frames in cache
	# Plus lets make some shadow Detection too

	bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500,detectShadows=True)

	# History -  Length of history i.e Number of frames affecting background
	# Detect Shadows - Trade off Speed but use only required


	# Setting up the image source , we can use cv2 also , lets use skvideo

	cap = skvideo.io.vread(VIDEO_SOURCE) # We can tweak size of frame and  color/blackNwhite using options #skvideo.io.vread is for ndarray

	# Lets skip 500 frames to train the bg subtractor


	# Training Data
	train_bg_subtractor(bg_subtractor, cap, num=500)

	# Kinda Exception Handling
	frame_number = -1

	for frame in cap:
		if not frame.any():
			log.error("Frame Capture Failed, Stopping...")
			break
		frame_number += 1
		# Now lets save each frame in a folder with below Naming Convention 
		utils.save_frame(frame, "./out/frame_%04d.png" % frame_number)

		# USE MASKING
		fg_mask =  bg_subtractor.apply(frame, None, 0.001)


		# Save the Masked Frame too in the same folder
		utils.save_frame(frame, "./out/fg_mask_%04d.png" % frame_number)


# Main Starts here ------------------------------------------------------------------

if __name__ == "__main__":
	log = utils.init_logging()

	if not os.path.exists(IMAGE_DIR):
		log.debug("Creating image directory '%s'...", IMAGE_DIR)
		os.makedirs(IMAGE_DIR)

	main()






    